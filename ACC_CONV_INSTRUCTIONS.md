# acc_conv Scheduler Refactor Notes

This document records the software-side direction for the next refactor.  The
new control-plane standard is the generated scheduler interface under
`Scheduler_generater/`.  Do not design new code around the older hand-written
`lml_acc_conv.hpp` instruction helpers except as a temporary compatibility
layer for the currently running bitstream.

## Authoritative Sources

Use these files as the source of truth:

```text
../scheduler_gen/spec/conv_acc_control.json
Scheduler_generater/gen_ref/scheduler_control.hpp
Scheduler_generater/gen_ref/scheduler_core.sv
Scheduler_generater/gen_ref/scheduler_wrapper.sv
```

The intended C++ API is the generated namespace `scheduler`, especially:

```cpp
scheduler::Instructions
scheduler::FpgaScheduler
```

The application-level tensor operator should build programs through generated
methods such as:

```cpp
ins.config_conv_u_cfg(cin_mask, cout_mask);
ins.config_conv_u_kernel0(kernel0);
...
ins.launch_read_u_to_ipp(base, c_stride, row_stride, c, h, w);
ins.launch_read_u_to_opp(base, c_stride, row_stride, c, h, w);
ins.launch_conv_u_run(h, w);
ins.launch_write_back_u_run(base, c_stride, row_stride, c, h, w);
ins.config_write_back_u_q_cfg(q_multiplier, q_shift, q_zero_point);
ins.sync();
ins.nop();
```

## Scheduler Model

The generator models transaction-level modules and shared resources.  It does
not generate datapath muxes, FIFO backpressure, or ping-pong select/exchange
logic.

Current spec structure:

```text
System:
  Nop
  Sync, depends on read_u, conv_u, write_back_u

Pingpongs:
  ipp
  opp

Modules:
  read_u
  conv_u
  write_back_u
```

Resource rule:

```text
Using means the module occupies that resource until ap_done.
Deps means the command waits until the listed module/resource is idle.
```

Important boundary:

```text
ipp/opp exchange is still owned by the downstream datapath/control RTL.
The scheduler only knows that a command uses ipp or opp as a resource.
Software should not assume the scheduler toggles ping-pong select signals.
```

## Generated Instruction Format

The generated scheduler uses 128-bit instructions:

```text
instruction[127:120] = 8-bit global opcode
instruction[119:0]   = payload
```

The generated C++ type is:

```cpp
namespace scheduler {
struct Instruction {
    std::uint64_t hi;
    std::uint64_t lo;
};
}
```

The generated C++ API owns field packing.  New application code should not pack
opcodes or bit fields by hand.

## Current Generated Commands

From `../scheduler_gen/spec/conv_acc_control.json`:

```text
read_u.to_ipp(base_addr, c_offset, h_offset, c, h, w)
  Using: ipp

read_u.to_opp(base_addr, c_offset, h_offset, c, h, w)
  Using: opp

conv_u.run(h, w)
  Deps: ipp
  Using: opp

write_back_u.run(base_addr, c_offset, h_offset, c, h, w)
  Deps: opp
```

Configuration registers:

```text
conv_u.cfg(cin_mask, cout_mask)
conv_u.kernel0 ... conv_u.kernel15, each 72 bits
write_back_u.q_cfg(q_multiplier, q_shift, q_zero_point)
```

A future code path should express a convolution tile with this semantic order:

```cpp
ins.config_conv_u_cfg(cin_mask, cout_mask);
ins.config_conv_u_kernel0(...);
...
ins.config_write_back_u_q_cfg(1, 8, 0);

ins.launch_read_u_to_ipp(input_phys, input_c_stride, input_row_stride,
                         channels, input_h, input_w);
ins.launch_conv_u_run(input_h, input_w);
ins.launch_write_back_u_run(output_phys, output_c_stride, output_row_stride,
                            channels, output_h, output_w);
ins.sync();
ins.nop();
```

If the hardware datapath requires loading bias/psum into `opp` before compute,
that should be represented as a `read_u.to_opp(...)` transaction in the
generated model, not as a special hand-packed legacy opcode.

## Tensor Operator Direction

The next software abstraction should be a tensor-level convolution operator,
not scattered ad hoc test code.  Suggested ownership:

```text
TensorConv3x3Operator
  - owns tile planning
  - owns instruction-buffer construction
  - owns FPGA workspace packing/unpacking
  - owns CPU fallback for uncovered edge regions
  - exposes one run() entry point for a whole NCHW tensor layer
```

The operator input should be shape-oriented:

```cpp
struct Conv3x3TensorArgs {
    const uint8_t* input;
    uint8_t* output;
    const int8_t* weights;
    uint32_t in_h;
    uint32_t in_w;
    uint32_t cin;
    uint32_t cout;
};
```

The operator should hide fixed hardware tile details from callers.  For the
current fixed-tile bitstream, full FPGA tiles are based on:

```text
input tile  : 32 x 32
output tile : 30 x 30
channel group: 4 input channels x 4 output channels
```

The operator should automatically split a layer into:

```text
FPGA-covered region:
  output tiles where a full 30x30 output tile fits.

CPU edge region:
  right and bottom output regions that are not divisible by 30.
  any channel leftovers not representable by the 4-channel hardware group.
```

CPU edge handling should be correct first.  It can reuse the existing C++
reference implementation in `yolo_conv3x3.cpp` or a focused region helper.

## Async Execution Policy

For a tensor layer, prefer this schedule:

```text
1. Build and flush the FPGA instruction stream for all full tiles.
2. Start FPGA execution.
3. While FPGA is running, compute uncovered edge regions on CPU.
4. Wait for FPGA done.
5. Invalidate/copy FPGA output tiles into the final output tensor.
6. Compare or continue to the next layer.
```

The application should use the generated scheduler style:

```cpp
scheduler::FpgaScheduler fpga(ctrl_base);
fpga.run(ins);

compute_cpu_edges(...);

while (fpga.is_busy()) {
}
```

For the current legacy control wrapper, the same policy can be temporarily
implemented with `lml::accelerator::start(ins)` followed later by
`wait_done()`.  Keep that as a compatibility bridge only.

## Legacy Compatibility Boundary

The existing `lml_acc_conv.hpp` path uses old hand-packed helper names such as:

```cpp
lml::instr::bias(...)
lml::instr::read_conv(...)
lml::instr::write_axi(...)
lml::instr::regs(...)
```

This path is allowed only to keep current board tests running while the RTL and
application move to `Scheduler_generater`.  New documentation, new tensor-layer
code, and future refactors should use the generated scheduler vocabulary:

```text
read_u.to_ipp
read_u.to_opp
conv_u.run
write_back_u.run
conv_u.cfg/kernelN
write_back_u.q_cfg
```

Do not add new user-facing abstractions around legacy opcodes.

## Refactor Checklist

1. Keep `../scheduler_gen/spec/conv_acc_control.json`
   as the first file to update when command/register semantics change.
2. Regenerate `Scheduler_generater/gen_ref/scheduler_control.hpp` after spec
   changes.
3. Make C++ tile builders call generated methods, not manual bit packing.
4. Keep ipp/opp select/exchange in downstream RTL; only resource occupancy goes
   into the scheduler spec.
5. Move whole-layer splitting into a tensor operator.
6. Run FPGA full tiles asynchronously and cover non-divisible edges on CPU.
7. Wait for FPGA done before consuming or comparing FPGA-covered output tiles.
