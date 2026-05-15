# Scheduler Control Generator Usage Manual

This document describes how to use the current JSON-to-scheduler generator in
another FPGA project.  The current version intentionally keeps the interface
small and explicit so that both humans and AI tools can generate a valid
control specification.

For the short user-facing C++ API style, see
[`docs/scheduler_cpp_api.md`](scheduler_cpp_api.md).

## Scope

The generator targets transaction-level PL modules.  Each scheduled module is
expected to look like an HLS-style block with:

```verilog
ap_start
ap_ready
ap_done
ap_idle
```

The scheduler is responsible for:

- fetching 128-bit instructions from an external FIFO-like interface,
- checking module/resource dependencies before issuing a command,
- holding `ap_start` until the target module accepts it with `ap_ready`,
- latching per-module command arguments,
- latching generated configuration registers,
- exposing a small AXI-Lite control/status interface.

The scheduler is not intended to replace FIFO backpressure networks or a
streaming datapath scheduler.  Streaming details, ping-pong select signals, and
datapath muxes should remain in user RTL.

## Quick Start

Validate a JSON spec without writing generated files:

```bash
python3 scripts/gen_control.py --check-only experiments/scheduler_gen/spec/conv_acc_control.json
```

Generate RTL and the host-side C++ control header:

```bash
python3 scripts/gen_control.py \
  experiments/scheduler_gen/spec/conv_acc_control.json \
  experiments/scheduler_gen/gen_ref
```

Generated files:

```text
experiments/scheduler_gen/gen_ref/
  scheduler_core.sv
  scheduler_wrapper.sv
  scheduler_ctrl_axi_lite.sv
  scheduler_control.hpp
```

Compile-check the generated RTL:

```bash
iverilog -g2012 -o /tmp/scheduler_gen_ref.vvp \
  experiments/scheduler_gen/rtl/soft_reset.sv \
  experiments/scheduler_gen/gen_ref/scheduler_ctrl_axi_lite.sv \
  experiments/scheduler_gen/gen_ref/scheduler_core.sv \
  experiments/scheduler_gen/gen_ref/scheduler_wrapper.sv
```

Compile-check the generated C++ header:

```bash
g++ -std=c++14 -fsyntax-only -x c++ experiments/scheduler_gen/gen_ref/scheduler_control.hpp
```

The default template JSON used by `--emit-template` lives next to the script:

```text
scripts/template/conv_acc_control.template.json
```

## Instruction Format

The current generator supports only 128-bit instructions.

```text
instruction[127:120] = 8-bit global opcode
instruction[119:0]   = payload
```

The default JSON therefore uses:

```json
{
  "InstructionWidth": 128,
  "OpcodeWidth": 8,
  "RegAddrWidth": 8,
  "PayloadWidth": 120
}
```

Do not use 64-bit instructions in this version.  Many useful commands and
register writes need more than 64 bits once the opcode, register address, and
payload fields are included.  A 128-bit instruction keeps each command
self-contained and avoids multi-word decoder state.

## JSON Structure

Top-level structure:

```json
{
  "InstructionWidth": 128,
  "OpcodeWidth": 8,
  "RegAddrWidth": 8,
  "PayloadWidth": 120,
  "System": {},
  "Pingpongs": {},
  "Modules": {}
}
```

### System

`System` defines global instructions that do not start a module.

Example:

```json
"System": {
  "Nop": {},
  "Sync": {
    "Deps": ["read_u", "conv_u", "write_back_u"]
  }
}
```

`Deps` lists modules or resources that must be idle before the instruction can
issue.  A `Sync` instruction usually depends on all running modules.

### Pingpongs

`Pingpongs` declares named shared resources.  The scheduler only models their
busy/idle state.  It does not generate ping-pong select logic or mux logic.

Example:

```json
"Pingpongs": {
  "ipp": {},
  "opp": {}
}
```

Use normal resource names such as `ipp`, `opp`, `feature_buf`, or `weight_buf`.

### Modules

Each module has optional `Command` and `Regs` sections.

Example:

```json
"read_u": {
  "CommandOpcodeWidth": 1,
  "Command": {
    "to_ipp": {
      "Args": [
        {"Name": "base_addr", "Width": 32, "Default": 0},
        {"Name": "length", "Width": 32}
      ],
      "Using": ["ipp"]
    },
    "to_opp": {
      "Args": [
        {"Name": "base_addr", "Width": 32, "Default": 0},
        {"Name": "length", "Width": 32}
      ],
      "Using": ["opp"]
    }
  },
  "Regs": {
    "cfg": [
      {"Name": "base_addr", "Width": 32, "Default": 0},
      {"Name": "length", "Width": 32}
    ]
  }
}
```

If `CommandOpcodeWidth` is omitted, the generator infers the minimum width
needed for the command count.  Keeping it explicit can make generated instruction
layouts easier to read during debugging.

Field defaults may be plain integers or Verilog-style literal strings.  Use the
string form when you want to preserve an exact HDL-style reset value in JSON.

字段默认值既可以是普通整数，也可以是 Verilog 风格字面量字符串。若希望在 JSON
里保留精确的 HDL reset 值，就使用字符串形式。

## Field Packing Rules

Command payload:

```text
payload[119 : 120-CommandOpcodeWidth] = module-local command id
remaining high-to-low bits             = Args fields in JSON order
unused low bits                        = zero or ignored
```

Register payload:

```text
payload[119:112] = global register address
payload[111:...] = register fields in JSON order
unused low bits  = zero or ignored
```

Field order matters.  The first field in JSON occupies the highest available
payload bits after the command id or register address.

## Dependency Semantics

The generator uses two kinds of runtime state:

- module busy bits,
- resource busy bits.

`Deps` means the command waits for the listed modules/resources to become idle.

`Using` means the command occupies the listed resources after issue.  The
resources are released when the issued module asserts `ap_done`.

Recommended modeling rule:

- writer-side use of a ping-pong/shared buffer goes in `Using`,
- reader-side ordering usually goes in `Deps` only,
- do not ask the scheduler to generate ping-pong exchange/select signals.

Example:

```json
"read_u": {
  "Command": {
    "to_ipp": {
      "Args": [
        {"Name": "base_addr", "Width": 32},
        {"Name": "length", "Width": 32}
      ],
      "Using": ["ipp"]
    }
  }
}
```

This means `read_u.to_ipp` can issue only when `read_u` and `ipp` are idle, and
it holds `ipp` busy until `read_u_ap_done`.

## Generated RTL Interface

Use `scheduler_wrapper.sv` as the top-level generated block.

It exposes:

- AXI-Lite slave ports,
- instruction FIFO control ports,
- one HLS-style interface per module,
- generated command/config output registers.

Typical ports:

```verilog
// instruction FIFO
output logic         ins_rd_start,
input  logic         ins_rd_done,
output logic         ins_fifo_rd_en,
input  logic         ins_fifo_n_empty,
input  logic [127:0] ins_fifo_data,

// generated module interface
output logic         conv_u_ap_start,
input  logic         conv_u_ap_ready,
input  logic         conv_u_ap_done,
input  logic         conv_u_ap_idle,
output logic [0:0]   conv_u_cmd_id,
output logic [15:0]  conv_u_cmd_args,
output logic [49:0]  conv_u_cfg
```

The wrapper holds `ap_start` high until:

```verilog
ap_start && ap_ready
```

Then it clears `ap_start`.  This supports HLS modules whose `ap_ready` may be
delayed for several cycles.

## AXI-Lite Control Register Protocol

The generated wrapper instantiates `scheduler_ctrl_axi_lite.sv`.

Current register convention:

```text
reg0[7:0]   = software write-one request bits, hardware clears completed bits
reg0[15:8]  = software configuration bits
reg0[31:16] = hardware status bits
reg1        = software config register
reg2        = software config register
reg3        = spare
```

Currently used request bits:

```text
bit 0 = soft reset request
bit 1 = start instruction fetch/program execution request
bit 2 = clear IRQ request
```

Currently used config/status bits:

```text
reg0[8]     = enable IRQ on program done
reg0[31:24] = hardware version
```

The generated `scheduler_control.hpp` provides a fixed-capacity instruction
builder and an FPGA scheduler object.  See
[`docs/scheduler_cpp_api.md`](scheduler_cpp_api.md) for the user-facing C++ API.

## Validation Errors

The generator checks common mistakes before writing RTL:

- missing required top-level keys,
- non-128-bit instruction width,
- invalid Verilog identifiers,
- duplicate fields in one command/register,
- command id width too small,
- command id plus args exceeding payload width,
- register address plus fields exceeding payload width,
- too many global register groups for `RegAddrWidth`,
- unknown `Using` resources,
- unknown `Deps` modules/resources,
- default values that do not fit field width.

Example:

```bash
python3 scripts/gen_control.py --check-only my_control.json
```

If validation fails, fix the JSON first.  Do not edit generated RTL directly.

## AI-Friendly Spec Writing Rules

When using an AI assistant to create or modify a spec, give it these rules
explicitly:

```text
Generate a JSON control spec for scripts/gen_control.py.
Use only 128-bit instructions:
InstructionWidth=128, OpcodeWidth=8, RegAddrWidth=8, PayloadWidth=120.
Use valid Verilog identifiers for all module, command, register, and field names.
Put shared buffer names in Pingpongs.
Each command belongs to exactly one module.
Use Args for per-command runtime arguments.
Use Regs for persistent configuration registers.
Use Using only for resources occupied by the command until ap_done.
Use Deps for modules/resources that must be idle before issue.
Do not add manual global opcodes.
Do not add ping-pong select or mux fields unless they are real module arguments.
Keep every command payload within 120 bits.
Keep every register payload within 120 bits including the 8-bit register address.
Return strict JSON only.
```

A useful AI prompt template:

```text
I have HLS-style modules with ap_start/ap_ready/ap_done/ap_idle.
Create a scheduler generator JSON spec.

Modules:
- read_u: commands ...
- conv_u: commands ...
- write_back_u: commands ...

Shared resources:
- ipp
- opp

Rules:
- 128-bit instruction only.
- Command Args are packed after module-local command id.
- Regs are persistent configuration registers.
- Using means write-side resource occupation until module done.
- Reader-side ordering should use Deps when needed.
- Return strict JSON only, no explanation.
```

After the AI returns JSON, always run:

```bash
python3 scripts/gen_control.py --check-only path/to/spec.json
```

Then inspect generated `scheduler_wrapper.sv` ports before integrating.

## Integration Checklist

1. Write the JSON spec.
2. Run `--check-only`.
3. Generate into a clean output directory.
4. Add these generated files to the RTL project:
   - `scheduler_core.sv`
   - `scheduler_wrapper.sv`
   - `scheduler_ctrl_axi_lite.sv`
   - `soft_reset.sv`
5. Connect AXI-Lite slave ports to the processor interconnect.
6. Connect the instruction FIFO interface.
7. Connect each generated module interface to the real HLS/datapath module.
8. Use generated command/config outputs inside user RTL.
9. Use `scheduler_control.hpp` to build instructions and run the scheduler from
   the processor side.
10. Do not manually edit generated RTL; change JSON and regenerate.

## Current Limitations

- Only 128-bit instructions are supported.
- Generated RTL is SystemVerilog for now.
- The AXI-Lite control block is copied from the current project template.
- The scheduler models resource occupancy but does not generate datapath muxes,
  ping-pong select toggles, or buffer exchange logic.
- The generated host header builds instruction words only.  It does not include
  platform-specific AXI-Lite or DMA/FIFO drivers.
- This version assumes each command starts one module instance.
