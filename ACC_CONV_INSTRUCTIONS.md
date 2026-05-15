# acc_conv Instruction Construction Notes

This document is for agents editing `~/petalinux_application/src/main.cpp`.
It describes the current RTL protocol and the instruction stream that should be used for board tests.

## MMIO Register Map

The control registers are mapped at `0x4000_0000` as 32-bit words:

```text
0x4000_0000 / regs[0] : control/status
0x4000_0004 / regs[1] : physical address of the first 128-bit instruction
0x4000_0008 / regs[2] : number of 128-bit instructions, not bytes
```

Control/status bits in `regs[0]`:

```text
bit1 : start command. Software writes 1 to start the instruction stream.
bit2 : hardware busy/done-in-progress status. Hardware sets it while running
       and automatically clears it to 0 after the instruction stream completes.
bit31:24 : hw_version, read-only hardware version byte.
```

Start sequence:

```cpp
regs[1] = instruction_buffer_phys_addr;
regs[2] = instruction_count;
regs[0] = 0x00000002; // bit1 start
```

Completion wait sequence:

```cpp
// After writing start, wait until bit2 is observed high, then wait until
// hardware clears bit2 back to 0. The clear-to-0 transition is completion.
```

The AXI-lite slave uses OR-on-write for `regs[0][7:0]`, and hardware feedback clears some low control/status bits. Do not assume `regs[0]` is a normal overwrite register.

## 128-bit Instruction Layout

Instruction memory is little-endian in two 64-bit words:

```cpp
struct instruction {
    uint64_t lo; // lower 64 bits, stored first in memory
    uint64_t hi; // upper 64 bits, stored second in memory
};
```

The hardware sees the full instruction as:

```text
inst[127:0] = {hi, lo}
```

Use `lml_acc_conv.hpp` helpers when possible. They already match this memory layout.

## Opcodes

```text
0x0 NOP
0x1 SYNC
0x2 BIAS       read DDR int32 bias/psum data into output pingpong A side
0x3 WRITE_AXI  quantize output pingpong completed side and write int8 to DDR
0x4 READ_CONV  optionally read input DDR into input pingpong and/or start conv
0x5 REGS       write kernel/config registers
```

## Common Constructors

Use the helpers from `lml_acc_conv.hpp`:

```cpp
ins.push(lml::instr::regs(addr, data_lo, data_hi));
ins.push(lml::instr::config_regs(c_offset, h_offset, C, H, W));
ins.push(lml::instr::write_back_config(q_multiplier, q_shift, q_zero_point));
ins.push(lml::instr::bias(base, c_offset, h_offset, C, H, W));
ins.push(lml::instr::read_conv(read_en, base, conv_en, ci_mask, co_mask, H, W));
ins.push(lml::instr::write_axi(base, c_offset, h_offset, C, H, W));
ins.push(lml::instr::mode(0x1)); // SYNC
ins.push(lml::instr::mode(0x0)); // NOP
```

Current instruction field widths:

```text
READ_CONV:
  [119]    read enable
  [118:87] read base address
  [59]     conv enable
  [58:55]  ci_mask
  [54:51]  co_mask
  [50:43]  conv input H, 8 bits
  [42:35]  conv input W, 8 bits

REGS addr=128 / config_regs, used by read_axi:
  c_offset : 25 bits
  h_offset : 13 bits
  C        : 3 bits
  H        : 6 bits, valid range 0..63
  W        : 12 bits

REGS addr=130 / write_back_config, used by write_axi:
  q_multiplier : 32 bits signed
  q_shift      : 6 bits unsigned
  q_zero_point : 16 bits signed

BIAS and WRITE_AXI:
  C        : 3 bits
  H        : 6 bits, valid range 0..63
  W        : 12 bits
```

For `bias()` and `write_axi()` fields:

```text
base      : byte physical address in DDR
c_offset  : channel stride in bytes
h_offset  : row stride in bytes
C         : channel count in current 4-channel group, usually 1..4
H         : row count
W         : width in bytes/elements for int8 writeback, or bytes per output row for bias read
```

## Current Software/Bitstream Scope

The current board image is the fixed 32x32 input-tile convolution version. Keep
the software defaults and examples on this convention unless the FPGA bitstream
is regenerated for a wider tile.

```text
conv input      : H=32, W=32
conv output     : H=30, W=30
bias row stride : 30 int32 = 120 bytes
write row stride: 30 int8  = 30 bytes
```

`READ_CONV` still carries H/W fields in the software helper, but for this
hardware version they should be left at, or explicitly set to, `32, 32`.

## WRITE_AXI Width Limit

`write_axi(base, c_offset, h_offset, C, H, W)` writes exactly `W` int8 output bytes per row for each channel.

The RTL reads output pingpong BRAM in 128-bit blocks, where one BRAM block contains four int32 accumulator lanes. Internally the write-back task generator computes:

```text
W_depth = ceil(W / 4)
total_blocks = C * H * W_depth
```

For each row it scans `cur_w_blk = 0 .. W_depth-1`, but it only emits byte write tasks for valid lanes:

```text
elem_base = cur_w_blk * 4
remaining = W - elem_base
valid_lanes = min(4, remaining)
emit lanes where lane < valid_lanes
elem_idx = elem_base + lane
dst_byte_addr = base + c * c_offset + h * h_offset + elem_idx
```

So padding lanes in the last BRAM block of a row are not written to DDR. A row ends when `elem_idx == W - 1`; this row-end marker also forces the AXI64 writer to finish the current burst, so write bursts do not merge across rows. The whole WRITE_AXI instruction ends when:

```text
cur_c == C - 1 && cur_h == H - 1 && elem_idx == W - 1
```

Application-side implication: set `W` to the actual number of int8 bytes that should be written per output row, not to the padded BRAM depth. For a 62-wide int8 output row, use `W = 62`; hardware will read `ceil(62 / 4) = 16` BRAM blocks per row but only write 62 bytes to DDR.

For the current 1-channel 32x32 input / 30x30 output test:

```cpp
bias(base_bias, 0, 120, 1, 30, 120);  // 30 rows, each row has 30 int32 = 120 bytes
write_axi(base_out, 0, 30, 1, 30, 30); // 30 rows, each row has 30 int8 = 30 bytes
```

## Kernel Register Layout

Each kernel group is 9 signed int8 weights packed into one `REGS` instruction.

`addr = co * 4 + ci`, range `0..15`:

```text
addr 0  : output channel 0, input channel 0
addr 1  : output channel 0, input channel 1
addr 2  : output channel 0, input channel 2
addr 3  : output channel 0, input channel 3
addr 4  : output channel 1, input channel 0
...
addr 15 : output channel 3, input channel 3
```

Packing order is row-major 3x3:

```text
kernel[0] kernel[1] kernel[2]
kernel[3] kernel[4] kernel[5]
kernel[6] kernel[7] kernel[8]
```

Pack into `regs(addr, lo, hi)` like this:

```cpp
uint64_t lo = 0;
for (int k = 0; k < 8; ++k) {
    lo |= uint64_t(uint8_t(kernel[k])) << (8 * k);
}
uint16_t hi = uint8_t(kernel[8]);
ins.push(lml::instr::regs(addr, lo, hi));
```

For a single input channel and single output channel test, only `addr=0` is required, with `ci_mask=0x1` and `co_mask=0x1`.

## Conv Input Shape

`config_regs()` is still used by `read_axi` to describe the input tile layout in DDR:

```cpp
ins.push(lml::instr::config_regs(0, 32, 1, 32, 32));
```

For the current 1-channel 32x32 input test this means:

```text
input channel stride = 0 bytes for this 1-channel test
input row stride     = 32 bytes
C                    = 1
H                    = 32
W                    = 32
```

Important limitation: `config_regs()` stores `H` in a 6-bit field, so keep input tile height at `H <= 63`. The current board bitstream is the fixed `32x32` convolution version.

`READ_CONV` carries the actual conv input tile shape separately:

```cpp
ins.push(lml::instr::read_conv(read_en, input_phys, conv_en,
                               ci_mask, co_mask, input_h, input_w));
```

The conv core computes valid 3x3 output shape `(input_h - 2) x (input_w - 2)`. Current board HLS is the fixed `input_h=32`, `input_w=32` version. Channel selection still uses masks; there is no separate conv `C` input. `READ_CONV H/W` must be nonzero whenever `conv_en=true`; otherwise the conv core returns immediately and does not update psum.

For the current fixed tile, use `H=32, W=32`:

```cpp
read_conv(..., input_h = 32, input_w = 32);
```

For input read config, `config_regs H` is row count and `config_regs W` is bytes per row. For a 32x32 input tile use:

```cpp
config_regs(c_stride, row_stride = 32, C, H = 32, W = 32);
```

This keeps the software stream aligned with the currently loaded fixed-tile hardware.

Masks select active channels:

```text
ci_mask bit i enables input channel i
co_mask bit i enables output channel i
```

## Important READ_CONV Semantics

`READ_CONV` has two independent enable bits:

```cpp
lml::instr::read_conv(read_en, base_addr, conv_en, ci_mask, co_mask, H, W)
```

The scheduler issues `read_axi` when `read_en=1`, and issues `conv` when `conv_en=1`. `H/W` are the input tile height and width used by the conv core. The helper defaults are `H=32, W=32`, matching the current fixed-tile hardware convention.

Current scheduler behavior is instruction-level conservative:

```text
READ_CONV can issue only when read_axi, conv, and output-pingpong-A are all idle.
```

Because of that rule, two separate `READ_CONV` instructions do not overlap with each other. For example, this sequence is serialized:

```cpp
ins.push(lml::instr::read_conv(true, input_phys, false, ci_mask, co_mask,
                               input_h, input_w));
ins.push(lml::instr::read_conv(false, input_phys, true, ci_mask, co_mask,
                               input_h, input_w));
```

The first instruction fills input pingpong from DDR. The second instruction starts conv only after the read instruction has finished.

To manually overlap read and conv, software must enable both parts in the same `READ_CONV` instruction:

```cpp
ins.push(lml::instr::read_conv(true, next_input_phys, true, ci_mask, co_mask,
                               input_h, input_w));
```

This starts `read_axi` and `conv` in the same scheduler issue cycle. The intended use is a pingpong pipeline: conv consumes the previously filled input side while read_axi fills the other input side for the next tile/window. The scheduler toggles input pingpong selection on `conv_en`, so the software stream must be ordered so the first conv has valid input already available.

Typical flow:

```cpp
// Prime the pipeline: load the first input tile/window.
ins.push(lml::instr::read_conv(true, input0_phys, false, ci_mask, co_mask,
                               input_h, input_w));

// Steady state: compute current tile/window while loading the next one.
ins.push(lml::instr::read_conv(true, input1_phys, true, ci_mask, co_mask,
                               input_h, input_w));
ins.push(lml::instr::read_conv(true, input2_phys, true, ci_mask, co_mask,
                               input_h, input_w));

// Drain the pipeline: compute the last loaded tile/window.
// The base address is ignored when read_en=false, but keep a readable value.
ins.push(lml::instr::read_conv(false, input2_phys, true, ci_mask, co_mask,
                               input_h, input_w));
```

If the test has only one input tile/window, use the serialized two-instruction form. If the test has multiple windows and is arranged as a pipeline, use the combined `read_en=true, conv_en=true` form for the middle windows. Keep `READ_CONV H/W`, `config_regs H/W`, `BIAS H/W`, and `WRITE_AXI H/W` consistent: conv uses input `H/W`, while bias/writeback use output shape and byte widths.

For a 32x32 input tile, the intended shapes are:

```text
conv input      : H=32, W=32
conv output     : H=30, W=30
bias row stride : 30 int32 = 120 bytes
write row stride: 30 int8  = 30 bytes
```

This shape is directly representable by the current instruction fields:

```cpp
config_regs(c_stride, 32, 4, 32, 32);
bias(bias_base, c_stride, 120, 4, 30, 120);
read_conv(true, input_base, false, 0xf, 0xf, 32, 32);
read_conv(false, input_base, true, 0xf, 0xf, 32, 32);
write_axi(out_base, 30 * 30, 30, 4, 30, 30);
```

## Minimal Quantized Bias Writeback Test

This tests `BIAS -> WRITE_AXI` without convolution:

```cpp
int8_t dummy_kernel[9] = {1,0,0,0,0,0,0,0,0};
ins.push(lml::instr::regs(0, pack_kernel_lo(dummy_kernel), pack_kernel_hi(dummy_kernel)));
ins.push(lml::instr::config_regs(0, 32, 1, 32, 32));

// bias buffer contains 30*30 int32 accumulators.
// write_axi reads q parameters from REGS addr=130. This example overrides them to q_multiplier=3, q_shift=5, q_zero_point=-2.
ins.push(lml::instr::write_back_config(3, 5, -2));
ins.push(lml::instr::bias(bias_phys, 0, 120, 1, 30, 120));
ins.push(lml::instr::write_axi(out_phys, 0, 30, 1, 30, 30));
ins.push(lml::instr::mode(0x1));
ins.push(lml::instr::mode(0x0));
```

Expected output byte for each accumulator is:

```cpp
const int32_t q_multiplier = 3;
const int32_t q_shift = 5;
const int32_t q_zero_point = -2;
const int64_t prod = int64_t(acc) * q_multiplier;
const int64_t rounding = (prod >= 0) ? (1ll << (q_shift - 1)) : -(1ll << (q_shift - 1));
int32_t q = int32_t((prod + rounding) >> q_shift) + q_zero_point;
q = clamp(q, -128, 127);
uint8_t out = uint8_t(int8_t(q));
```

## Minimal 1-channel Random Conv Test

Use this instruction order:

```cpp
ins.clear();
ins.push(lml::instr::regs(0, pack_kernel_lo(kernel), pack_kernel_hi(kernel)));
ins.push(lml::instr::config_regs(0, 32, 1, 32, 32));
ins.push(lml::instr::write_back_config(3, 5, -2));
ins.push(lml::instr::bias(bias_phys, 0, 120, 1, 30, 120));
ins.push(lml::instr::read_conv(true,  input_phys, false, 0x1, 0x1, 32, 32));
ins.push(lml::instr::read_conv(false, input_phys, true,  0x1, 0x1, 32, 32));
ins.push(lml::instr::write_axi(out_phys, 0, 30, 1, 30, 30));
ins.push(lml::instr::mode(0x1));
ins.push(lml::instr::mode(0x0));
```

Expected software model for this 1-channel case:

```cpp
int32_t acc = bias[y][x];
for (int ky = 0; ky < 3; ++ky) {
    for (int kx = 0; kx < 3; ++kx) {
        int8_t in = int8_t(input[(y + ky) * 32 + (x + kx)]);
        int8_t w  = kernel[ky * 3 + kx];
        acc += int32_t(in) * int32_t(w);
    }
}
const int32_t q_multiplier = 3;
const int32_t q_shift = 5;
const int32_t q_zero_point = -2;
const int64_t prod = int64_t(acc) * q_multiplier;
const int64_t rounding = (prod >= 0) ? (1ll << (q_shift - 1)) : -(1ll << (q_shift - 1));
int32_t q = int32_t((prod + rounding) >> q_shift) + q_zero_point;
q = clamp(q, -128, 127);
uint8_t expected = uint8_t(int8_t(q));
```

## Common Failure Modes

1. Passing byte length instead of instruction count to `0x4000_0008`.
   The length register expects number of 128-bit instructions.

2. Storing `{hi, lo}` in memory.
   Memory order must be `lo` first, then `hi`.

3. Expecting two separate `READ_CONV` instructions to overlap.
   `read_conv(true, ..., false, ...)` followed by `read_conv(false, ..., true, ...)` is serialized by the scheduler. Use one instruction with both enable bits set for software-controlled overlap.

4. Combining `READ_CONV(read=true, conv=true)` before priming input pingpong.
   The combined form starts conv immediately, so the input side selected for conv must already contain valid data.

5. Wrong kernel register index.
   For single-channel tests use register address `0`, not `128`.
   Address `128` is for config, not kernel data.

6. Wrong row stride units.
   `bias(... h_offset=120 ...)` because bias/psum rows are `30 int32 = 120 bytes`.
   `write_axi(... h_offset=30 ...)` because output rows are `30 int8 = 30 bytes`.
   `config_regs(... h_offset=32 ...)` because input rows are `32 int8 = 32 bytes`.

7. Passing a shape that does not match the current fixed-tile bitstream.
   This hardware version is expected to run `H=32, W=32`. A software stream using
   `H=32, W=64` can complete but compare against the wrong layout.

8. Running an old DmaApp binary or old `lml_acc_conv.hpp`.
   Decode the instruction dump. `READ_CONV` must show nonzero `[50:43] H` and `[42:35] W`. If both are 0, conv returns immediately and the output stays at the previous/zero contents even though `reg0` reaches done.

9. Mixing old and new tile conventions.
   Keep the generated `conv.v`, software constants, and this document on the same convention. The current convention in this repository is fixed `32x32` input tiles.
