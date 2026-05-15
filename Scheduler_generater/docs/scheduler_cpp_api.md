# Scheduler C++ API Guide / 调度器 C++ API 使用说明

This document describes the generic user-facing C++ API generated from a JSON
control specification.  It is written as project usage documentation: users
should not need to know opcodes, command IDs, register addresses, start bits, or
reset bits.

本文档说明由 JSON 控制规格生成的通用 C++ API。它面向项目使用者：用户不需要关心
opcode、command id、寄存器地址、start bit 或 reset bit。

## Overview / 总体流程

The generated API has two main objects:

- `scheduler::Instructions`: a fixed-capacity instruction buffer builder.
- `scheduler::FpgaScheduler`: an accelerator control object backed by an AXI-Lite
  register base address.

生成 API 主要有两个对象：

- `scheduler::Instructions`：固定容量的指令缓冲区构造器。
- `scheduler::FpgaScheduler`：由 AXI-Lite 寄存器基地址构造的加速器控制对象。

Typical usage:

典型用法：

```cpp
#include "scheduler_control.hpp"

scheduler::Instruction buffer[128];

scheduler::Instructions ins(
    buffer,
    128,
    instruction_buffer_device_addr
);

ins.config_module_a_cfg(...);
ins.launch_module_a_run(...);
ins.launch_module_b_run(...);
ins.sync();

scheduler::FpgaScheduler fpga(scheduler_reg_base);

if (!fpga.valid()) {
    // Hardware version does not match this generated header.
}

fpga.run(ins);

while (fpga.is_busy()) {
}
```

The intended public surface is:

用户主要使用以下公开接口：

- `Instructions::config_<module>_<register>(...)`
- `Instructions::launch_<module>_<command>(...)`
- `Instructions::sync()`
- `Instructions::nop()`
- `FpgaScheduler::run(...)`
- `FpgaScheduler::is_busy()`
- `FpgaScheduler::get_state()`
- `FpgaScheduler::check_state(...)`
- `FpgaScheduler::valid()`

## JSON-To-API Mapping / JSON 到 API 的映射

A module command in JSON:

JSON 中的模块命令：

```json
{
  "Modules": {
    "dma_read": {
      "Command": {
        "to_input_buffer": {
          "Args": [
            {"Name": "base_addr", "Width": 32},
            {"Name": "length", "Width": 32}
          ],
          "Using": ["input_buffer"]
        }
      }
    }
  }
}
```

Generates a launch method:

会生成 launch 方法：

```cpp
bool launch_dma_read_to_input_buffer(
    std::uint32_t base_addr,
    std::uint32_t length
);
```

A module register group in JSON:

JSON 中的模块配置寄存器：

```json
{
  "Modules": {
    "compute": {
      "Regs": {
        "cfg": [
          {"Name": "mode", "Width": 4},
          {"Name": "height", "Width": 16},
          {"Name": "width", "Width": 16}
        ]
      }
    }
  }
}
```

Generates a config method:

会生成 config 方法：

```cpp
bool config_compute_cfg(
    std::uint8_t mode,
    std::uint16_t height,
    std::uint16_t width
);
```

General naming rule:

通用命名规则：

```text
config_<module_name>_<json_register_name>(...)
launch_<module_name>_<json_command_name>(...)
```

Function arguments follow the field order in JSON.  The generated API chooses
the smallest common C++ integer type that can hold each field width.

函数参数顺序与 JSON 中字段顺序一致。生成器会为每个字段选择能容纳该位宽的常用
C++ 整数类型。

Field defaults may be plain integers or Verilog-style literal strings.  Both
forms are accepted by the generator.

字段默认值既可以是普通整数，也可以是 Verilog 风格字面量字符串。生成器两种形式
都接受。

## Instruction Buffer / 指令缓冲区

The API does not allocate instruction memory.  The user provides memory from
their own memory pool, reserved memory, DMA buffer, or bare-metal static storage.

API 不负责分配指令内存。用户从自己的内存池、reserved memory、DMA buffer 或裸机静态
数组中提供内存。

```cpp
scheduler::Instructions ins(
    scheduler::Instruction *data,
    std::uint32_t capacity,
    std::uint32_t device_addr
);
```

Arguments:

参数：

- `data`: CPU-visible pointer to the instruction buffer.
- `capacity`: maximum number of 128-bit instructions in the buffer.
- `device_addr`: FPGA-visible address of the same buffer.

- `data`：CPU 可访问的指令缓冲区指针。
- `capacity`：缓冲区最多能容纳的 128-bit 指令数量。
- `device_addr`：同一缓冲区在 FPGA 侧可见的地址。

Useful methods:

常用方法：

```cpp
ins.size();       // current instruction count / 当前指令数
ins.capacity();   // maximum instruction count / 最大容量
ins.remaining();  // capacity - size / 剩余容量
ins.device_addr();
ins.clear();
ins.empty();
ins.full();
```

All instruction-building methods return `bool`.  `false` means the buffer is
full or an argument is invalid.

所有构建指令的方法都返回 `bool`。返回 `false` 表示缓冲区已满，或参数非法。

## Scheduler Object / 调度器对象

```cpp
scheduler::FpgaScheduler fpga(std::uintptr_t scheduler_reg_base);
```

The constructor checks the generated hardware version and performs a blocking
reset.  Normal user code does not directly write the reset bit.

构造函数会检查生成头文件中的硬件版本，并执行一次阻塞 reset。普通用户代码不直接写
reset bit。

Run a prepared instruction buffer:

运行一段已构建好的指令：

```cpp
fpga.run(ins);
```

Run by raw FPGA-visible address and instruction count:

也可以直接用 FPGA 可见地址和指令数量运行：

```cpp
fpga.run(instruction_buffer_device_addr, instruction_count);
```

Check state:

状态检查：

```cpp
bool busy = fpga.is_busy();
scheduler::State state = fpga.get_state();
bool idle = fpga.check_state(scheduler::State::Idle);
```

States:

状态枚举：

```cpp
scheduler::State::Idle
scheduler::State::Working
scheduler::State::Resetting
```

Other methods:

其他方法：

```cpp
fpga.valid();       // generated header matches hardware version / 版本匹配
fpga.set_irq(true);
fpga.set_irq(false);
fpga.hw_version();
```

## System Instructions / 系统指令

```cpp
ins.nop();
```

Adds a no-op instruction.

加入一条空操作指令。

```cpp
ins.sync();
```

Adds a synchronization instruction.  Its exact dependencies are generated from
the JSON `System.Sync.Deps` field.

加入一条同步指令。它具体等待哪些模块或资源，由 JSON 中的 `System.Sync.Deps`
决定。

## Launch Methods / 启动指令构建函数

Every JSON command generates one `launch_*` method.

每个 JSON command 会生成一个 `launch_*` 方法。

Example JSON:

示例 JSON：

```json
{
  "Modules": {
    "writer": {
      "Command": {
        "run": {
          "Args": [
            {"Name": "base_addr", "Width": 32},
            {"Name": "length", "Width": 32}
          ],
          "Using": ["output_buffer"]
        }
      }
    }
  }
}
```

Generated C++:

生成的 C++：

```cpp
bool launch_writer_run(
    std::uint32_t base_addr,
    std::uint32_t length
);
```

Calling this method only appends one instruction into `Instructions`; it does
not immediately start hardware execution.  Hardware starts when `fpga.run(ins)`
is called.

调用这个方法只是在 `Instructions` 中追加一条指令，不会立刻启动硬件。真正启动硬件
发生在 `fpga.run(ins)`。

## Config Methods / 配置指令构建函数

Every JSON register group generates one `config_*` method.

每个 JSON register group 会生成一个 `config_*` 方法。

Example JSON:

示例 JSON：

```json
{
  "Modules": {
    "filter": {
      "Regs": {
        "cfg": [
          {"Name": "mode", "Width": 4},
          {"Name": "scale", "Width": 16},
          {"Name": "zero_point", "Width": 16, "Signed": true}
        ]
      }
    }
  }
}
```

Generated C++:

生成的 C++：

```cpp
bool config_filter_cfg(
    std::uint8_t mode,
    std::uint16_t scale,
    std::int16_t zero_point
);
```

The function name is derived from the module name and JSON register name.  The
parameter names and order are derived from the JSON field list.

函数名来自模块名和 JSON 寄存器名。参数名和顺序来自 JSON 字段列表。

## Wide Fields / 宽字段

Fields wider than 64 bits use a 128-bit container type.  The current hand-written
reference uses `scheduler::Instruction` as this wide container; the generated
version may expose a clearer alias such as:

超过 64 bit 的字段使用 128-bit 容器类型。当前手写参考直接使用
`scheduler::Instruction` 作为宽字段容器；生成版本可以暴露一个更清楚的别名：

```cpp
using Field128 = Instruction;
```

Example JSON:

示例 JSON：

```json
{
  "Regs": {
    "kernel0": [
      {"Name": "data", "Width": 72, "Signed": true}
    ]
  }
}
```

Generated C++:

生成的 C++：

```cpp
bool config_compute_kernel0(scheduler::Field128 data);
```

For a 72-bit field:

对于 72-bit 字段：

```text
data.lo[63:0] -> field[63:0]
data.hi[7:0]  -> field[71:64]
data.hi[63:8] -> ignored / 忽略
```

Example:

示例：

```cpp
scheduler::Field128 kernel = {
    0x0000000000000012ULL,
    0x3456789abcdef001ULL
};

ins.config_compute_kernel0(kernel);
```

The packer automatically truncates unused high bits according to the field
width in JSON.

打包函数会按照 JSON 中的字段位宽自动截断高位。

## Memory And Cache Notes / 内存与 Cache 注意事项

- `Instructions` does not flush caches.
- `device_addr` must be the FPGA-visible address, not necessarily the CPU
  virtual address.
- If the FPGA reads the instruction buffer through DMA/AXI, the user project
  must ensure the buffer is visible to hardware.  This may require cache flush,
  uncached mapping, or a platform-specific DMA buffer.
- `FpgaScheduler` hides start/reset details.  User code should call `run()` and
  observe `is_busy()` or `get_state()`.

- `Instructions` 不负责 flush cache。
- `device_addr` 必须是 FPGA 侧可见地址，不一定等于 CPU 虚拟地址。
- 如果 FPGA 通过 DMA/AXI 读取指令缓冲区，用户工程必须保证硬件能看到最新内容。这可能
  需要 cache flush、uncached mapping 或平台相关 DMA buffer。
- `FpgaScheduler` 隐藏 start/reset 细节。用户代码只应调用 `run()`，并用
  `is_busy()` 或 `get_state()` 观察状态。
