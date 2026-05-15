#pragma once

#include <cstddef>
#include <cstdint>

namespace scheduler {

struct Instruction {
    std::uint64_t hi;
    std::uint64_t lo;
};

using Field128 = Instruction;

namespace detail {

inline void pack_u64(Instruction &dst, std::uint64_t value, unsigned width, unsigned lsb)
{
    if (width == 0 || lsb >= 128)
        return;

    std::uint64_t masked = value;
    if (width < 64)
        masked &= ((UINT64_C(1) << width) - 1);

    if (lsb < 64) {
        unsigned lo_width = width;
        if (lo_width > 64 - lsb)
            lo_width = 64 - lsb;
        const std::uint64_t lo_mask =
            (lo_width == 64) ? UINT64_MAX : ((UINT64_C(1) << lo_width) - 1);
        dst.lo |= (masked & lo_mask) << lsb;
        if (width > lo_width)
            dst.hi |= masked >> lo_width;
    } else {
        dst.hi |= masked << (lsb - 64);
    }
}

inline void pack_field128(Instruction &dst, Field128 value, unsigned width, unsigned lsb)
{
    if (width <= 64) {
        pack_u64(dst, value.lo, width, lsb);
        return;
    }
    pack_u64(dst, value.lo, 64, lsb);
    pack_u64(dst, value.hi, width - 64, lsb + 64);
}

inline Instruction make_instruction(std::uint64_t opcode)
{
    Instruction out = {0, 0};
    pack_u64(out, opcode, 8, 120);
    return out;
}

constexpr std::uint64_t OpSysNop = 0x0ull;
constexpr std::uint64_t OpSysSync = 0x1ull;
constexpr std::uint64_t OpRegs = 0x2ull;
constexpr std::uint64_t OpCmdReadU = 0x3ull;
constexpr std::uint64_t CmdReadUToIpp = 0ull;
constexpr std::uint64_t CmdReadUToOpp = 1ull;
constexpr std::uint64_t OpCmdConvU = 0x4ull;
constexpr std::uint64_t CmdConvURun = 0ull;
constexpr std::uint64_t OpCmdWriteBackU = 0x5ull;
constexpr std::uint64_t CmdWriteBackURun = 0ull;
constexpr std::uint64_t RegConvUCfg = 0ull;
constexpr std::uint64_t RegConvUKernel0 = 1ull;
constexpr std::uint64_t RegConvUKernel1 = 2ull;
constexpr std::uint64_t RegConvUKernel2 = 3ull;
constexpr std::uint64_t RegConvUKernel3 = 4ull;
constexpr std::uint64_t RegConvUKernel4 = 5ull;
constexpr std::uint64_t RegConvUKernel5 = 6ull;
constexpr std::uint64_t RegConvUKernel6 = 7ull;
constexpr std::uint64_t RegConvUKernel7 = 8ull;
constexpr std::uint64_t RegConvUKernel8 = 9ull;
constexpr std::uint64_t RegConvUKernel9 = 10ull;
constexpr std::uint64_t RegConvUKernel10 = 11ull;
constexpr std::uint64_t RegConvUKernel11 = 12ull;
constexpr std::uint64_t RegConvUKernel12 = 13ull;
constexpr std::uint64_t RegConvUKernel13 = 14ull;
constexpr std::uint64_t RegConvUKernel14 = 15ull;
constexpr std::uint64_t RegConvUKernel15 = 16ull;
constexpr std::uint64_t RegWriteBackUQCfg = 17ull;

} // namespace detail

class Instructions {
public:
    Instructions(Instruction *data, std::uint32_t capacity, std::uint32_t device_addr)
        : data_(data), size_(0), capacity_(capacity), device_addr_(device_addr)
    {
    }

    Instruction *data() const { return data_; }
    std::uint32_t size() const { return size_; }
    std::uint32_t capacity() const { return capacity_; }
    std::uint32_t remaining() const { return capacity_ - size_; }
    std::uint32_t device_addr() const { return device_addr_; }
    bool empty() const { return size_ == 0; }
    bool full() const { return size_ >= capacity_; }

    void clear()
    {
        size_ = 0;
    }

    bool nop()
    {
        return push(detail::make_instruction(detail::OpSysNop));
    }

    bool sync()
    {
        return push(detail::make_instruction(detail::OpSysSync));
    }

    bool launch_read_u_to_ipp(std::uint32_t base_addr, std::uint32_t c_offset, std::uint16_t h_offset, std::uint8_t c, std::uint8_t h, std::uint16_t w)
    {
        Instruction inst = detail::make_instruction(detail::OpCmdReadU);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::CmdReadUToIpp), 1, 119);
        detail::pack_u64(inst, static_cast<std::uint64_t>(base_addr), 32, 87);
        detail::pack_u64(inst, static_cast<std::uint64_t>(c_offset), 32, 55);
        detail::pack_u64(inst, static_cast<std::uint64_t>(h_offset), 16, 39);
        detail::pack_u64(inst, static_cast<std::uint64_t>(c), 3, 36);
        detail::pack_u64(inst, static_cast<std::uint64_t>(h), 6, 30);
        detail::pack_u64(inst, static_cast<std::uint64_t>(w), 16, 14);
        return push(inst);
    }

    bool launch_read_u_to_opp(std::uint32_t base_addr, std::uint32_t c_offset, std::uint16_t h_offset, std::uint8_t c, std::uint8_t h, std::uint16_t w)
    {
        Instruction inst = detail::make_instruction(detail::OpCmdReadU);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::CmdReadUToOpp), 1, 119);
        detail::pack_u64(inst, static_cast<std::uint64_t>(base_addr), 32, 87);
        detail::pack_u64(inst, static_cast<std::uint64_t>(c_offset), 32, 55);
        detail::pack_u64(inst, static_cast<std::uint64_t>(h_offset), 16, 39);
        detail::pack_u64(inst, static_cast<std::uint64_t>(c), 3, 36);
        detail::pack_u64(inst, static_cast<std::uint64_t>(h), 6, 30);
        detail::pack_u64(inst, static_cast<std::uint64_t>(w), 16, 14);
        return push(inst);
    }

    bool launch_conv_u_run(std::uint8_t h, std::uint8_t w)
    {
        Instruction inst = detail::make_instruction(detail::OpCmdConvU);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::CmdConvURun), 1, 119);
        detail::pack_u64(inst, static_cast<std::uint64_t>(h), 8, 111);
        detail::pack_u64(inst, static_cast<std::uint64_t>(w), 8, 103);
        return push(inst);
    }

    bool config_conv_u_cfg(std::uint8_t cin_mask, std::uint8_t cout_mask)
    {
        Instruction inst = detail::make_instruction(detail::OpRegs);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::RegConvUCfg), 8, 112);
        detail::pack_u64(inst, static_cast<std::uint64_t>(cin_mask), 4, 108);
        detail::pack_u64(inst, static_cast<std::uint64_t>(cout_mask), 4, 104);
        return push(inst);
    }

    bool config_conv_u_kernel0(Field128 data)
    {
        Instruction inst = detail::make_instruction(detail::OpRegs);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::RegConvUKernel0), 8, 112);
        detail::pack_field128(inst, data, 72, 40);
        return push(inst);
    }

    bool config_conv_u_kernel1(Field128 data)
    {
        Instruction inst = detail::make_instruction(detail::OpRegs);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::RegConvUKernel1), 8, 112);
        detail::pack_field128(inst, data, 72, 40);
        return push(inst);
    }

    bool config_conv_u_kernel2(Field128 data)
    {
        Instruction inst = detail::make_instruction(detail::OpRegs);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::RegConvUKernel2), 8, 112);
        detail::pack_field128(inst, data, 72, 40);
        return push(inst);
    }

    bool config_conv_u_kernel3(Field128 data)
    {
        Instruction inst = detail::make_instruction(detail::OpRegs);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::RegConvUKernel3), 8, 112);
        detail::pack_field128(inst, data, 72, 40);
        return push(inst);
    }

    bool config_conv_u_kernel4(Field128 data)
    {
        Instruction inst = detail::make_instruction(detail::OpRegs);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::RegConvUKernel4), 8, 112);
        detail::pack_field128(inst, data, 72, 40);
        return push(inst);
    }

    bool config_conv_u_kernel5(Field128 data)
    {
        Instruction inst = detail::make_instruction(detail::OpRegs);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::RegConvUKernel5), 8, 112);
        detail::pack_field128(inst, data, 72, 40);
        return push(inst);
    }

    bool config_conv_u_kernel6(Field128 data)
    {
        Instruction inst = detail::make_instruction(detail::OpRegs);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::RegConvUKernel6), 8, 112);
        detail::pack_field128(inst, data, 72, 40);
        return push(inst);
    }

    bool config_conv_u_kernel7(Field128 data)
    {
        Instruction inst = detail::make_instruction(detail::OpRegs);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::RegConvUKernel7), 8, 112);
        detail::pack_field128(inst, data, 72, 40);
        return push(inst);
    }

    bool config_conv_u_kernel8(Field128 data)
    {
        Instruction inst = detail::make_instruction(detail::OpRegs);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::RegConvUKernel8), 8, 112);
        detail::pack_field128(inst, data, 72, 40);
        return push(inst);
    }

    bool config_conv_u_kernel9(Field128 data)
    {
        Instruction inst = detail::make_instruction(detail::OpRegs);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::RegConvUKernel9), 8, 112);
        detail::pack_field128(inst, data, 72, 40);
        return push(inst);
    }

    bool config_conv_u_kernel10(Field128 data)
    {
        Instruction inst = detail::make_instruction(detail::OpRegs);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::RegConvUKernel10), 8, 112);
        detail::pack_field128(inst, data, 72, 40);
        return push(inst);
    }

    bool config_conv_u_kernel11(Field128 data)
    {
        Instruction inst = detail::make_instruction(detail::OpRegs);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::RegConvUKernel11), 8, 112);
        detail::pack_field128(inst, data, 72, 40);
        return push(inst);
    }

    bool config_conv_u_kernel12(Field128 data)
    {
        Instruction inst = detail::make_instruction(detail::OpRegs);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::RegConvUKernel12), 8, 112);
        detail::pack_field128(inst, data, 72, 40);
        return push(inst);
    }

    bool config_conv_u_kernel13(Field128 data)
    {
        Instruction inst = detail::make_instruction(detail::OpRegs);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::RegConvUKernel13), 8, 112);
        detail::pack_field128(inst, data, 72, 40);
        return push(inst);
    }

    bool config_conv_u_kernel14(Field128 data)
    {
        Instruction inst = detail::make_instruction(detail::OpRegs);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::RegConvUKernel14), 8, 112);
        detail::pack_field128(inst, data, 72, 40);
        return push(inst);
    }

    bool config_conv_u_kernel15(Field128 data)
    {
        Instruction inst = detail::make_instruction(detail::OpRegs);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::RegConvUKernel15), 8, 112);
        detail::pack_field128(inst, data, 72, 40);
        return push(inst);
    }

    bool launch_write_back_u_run(std::uint32_t base_addr, std::uint32_t c_offset, std::uint16_t h_offset, std::uint8_t c, std::uint8_t h, std::uint16_t w)
    {
        Instruction inst = detail::make_instruction(detail::OpCmdWriteBackU);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::CmdWriteBackURun), 1, 119);
        detail::pack_u64(inst, static_cast<std::uint64_t>(base_addr), 32, 87);
        detail::pack_u64(inst, static_cast<std::uint64_t>(c_offset), 32, 55);
        detail::pack_u64(inst, static_cast<std::uint64_t>(h_offset), 16, 39);
        detail::pack_u64(inst, static_cast<std::uint64_t>(c), 3, 36);
        detail::pack_u64(inst, static_cast<std::uint64_t>(h), 6, 30);
        detail::pack_u64(inst, static_cast<std::uint64_t>(w), 16, 14);
        return push(inst);
    }

    bool config_write_back_u_q_cfg(std::int32_t q_multiplier, std::uint8_t q_shift, std::int16_t q_zero_point)
    {
        Instruction inst = detail::make_instruction(detail::OpRegs);
        detail::pack_u64(inst, static_cast<std::uint64_t>(detail::RegWriteBackUQCfg), 8, 112);
        detail::pack_u64(inst, static_cast<std::uint64_t>(q_multiplier), 32, 80);
        detail::pack_u64(inst, static_cast<std::uint64_t>(q_shift), 6, 74);
        detail::pack_u64(inst, static_cast<std::uint64_t>(q_zero_point), 16, 58);
        return push(inst);
    }

private:
    bool push(Instruction instruction)
    {
        if (full())
            return false;
        data_[size_++] = instruction;
        return true;
    }

    Instruction *data_;
    std::uint32_t size_;
    std::uint32_t capacity_;
    std::uint32_t device_addr_;
};

enum class State {
    Idle,
    Working,
    Resetting,
};

class FpgaScheduler {
public:
    explicit FpgaScheduler(std::uintptr_t base_addr)
        : regs_(reinterpret_cast<volatile std::uint32_t *>(base_addr)),
          bytes_(reinterpret_cast<volatile std::uint8_t *>(base_addr)),
          valid_(hw_version() == ExpectedHwVersion)
    {
        reset_blocking();
    }

    explicit FpgaScheduler(volatile void *base_addr)
        : FpgaScheduler(reinterpret_cast<std::uintptr_t>(base_addr))
    {
    }

    bool valid() const
    {
        return valid_;
    }

    void set_irq(bool enable) const
    {
        std::uint8_t cfg = bytes_[1];
        if (enable)
            cfg = static_cast<std::uint8_t>(cfg | CfgIrqEnable);
        else
            cfg = static_cast<std::uint8_t>(cfg & ~CfgIrqEnable);
        bytes_[1] = cfg;
    }

    void run(const Instructions &instructions) const
    {
        run(instructions.device_addr(), instructions.size());
    }

    void run(std::uint32_t instruction_addr, std::uint32_t instruction_count) const
    {
        regs_[RegInstrBase] = instruction_addr;
        regs_[RegInstrCount] = instruction_count;
        bytes_[0] = ReqStart;
    }

    State get_state() const
    {
        const std::uint32_t ctrl = regs_[RegCtrl];
        if ((ctrl & StateResetting) != 0u)
            return State::Resetting;
        if ((ctrl & StateWorking) != 0u)
            return State::Working;
        return State::Idle;
    }

    bool check_state(State state) const
    {
        return get_state() == state;
    }

    bool is_busy() const
    {
        return get_state() != State::Idle;
    }

    std::uint8_t hw_version() const
    {
        return static_cast<std::uint8_t>((regs_[RegCtrl] >> 24) & 0xffu);
    }

private:
    static constexpr std::uint8_t ExpectedHwVersion = 1;
    static constexpr std::uint32_t RegCtrl = 0;
    static constexpr std::uint32_t RegInstrBase = 1;
    static constexpr std::uint32_t RegInstrCount = 2;
    static constexpr std::uint8_t ReqReset = 1u << 0;
    static constexpr std::uint8_t ReqStart = 1u << 1;
    static constexpr std::uint8_t CfgIrqEnable = 1u << 0;
    static constexpr std::uint32_t StateWorking = 1u << 16;
    static constexpr std::uint32_t StateResetting = 1u << 17;

    void reset_blocking() const
    {
        bytes_[0] = ReqReset;
        while ((bytes_[0] & ReqReset) != 0u) {
        }
    }

    volatile std::uint32_t *regs_;
    volatile std::uint8_t *bytes_;
    bool valid_;
};

} // namespace scheduler
