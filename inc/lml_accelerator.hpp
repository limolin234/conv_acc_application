#ifndef LML_ACCELERATOR_HPP
#define LML_ACCELERATOR_HPP

#include <stdint.h>
#include <time.h>
#include <unistd.h>

#include "lml_acc_conv.hpp"

namespace lml {

class accelerator {
public:
    accelerator(volatile uint32_t* regs, const void* virt_base,
                uint32_t phys_base, uint32_t timeout_ms = 5000)
        : regs_(regs),
          virt_base_(reinterpret_cast<uintptr_t>(virt_base)),
          phys_base_(phys_base),
          timeout_ms_(timeout_ms) {}

    bool valid() const {
        return regs_ != nullptr && virt_base_ != 0;
    }

    uint32_t phys(const void* ptr) const {
        const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
        return phys_base_ + static_cast<uint32_t>(p - virt_base_);
    }

    void set_timeout_ms(uint32_t timeout_ms) {
        timeout_ms_ = timeout_ms;
    }

    uint32_t timeout_ms() const {
        return timeout_ms_;
    }

    volatile uint32_t* regs() const {
        return regs_;
    }

    void start(const instruction_buffer& ins) const {
        regs_[kRegInsAddr] = ins.phys_addr();
        regs_[kRegInsLen] = static_cast<uint32_t>(ins.size());
        regs_[kRegCtrl] = kCtrlStart;
    }

    bool wait_done() const {
        const uint64_t deadline = monotonic_ms() + timeout_ms_;
        while (monotonic_ms() < deadline) {
            const uint32_t ctrl = regs_[kRegCtrl];
            if ((ctrl & (kCtrlStart | kCtrlBusy)) == 0) {
                return true;
            }
            usleep(1000);
        }
        return false;
    }

    bool run(const instruction_buffer& ins) const {
        start(ins);
        return wait_done();
    }

    uint32_t ctrl() const {
        return regs_[kRegCtrl];
    }

private:
    static constexpr uint32_t kRegCtrl = 0;
    static constexpr uint32_t kRegInsAddr = 1;
    static constexpr uint32_t kRegInsLen = 2;
    static constexpr uint32_t kCtrlStart = 1u << 1;
    static constexpr uint32_t kCtrlBusy = 1u << 2;

    static uint64_t monotonic_ms() {
        timespec ts{};
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<uint64_t>(ts.tv_sec) * 1000u +
               static_cast<uint64_t>(ts.tv_nsec / 1000000u);
    }

    volatile uint32_t* regs_ = nullptr;
    uintptr_t virt_base_ = 0;
    uint32_t phys_base_ = 0;
    uint32_t timeout_ms_ = 5000;
};

} // namespace lml

#endif // LML_ACCELERATOR_HPP
