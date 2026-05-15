#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "lml_acc_conv.hpp"
#include "yolo_conv3x3.hpp"

namespace {

constexpr uint32_t kExchPhysAddr = 0x30000000u;
constexpr size_t kExchSize = 0x10000000ul;
constexpr uint32_t kCtrlPhysAddr = 0x40000000u;
constexpr size_t kCtrlSize = 0x1000ul;
constexpr uint32_t kRegCtrl = 0;
constexpr uint32_t kRegInsAddr = 1;
constexpr uint32_t kRegInsLen = 2;
constexpr uint32_t kCtrlStart = 1u << 1;
constexpr uint32_t kInW = lml::kDefaultConvInputW;
constexpr uint32_t kInH = lml::kDefaultConvInputH;
constexpr uint32_t kOutW = kInW - 2;
constexpr uint32_t kOutH = kInH - 2;
constexpr uint32_t kChannels = 4;
constexpr uint32_t kBankStrideBytes = 512 * 16;
constexpr uint32_t kInputRowStrideBytes = kInW;
constexpr uint32_t kBiasRowStrideBytes = kOutW * 4;
constexpr uint32_t kOutputRowStrideBytes = kOutW;
constexpr uint32_t kOutputChannelStrideBytes = kOutW * kOutH;
constexpr size_t kInputBytes = kChannels * kBankStrideBytes;
constexpr size_t kBiasBytes = kChannels * kBankStrideBytes;
constexpr size_t kOutputBytes = kChannels * kOutW * kOutH;
constexpr uint32_t kMaxInstructions = 512;
constexpr uint8_t kSentinel = 0xa5;

struct mmio_region {
    void* addr = MAP_FAILED;
    size_t size = 0;
    ~mmio_region() {
        if (addr != MAP_FAILED) munmap(addr, size);
    }
    bool map(int fd, uint32_t phys, size_t bytes) {
        size = bytes;
        addr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                    static_cast<off_t>(phys));
        return addr != MAP_FAILED;
    }
};

struct ctx {
    volatile uint32_t* regs = nullptr;
    lml::exch_pool* pool = nullptr;
    uint32_t timeout_ms = 5000;
};

static uint64_t monotonic_ms() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000u +
           static_cast<uint64_t>(ts.tv_nsec / 1000000u);
}

static void try_msync(void* ptr, size_t bytes, int flags) {
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return;
    const uintptr_t begin = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t mask = static_cast<uintptr_t>(page_size) - 1u;
    const uintptr_t aligned_begin = begin & ~mask;
    const uintptr_t aligned_end = (begin + bytes + mask) & ~mask;
    (void)msync(reinterpret_cast<void*>(aligned_begin),
                aligned_end - aligned_begin, flags);
}

static uint8_t load8(const uint8_t* p) {
    return *reinterpret_cast<const volatile uint8_t*>(p);
}

static void store32_le(uint8_t* p, uint32_t value) {
    p[0] = static_cast<uint8_t>(value);
    p[1] = static_cast<uint8_t>(value >> 8);
    p[2] = static_cast<uint8_t>(value >> 16);
    p[3] = static_cast<uint8_t>(value >> 24);
}

static uint32_t input_offset(uint32_t c, uint32_t y, uint32_t x) {
    return c * kBankStrideBytes + y * kInputRowStrideBytes + x;
}

static uint32_t bias_offset(uint32_t c, uint32_t y, uint32_t x) {
    return c * kBankStrideBytes + y * kBiasRowStrideBytes + x * 4;
}

static uint32_t output_offset(uint32_t c, uint32_t y, uint32_t x) {
    return c * kOutputChannelStrideBytes + y * kOutputRowStrideBytes + x;
}

static uint8_t quant_from_acc(int32_t acc) {
    return lml::yolo::quantize_acc(acc);
}

static uint64_t pack_kernel_lo(const int8_t kernel[9]) {
    uint64_t lo = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        lo |= static_cast<uint64_t>(static_cast<uint8_t>(kernel[i])) << (i * 8);
    }
    return lo;
}

static uint16_t pack_kernel_hi(const int8_t kernel[9]) {
    return static_cast<uint8_t>(kernel[8]);
}

static void print_bytes(const char* label, const uint8_t* data, size_t bytes) {
    printf("%s:", label);
    for (size_t i = 0; i < bytes; ++i) {
        if ((i % 16) == 0) printf("\n  %02zu:", i);
        printf(" %02x", load8(data + i));
    }
    printf("\n");
}

static void print_stream(const lml::instruction_buffer& ins) {
    printf("  instruction stream base=0x%08x count=%zu bytes=%zu\n",
           ins.phys_addr(), ins.size(), ins.size() * sizeof(lml::instruction));
    for (size_t i = 0; i < ins.size(); ++i) {
        printf("    [%02zu] phys=0x%08x hi=0x%016" PRIx64 " lo=0x%016" PRIx64 "\n",
               i,
               ins.phys_addr() + static_cast<uint32_t>(i * sizeof(lml::instruction)),
               ins[i].hi, ins[i].lo);
    }
}

static bool wait_start_clear(volatile uint32_t* regs, uint32_t timeout_ms) {
    const uint64_t deadline = monotonic_ms() + timeout_ms;
    while (monotonic_ms() < deadline) {
        if ((regs[kRegCtrl] & kCtrlStart) == 0) return true;
        usleep(1000);
    }
    return false;
}

static bool wait_byte_change(uint8_t* p, uint8_t sentinel, uint32_t timeout_ms) {
    const uint64_t deadline = monotonic_ms() + timeout_ms;
    while (monotonic_ms() < deadline) {
        try_msync(p, 64, MS_INVALIDATE);
        if (load8(p) != sentinel) return true;
        usleep(1000);
    }
    return false;
}

static bool run_stream(ctx& c, lml::instruction_buffer& ins,
                       uint8_t* wait_ptr, const char* name) {
    printf("%s:\n", name);
    print_stream(ins);
    try_msync(ins.data(), ins.size() * sizeof(lml::instruction), MS_SYNC);
    c.regs[kRegInsAddr] = ins.phys_addr();
    c.regs[kRegInsLen] = static_cast<uint32_t>(ins.size());
    c.regs[kRegCtrl] = kCtrlStart;
    if (!wait_byte_change(wait_ptr, kSentinel, c.timeout_ms)) {
        fprintf(stderr, "  timeout waiting output, reg0=0x%08x\n", c.regs[kRegCtrl]);
        return false;
    }
    wait_start_clear(c.regs, c.timeout_ms);
    try_msync(wait_ptr, kOutputBytes, MS_INVALIDATE);
    printf("  reg0=0x%08x\n", c.regs[kRegCtrl]);
    print_bytes("  output sample", wait_ptr, 64);
    return true;
}

static void fill_bias_constant(uint8_t* bias, int32_t acc) {
    memset(bias, 0, kBiasBytes);
    for (uint32_t c = 0; c < kChannels; ++c) {
        for (uint32_t y = 0; y < kOutH; ++y) {
            for (uint32_t x = 0; x < kOutW; ++x) {
                store32_le(bias + bias_offset(c, y, x), static_cast<uint32_t>(acc));
            }
        }
    }
}

static bool check_constant(const uint8_t* out, uint8_t exp, const char* name) {
    for (uint32_t c = 0; c < kChannels; ++c) {
        for (uint32_t y = 0; y < kOutH; ++y) {
            for (uint32_t x = 0; x < kOutW; ++x) {
                const uint32_t off = output_offset(c, y, x);
                const uint8_t got = load8(out + off);
                if (got != exp) {
                    fprintf(stderr,
                            "  %s mismatch c=%u y=%u x=%u off=%u got=0x%02x exp=0x%02x\n",
                            name, c, y, x, off, got, exp);
                    return false;
                }
            }
        }
    }
    printf("  PASS %s constant=0x%02x\n", name, exp);
    return true;
}

static bool case_bias_write(ctx& c, lml::instruction_buffer& ins,
                            uint8_t* bias, uint8_t* out,
                            int32_t acc, const char* name) {
    ins.clear();
    fill_bias_constant(bias, acc);
    memset(out, kSentinel, kOutputBytes);
    try_msync(bias, kBiasBytes, MS_SYNC);
    try_msync(out, kOutputBytes, MS_SYNC);
    ins.push(lml::instr::bias(c.pool->phys(bias), kBankStrideBytes,
                              kBiasRowStrideBytes, kChannels, kOutH,
                              kOutW * 4));
    ins.push(lml::instr::write_axi(c.pool->phys(out),
                                   kOutputChannelStrideBytes,
                                   kOutputRowStrideBytes, kChannels,
                                   kOutH, kOutW));
    ins.push(lml::instr::mode(0x1));
    ins.push(lml::instr::mode(0x0));
    if (!run_stream(c, ins, out, name)) return false;
    return check_constant(out, quant_from_acc(acc), name);
}

static void fill_input_constant(uint8_t* input, int8_t value) {
    memset(input, 0, kInputBytes);
    for (uint32_t c = 0; c < kChannels; ++c) {
        for (uint32_t y = 0; y < kInH; ++y) {
            for (uint32_t x = 0; x < kInW; ++x) {
                input[input_offset(c, y, x)] = static_cast<uint8_t>(value);
            }
        }
    }
}

static void push_all_kernel_regs(lml::instruction_buffer& ins,
                                 const int8_t kernel[9]) {
    for (uint32_t i = 0; i < kChannels * kChannels; ++i) {
        ins.push(lml::instr::regs(static_cast<uint8_t>(i),
                                  pack_kernel_lo(kernel),
                                  pack_kernel_hi(kernel)));
    }
}

static bool case_two_group_kernel_change(ctx& c, lml::instruction_buffer& ins,
                                         uint8_t* in0, uint8_t* in1,
                                         uint8_t* bias, uint8_t* out) {
    ins.clear();
    fill_input_constant(in0, 8);
    fill_input_constant(in1, 8);
    fill_bias_constant(bias, 0);
    memset(out, kSentinel, kOutputBytes);
    try_msync(in0, kInputBytes, MS_SYNC);
    try_msync(in1, kInputBytes, MS_SYNC);
    try_msync(bias, kBiasBytes, MS_SYNC);
    try_msync(out, kOutputBytes, MS_SYNC);

    const int8_t k0[9] = {8, 0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t k1[9] = {16, 0, 0, 0, 0, 0, 0, 0, 0};
    ins.push(lml::instr::config_regs(kBankStrideBytes, kInputRowStrideBytes,
                                     kChannels, kInH, kInW));
    ins.push(lml::instr::bias(c.pool->phys(bias), kBankStrideBytes,
                              kBiasRowStrideBytes, kChannels, kOutH,
                              kOutW * 4));
    push_all_kernel_regs(ins, k0);
    ins.push(lml::instr::read_conv(true, c.pool->phys(in0), false, 0xf, 0xf,
                                   kInH, kInW));
    ins.push(lml::instr::read_conv(false, c.pool->phys(in0), true, 0xf, 0xf,
                                   kInH, kInW));
    ins.push(lml::instr::mode(0x1));
    push_all_kernel_regs(ins, k1);
    ins.push(lml::instr::read_conv(true, c.pool->phys(in1), false, 0xf, 0xf,
                                   kInH, kInW));
    ins.push(lml::instr::read_conv(false, c.pool->phys(in1), true, 0xf, 0xf,
                                   kInH, kInW));
    ins.push(lml::instr::write_axi(c.pool->phys(out),
                                   kOutputChannelStrideBytes,
                                   kOutputRowStrideBytes, kChannels,
                                   kOutH, kOutW));
    ins.push(lml::instr::bias(c.pool->phys(bias), kBankStrideBytes,
                              kBiasRowStrideBytes, kChannels, kOutH,
                              kOutW * 4));
    ins.push(lml::instr::mode(0x1));
    ins.push(lml::instr::mode(0x0));
    if (!run_stream(c, ins, out, "two_group_kernel_change_accum")) return false;

    const int32_t acc = 4 * 8 * 8 + 4 * 8 * 16;
    return check_constant(out, quant_from_acc(acc),
                          "two_group_kernel_change_accum");
}

static bool case_write_without_new_bias(ctx& c, lml::instruction_buffer& ins,
                                        uint8_t* bias, uint8_t* out0,
                                        uint8_t* out1) {
    ins.clear();
    fill_bias_constant(bias, 256 * 7);
    memset(out0, kSentinel, kOutputBytes);
    memset(out1, kSentinel, kOutputBytes);
    try_msync(bias, kBiasBytes, MS_SYNC);
    try_msync(out0, kOutputBytes, MS_SYNC);
    try_msync(out1, kOutputBytes, MS_SYNC);
    ins.push(lml::instr::bias(c.pool->phys(bias), kBankStrideBytes,
                              kBiasRowStrideBytes, kChannels, kOutH,
                              kOutW * 4));
    ins.push(lml::instr::write_axi(c.pool->phys(out0),
                                   kOutputChannelStrideBytes,
                                   kOutputRowStrideBytes, kChannels,
                                   kOutH, kOutW));
    ins.push(lml::instr::write_axi(c.pool->phys(out1),
                                   kOutputChannelStrideBytes,
                                   kOutputRowStrideBytes, kChannels,
                                   kOutH, kOutW));
    ins.push(lml::instr::mode(0x1));
    ins.push(lml::instr::mode(0x0));
    if (!run_stream(c, ins, out1, "write_without_new_bias_probe")) return false;
    const bool out0_ok = check_constant(out0, 7, "write_without_new_bias out0");
    const bool out1_same = check_constant(out1, 7, "write_without_new_bias out1");
    printf("  NOTE: out1 PASS means WRITE_AXI did not clear psum by itself.\n");
    return out0_ok && out1_same;
}

static bool case_bias_between_writes(ctx& c, lml::instruction_buffer& ins,
                                     uint8_t* bias0, uint8_t* bias1,
                                     uint8_t* out0, uint8_t* out1) {
    ins.clear();
    fill_bias_constant(bias0, 256 * 9);
    fill_bias_constant(bias1, 256 * 2);
    memset(out0, kSentinel, kOutputBytes);
    memset(out1, kSentinel, kOutputBytes);
    try_msync(bias0, kBiasBytes, MS_SYNC);
    try_msync(bias1, kBiasBytes, MS_SYNC);
    try_msync(out0, kOutputBytes, MS_SYNC);
    try_msync(out1, kOutputBytes, MS_SYNC);

    ins.push(lml::instr::bias(c.pool->phys(bias0), kBankStrideBytes,
                              kBiasRowStrideBytes, kChannels, kOutH,
                              kOutW * 4));
    ins.push(lml::instr::write_axi(c.pool->phys(out0),
                                   kOutputChannelStrideBytes,
                                   kOutputRowStrideBytes, kChannels,
                                   kOutH, kOutW));
    ins.push(lml::instr::bias(c.pool->phys(bias1), kBankStrideBytes,
                              kBiasRowStrideBytes, kChannels, kOutH,
                              kOutW * 4));
    ins.push(lml::instr::write_axi(c.pool->phys(out1),
                                   kOutputChannelStrideBytes,
                                   kOutputRowStrideBytes, kChannels,
                                   kOutH, kOutW));
    ins.push(lml::instr::mode(0x1));
    ins.push(lml::instr::mode(0x0));

    if (!run_stream(c, ins, out1, "bias_between_writes")) return false;
    const bool out0_ok = check_constant(out0, 9, "bias_between_writes out0");
    const bool out1_ok = check_constant(out1, 2, "bias_between_writes out1");
    return out0_ok && out1_ok;
}

static void push_channel_kernel_regs(lml::instruction_buffer& ins,
                                     int8_t base) {
    int8_t kernel[9] = {};
    for (uint32_t co = 0; co < kChannels; ++co) {
        for (uint32_t ci = 0; ci < kChannels; ++ci) {
            memset(kernel, 0, sizeof(kernel));
            kernel[0] = static_cast<int8_t>(base + co * 4 + ci);
            ins.push(lml::instr::regs(static_cast<uint8_t>(co * kChannels + ci),
                                      pack_kernel_lo(kernel),
                                      pack_kernel_hi(kernel)));
        }
    }
}

static bool case_varied_kernel_reg_map(ctx& c, lml::instruction_buffer& ins,
                                       uint8_t* in0, uint8_t* in1,
                                       uint8_t* bias, uint8_t* out) {
    ins.clear();
    memset(in0, 0, kInputBytes);
    memset(in1, 0, kInputBytes);
    for (uint32_t ci = 0; ci < kChannels; ++ci) {
        for (uint32_t y = 0; y < kInH; ++y) {
            for (uint32_t x = 0; x < kInW; ++x) {
                in0[input_offset(ci, y, x)] = static_cast<uint8_t>(8 + ci);
                in1[input_offset(ci, y, x)] = static_cast<uint8_t>(12 + ci);
            }
        }
    }
    fill_bias_constant(bias, 0);
    memset(out, kSentinel, kOutputBytes);
    try_msync(in0, kInputBytes, MS_SYNC);
    try_msync(in1, kInputBytes, MS_SYNC);
    try_msync(bias, kBiasBytes, MS_SYNC);
    try_msync(out, kOutputBytes, MS_SYNC);

    ins.push(lml::instr::config_regs(kBankStrideBytes, kInputRowStrideBytes,
                                     kChannels, kInH, kInW));
    ins.push(lml::instr::bias(c.pool->phys(bias), kBankStrideBytes,
                              kBiasRowStrideBytes, kChannels, kOutH,
                              kOutW * 4));
    push_channel_kernel_regs(ins, 1);
    ins.push(lml::instr::read_conv(true, c.pool->phys(in0), false, 0xf, 0xf,
                                   kInH, kInW));
    ins.push(lml::instr::read_conv(false, c.pool->phys(in0), true, 0xf, 0xf,
                                   kInH, kInW));
    ins.push(lml::instr::mode(0x1));
    push_channel_kernel_regs(ins, 17);
    ins.push(lml::instr::read_conv(true, c.pool->phys(in1), false, 0xf, 0xf,
                                   kInH, kInW));
    ins.push(lml::instr::read_conv(false, c.pool->phys(in1), true, 0xf, 0xf,
                                   kInH, kInW));
    ins.push(lml::instr::write_axi(c.pool->phys(out),
                                   kOutputChannelStrideBytes,
                                   kOutputRowStrideBytes, kChannels,
                                   kOutH, kOutW));
    ins.push(lml::instr::bias(c.pool->phys(bias), kBankStrideBytes,
                              kBiasRowStrideBytes, kChannels, kOutH,
                              kOutW * 4));
    ins.push(lml::instr::mode(0x1));
    ins.push(lml::instr::mode(0x0));

    if (!run_stream(c, ins, out, "varied_kernel_reg_map")) return false;

    for (uint32_t co = 0; co < kChannels; ++co) {
        int32_t acc = 0;
        for (uint32_t ci = 0; ci < kChannels; ++ci) {
            acc += static_cast<int32_t>(8 + ci) *
                   static_cast<int32_t>(1 + co * 4 + ci);
            acc += static_cast<int32_t>(12 + ci) *
                   static_cast<int32_t>(17 + co * 4 + ci);
        }
        const uint8_t exp = quant_from_acc(acc);
        for (uint32_t y = 0; y < kOutH; ++y) {
            for (uint32_t x = 0; x < kOutW; ++x) {
                const uint8_t got = load8(out + output_offset(co, y, x));
                if (got != exp) {
                    fprintf(stderr,
                            "  varied_kernel_reg_map mismatch co=%u y=%u x=%u got=0x%02x exp=0x%02x acc=%" PRId32 "\n",
                            co, y, x, got, exp, acc);
                    return false;
                }
            }
        }
        printf("  PASS varied_kernel_reg_map co=%u constant=0x%02x\n", co, exp);
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    uint32_t timeout_ms = 5000;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--timeout-ms") == 0 && i + 1 < argc) {
            timeout_ms = static_cast<uint32_t>(strtoul(argv[++i], nullptr, 0));
        } else {
            fprintf(stderr, "Usage: %s [--timeout-ms N]\n", argv[0]);
            return 2;
        }
    }

    const int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem");
        return 1;
    }
    mmio_region exch;
    mmio_region ctrl;
    if (!exch.map(fd, kExchPhysAddr, kExchSize)) {
        perror("mmap exchange DDR");
        close(fd);
        return 1;
    }
    if (!ctrl.map(fd, kCtrlPhysAddr, kCtrlSize)) {
        perror("mmap acc_conv control");
        close(fd);
        return 1;
    }
    lml::exch_pool pool(exch.addr, kExchPhysAddr, kExchSize);
    if (!pool.valid()) {
        fprintf(stderr, "tlsf_create_with_pool failed\n");
        close(fd);
        return 1;
    }

    ctx c{static_cast<volatile uint32_t*>(ctrl.addr), &pool, timeout_ms};
    lml::instruction_buffer ins(pool, kMaxInstructions);
    auto* bias0 = static_cast<uint8_t*>(pool.alloc(kBiasBytes, 64));
    auto* bias1 = static_cast<uint8_t*>(pool.alloc(kBiasBytes, 64));
    auto* in0 = static_cast<uint8_t*>(pool.alloc(kInputBytes, 64));
    auto* in1 = static_cast<uint8_t*>(pool.alloc(kInputBytes, 64));
    auto* out0 = static_cast<uint8_t*>(pool.alloc(kOutputBytes, 64));
    auto* out1 = static_cast<uint8_t*>(pool.alloc(kOutputBytes, 64));
    if (!ins.valid() || !bias0 || !bias1 || !in0 || !in1 || !out0 || !out1) {
        fprintf(stderr, "allocation failed\n");
        close(fd);
        return 1;
    }

    printf("acc_conv focused unit tests\n");
    printf("  ins=0x%08x bias0=0x%08x bias1=0x%08x in0=0x%08x in1=0x%08x out0=0x%08x out1=0x%08x\n",
           ins.phys_addr(), pool.phys(bias0), pool.phys(bias1),
           pool.phys(in0), pool.phys(in1), pool.phys(out0), pool.phys(out1));

    bool ok = true;
    ok = case_bias_write(c, ins, bias0, out0, 256 * 3, "bias_write_value_3") && ok;
    ok = case_bias_write(c, ins, bias1, out0, -256 * 5, "bias_overwrite_value_minus5") && ok;
    ok = case_write_without_new_bias(c, ins, bias0, out0, out1) && ok;
    ok = case_bias_between_writes(c, ins, bias0, bias1, out0, out1) && ok;
    ok = case_two_group_kernel_change(c, ins, in0, in1, bias0, out0) && ok;
    ok = case_varied_kernel_reg_map(c, ins, in0, in1, bias0, out0) && ok;

    close(fd);
    if (!ok) return 1;
    printf("PASS: focused unit tests completed\n");
    return 0;
}
