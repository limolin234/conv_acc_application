#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <vector>

#include "lml_acc_conv.hpp"
#include "lml_accelerator.hpp"
#include "lml_conv_tile.hpp"
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
constexpr uint32_t kCtrlBusy = 1u << 2;

constexpr uint32_t kInW = lml::kDefaultConvInputW;
constexpr uint32_t kInH = lml::kDefaultConvInputH;
constexpr uint32_t kOutW = kInW - 2;
constexpr uint32_t kOutH = kInH - 2;
constexpr uint32_t kChannels = 4;
constexpr uint32_t kWordsPerInputRow = (kInW + 15) / 16;
constexpr uint32_t kWordsPerOutputRow = (kOutW + 3) / 4;
constexpr uint32_t kBankStrideBytes = 512 * 16;
constexpr uint32_t kInputRowStrideBytes = kInW;
constexpr uint32_t kBiasRowStrideBytes = kOutW * 4;
constexpr uint32_t kOutputRowStrideBytes = kOutW;
constexpr uint32_t kOutputChannelStrideBytes = kOutW * kOutH;
constexpr size_t kInputBytes = kChannels * kBankStrideBytes;
constexpr size_t kBiasBytes = kChannels * kBankStrideBytes;
constexpr size_t kOutputBytes = kChannels * kOutputChannelStrideBytes;
constexpr size_t kTensorSlackBytes = 65536;
constexpr uint32_t kDefaultQuantRounds = 8;
constexpr uint32_t kDefaultThroughputBatches = 16;
constexpr uint32_t kDefaultThroughputTasks = 255;
constexpr uint32_t kDefaultConvThroughputBatches = 8;
constexpr uint32_t kDefaultConvThroughputGroups = 4;
constexpr uint32_t kMaxVerifySamples = 4096;
constexpr uint32_t kMaxInstructions = 512;
constexpr uint8_t kSentinel = 0xa5;

struct mmio_region {
    void* addr = MAP_FAILED;
    size_t size = 0;

    ~mmio_region() {
        if (addr != MAP_FAILED) {
            munmap(addr, size);
        }
    }

    bool map(int fd, uint32_t phys, size_t bytes) {
        size = bytes;
        addr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                    static_cast<off_t>(phys));
        return addr != MAP_FAILED;
    }
};

struct tensors {
    lml::instruction_buffer ins;
    uint8_t* input = nullptr;
    uint8_t* bias = nullptr;
    uint8_t* out = nullptr;

    explicit tensors(lml::exch_pool& pool)
        : ins(pool, kMaxInstructions),
          input(static_cast<uint8_t*>(pool.alloc(kInputBytes + kTensorSlackBytes, 64))),
          bias(static_cast<uint8_t*>(pool.alloc(kBiasBytes + kTensorSlackBytes, 64))),
          out(static_cast<uint8_t*>(pool.alloc(kOutputBytes + kTensorSlackBytes, 64))) {}

    bool valid() const {
        return ins.valid() && input && bias && out;
    }
};

static void usage(const char* argv0) {
    fprintf(stderr,
            "Usage: %s [--quant | --conv | --all | --throughput | --read-bench | --write-bench | --conv-throughput | --conv-throughput-sweep | --writeback-basic | --partial-writeback | --overlap-compare | --yolo-cpp | --yolo-bench] [--timeout-ms N] [--seed N] [--rounds N] [--batches N] [--tasks N] [--groups N] [--sample-verify] [--verify-all] [--dump DIR]\n",
            argv0);
}

static uint64_t monotonic_ms() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000u +
           static_cast<uint64_t>(ts.tv_nsec / 1000000u);
}

static uint64_t monotonic_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
           static_cast<uint64_t>(ts.tv_nsec);
}

static uint32_t xorshift32(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

static uint8_t rand_u8(uint32_t& s, uint8_t mod) {
    return static_cast<uint8_t>(xorshift32(s) % mod);
}

static uint32_t rand_aligned_offset(uint32_t& s, uint32_t max_offset,
                                    uint32_t align) {
    const uint32_t slots = max_offset / align;
    return (xorshift32(s) % (slots + 1u)) * align;
}

static int8_t rand_i8_small(uint32_t& s) {
    return static_cast<int8_t>(static_cast<int>(rand_u8(s, 5)) - 2);
}

static void store32_le(uint8_t* p, uint32_t value) {
    p[0] = static_cast<uint8_t>(value);
    p[1] = static_cast<uint8_t>(value >> 8);
    p[2] = static_cast<uint8_t>(value >> 16);
    p[3] = static_cast<uint8_t>(value >> 24);
}

static uint8_t load8(const uint8_t* p) {
    return *reinterpret_cast<const volatile uint8_t*>(p);
}

static void print_bytes(const char* label, const uint8_t* data, size_t bytes) {
    printf("%s:", label);
    for (size_t i = 0; i < bytes; ++i) {
        if ((i % 16) == 0) {
            printf("\n  %02zu:", i);
        }
        printf(" %02x", load8(data + i));
    }
    printf("\n");
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

static bool wait_output_change(uint8_t* out, uint8_t sentinel,
                               uint32_t timeout_ms) {
    const uint64_t deadline = monotonic_ms() + timeout_ms;
    while (monotonic_ms() < deadline) {
        try_msync(out, 4096, MS_INVALIDATE);
        for (uint32_t i = 0; i < 80; ++i) {
            if (load8(out + i) != sentinel) {
                return true;
            }
        }
        usleep(1000);
    }
    return false;
}

static bool wait_start_clear(volatile uint32_t* regs, uint32_t timeout_ms) {
    const uint64_t deadline = monotonic_ms() + timeout_ms;
    while (monotonic_ms() < deadline) {
        if ((regs[kRegCtrl] & kCtrlStart) == 0) {
            return true;
        }
        usleep(1000);
    }
    return false;
}

static bool wait_busy_done(volatile uint32_t* regs, uint32_t timeout_ms) {
    const uint64_t deadline = monotonic_ms() + timeout_ms;
    while (monotonic_ms() < deadline) {
        const uint32_t ctrl = regs[kRegCtrl];
        if ((ctrl & (kCtrlStart | kCtrlBusy)) == 0) {
            return true;
        }
        usleep(50);
    }
    return false;
}

static void write_regs_and_start(volatile uint32_t* regs,
                                 const lml::instruction_buffer& ins) {
    regs[kRegInsAddr] = ins.phys_addr();              // 0x40000004
    regs[kRegInsLen] = static_cast<uint32_t>(ins.size()); // 0x40000008
    regs[kRegCtrl] = kCtrlStart;                      // 0x40000000 bit1
}

static void print_instruction_stream(const lml::instruction_buffer& ins) {
    printf("  instruction stream:\n");
    printf("    base=0x%08x bytes=%zu count=%zu\n",
           ins.phys_addr(), ins.size() * sizeof(lml::instruction), ins.size());
    for (size_t i = 0; i < ins.size(); ++i) {
        printf("    [%02zu] phys=0x%08x hi=0x%016" PRIx64 " lo=0x%016" PRIx64 "\n",
               i,
               ins.phys_addr() + static_cast<uint32_t>(i * sizeof(lml::instruction)),
               ins[i].hi,
               ins[i].lo);
    }
}

static void print_addr_range(const char* name, uint32_t base, size_t bytes) {
    printf("    %-5s base=0x%08x end=0x%08x bytes=%zu\n",
           name, base, base + static_cast<uint32_t>(bytes) - 1u, bytes);
}

static bool write_file(const char* path, const void* data, size_t bytes) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        perror(path);
        return false;
    }
    const size_t n = fwrite(data, 1, bytes, f);
    if (n != bytes) {
        perror(path);
        fclose(f);
        return false;
    }
    if (fclose(f) != 0) {
        perror(path);
        return false;
    }
    return true;
}

static bool path_join(char* out, size_t out_size, const char* dir,
                      const char* name) {
    const int n = snprintf(out, out_size, "%s/%s", dir, name);
    return n > 0 && static_cast<size_t>(n) < out_size;
}

static bool dump_file(const char* dir, const char* name,
                      const void* data, size_t bytes) {
    char path[512];
    if (!path_join(path, sizeof(path), dir, name)) {
        fprintf(stderr, "dump path too long: %s/%s\n", dir, name);
        return false;
    }
    return write_file(path, data, bytes);
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

static void push_common_regs(lml::instruction_buffer& ins,
                             const int8_t kernel[9]) {
    for (uint8_t i = 0; i < kChannels * kChannels; ++i) {
        ins.push(lml::instr::regs(i, pack_kernel_lo(kernel), pack_kernel_hi(kernel)));
    }
    ins.push(lml::instr::config_regs(kBankStrideBytes, kInputRowStrideBytes,
                                     kChannels, kInH, kInW));
}

static void push_kernel_regs(lml::instruction_buffer& ins,
                             const int8_t* weights, uint32_t cin,
                             uint32_t oc_base, uint32_t ic_base) {
    for (uint32_t co = 0; co < kChannels; ++co) {
        for (uint32_t ci = 0; ci < kChannels; ++ci) {
            const int8_t* k = weights + ((oc_base + co) * cin + ic_base + ci) * 9;
            ins.push(lml::instr::regs(static_cast<uint8_t>(co * kChannels + ci),
                                      pack_kernel_lo(k), pack_kernel_hi(k)));
        }
    }
}

static void fill_kernel_reg_pack(lml::instruction* regs,
                                 const int8_t kernel[9]) {
    for (uint8_t i = 0; i < kChannels * kChannels; ++i) {
        regs[i] = lml::instr::regs(i, pack_kernel_lo(kernel),
                                   pack_kernel_hi(kernel));
    }
}

static uint8_t quant_from_acc(int32_t acc) {
    return lml::yolo::quantize_acc(acc);
}

static uint32_t input_offset(uint32_t c, uint32_t y, uint32_t x) {
    return c * kBankStrideBytes + y * kInputRowStrideBytes + x;
}

static uint32_t bias_ddr_offset(uint32_t c, uint32_t y, uint32_t x) {
    return c * kBankStrideBytes + y * kBiasRowStrideBytes + x * 4;
}

static uint32_t output_offset(uint32_t c, uint32_t y, uint32_t x) {
    return c * kOutputChannelStrideBytes + y * kOutputRowStrideBytes + x;
}

static uint32_t output_bram_i32_offset(uint32_t c, uint32_t y, uint32_t x) {
    const uint32_t word = y * kWordsPerOutputRow + x / 4;
    return c * 512 * 4 + word * 4 + (x % 4);
}

static int32_t conv3x3_expected(const uint8_t* input, const int8_t kernel[9],
                                uint32_t co, uint32_t y, uint32_t x) {
    (void)co;
    int32_t acc = 0;
    for (uint32_t ci = 0; ci < kChannels; ++ci) {
        for (uint32_t ky = 0; ky < 3; ++ky) {
            for (uint32_t kx = 0; kx < 3; ++kx) {
                const int8_t in = static_cast<int8_t>(
                    input[input_offset(ci, y + ky, x + kx)]);
                acc += static_cast<int32_t>(in) *
                       static_cast<int32_t>(kernel[ky * 3 + kx]);
            }
        }
    }
    return acc;
}

static void print_conv_mismatch_context(const uint8_t* input,
                                        const uint8_t* out,
                                        const uint8_t* expected,
                                        const int8_t kernel[9],
                                        uint32_t co, uint32_t y,
                                        uint32_t x) {
    printf("  mismatch context co=%u y=%u x=%u:\n", co, y, x);
    printf("    nearby x:");
    const uint32_t x0 = (x > 3u) ? (x - 3u) : 0u;
    const uint32_t x1 = std::min(kOutW - 1u, x + 3u);
    for (uint32_t xx = x0; xx <= x1; ++xx) {
        const uint32_t off = output_offset(co, y, xx);
        const int32_t acc = conv3x3_expected(input, kernel, co, y, xx);
        printf(" x%u got=%02x exp=%02x acc=%" PRId32,
               xx, load8(out + off), expected[off], acc);
    }
    printf("\n");

    printf("    shift candidates:");
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            const int yy = static_cast<int>(y) + dy;
            const int xx = static_cast<int>(x) + dx;
            if (yy < 0 || xx < 0 ||
                yy >= static_cast<int>(kOutH) ||
                xx >= static_cast<int>(kOutW)) {
                continue;
            }
            const uint32_t off = output_offset(co, static_cast<uint32_t>(yy),
                                               static_cast<uint32_t>(xx));
            printf(" dy=%d dx=%d exp=%02x", dy, dx, expected[off]);
        }
    }
    printf("\n");
}

static bool test_one_quant_round(volatile uint32_t* regs, lml::exch_pool& pool,
                                 tensors& t, uint32_t timeout_ms,
                                 uint32_t& seed, uint32_t round) {
    t.ins.clear();
    memset(t.input, 0, kInputBytes + kTensorSlackBytes);
    memset(t.bias, 0, kBiasBytes + kTensorSlackBytes);
    memset(t.out, 0xa5, kOutputBytes + kTensorSlackBytes);

    const uint32_t bias_shift = rand_aligned_offset(seed, 2048, 64);
    const uint32_t out_shift = rand_aligned_offset(seed, 512, 1);
    uint8_t* bias_base = t.bias + bias_shift;
    uint8_t* out_base = t.out + out_shift;
    uint8_t expected[kChannels * kOutW * kOutH];

    for (uint32_t i = 0; i < kChannels * kOutW * kOutH; ++i) {
        const int32_t acc = static_cast<int32_t>(rand_u8(seed, 255)) - 127;
        store32_le(bias_base + bias_ddr_offset(
                       i / (kOutW * kOutH),
                       (i / kOutW) % kOutH,
                       i % kOutW),
                   static_cast<uint32_t>(acc << 8));
        expected[i] = quant_from_acc(acc << 8);
    }

    t.ins.push(lml::instr::bias(pool.phys(bias_base), kBankStrideBytes,
                                kBiasRowStrideBytes, kChannels, kOutH,
                                kOutW * 4));
    t.ins.push(lml::instr::write_axi(pool.phys(out_base),
                                     kOutputChannelStrideBytes,
                                     kOutputRowStrideBytes, kChannels,
                                     kOutH, kOutW));
    t.ins.push(lml::instr::mode(0x1));
    t.ins.push(lml::instr::mode(0x0));

    try_msync(pool.virt_base(), pool.bytes(), MS_SYNC);

    printf("quant-bias round %u:\n", round);
    printf("  tensor ranges for AXI compare:\n");
    print_addr_range("bias", pool.phys(bias_base), kBiasBytes);
    print_addr_range("out", pool.phys(out_base), kOutputBytes);
    print_instruction_stream(t.ins);

    write_regs_and_start(regs, t.ins);
    wait_busy_done(regs, timeout_ms);
    wait_output_change(out_base, 0xa5, timeout_ms);

    try_msync(out_base, kOutputBytes, MS_INVALIDATE);
    print_bytes("  output sample", out_base, 80);
    printf("  reg0=0x%08x\n", regs[kRegCtrl]);

    for (uint32_t i = 0; i < kChannels * kOutW * kOutH; ++i) {
        const uint32_t off = output_offset(i / (kOutW * kOutH),
                                           (i / kOutW) % kOutH,
                                           i % kOutW);
        const uint8_t got = load8(out_base + off);
        if (got != expected[i]) {
            fprintf(stderr,
                    "quant mismatch round=%u index=%u out_off=%u: got=0x%02x exp=0x%02x\n",
                    round, i, off, got, expected[i]);
            return false;
        }
    }

    printf("  PASS\n");
    return true;
}

static void fill_random_bias(uint8_t* bias_base, uint8_t* expected,
                             uint32_t& seed) {
    for (uint32_t i = 0; i < kChannels * kOutW * kOutH; ++i) {
        const int32_t acc = static_cast<int32_t>(rand_u8(seed, 255)) - 127;
        store32_le(bias_base + bias_ddr_offset(
                       i / (kOutW * kOutH),
                       (i / kOutW) % kOutH,
                       i % kOutW),
                   static_cast<uint32_t>(acc << 8));
        expected[i] = quant_from_acc(acc << 8);
    }
}

static bool check_output_block(const uint8_t* out_base,
                               const uint8_t* expected,
                               uint32_t round) {
    for (uint32_t i = 0; i < kChannels * kOutW * kOutH; ++i) {
        const uint32_t off = output_offset(i / (kOutW * kOutH),
                                           (i / kOutW) % kOutH,
                                           i % kOutW);
        const uint8_t got = load8(out_base + off);
        if (got != expected[i]) {
            fprintf(stderr,
                    "quant concat mismatch round=%u index=%u out_off=%u: got=0x%02x exp=0x%02x\n",
                    round, i, off, got, expected[i]);
            return false;
        }
    }
    return true;
}

static bool test_quant_writeback(volatile uint32_t* regs, lml::exch_pool& pool,
                                 tensors& t, uint32_t timeout_ms,
                                 uint32_t& seed, uint32_t rounds) {
    if (rounds == 0) {
        return true;
    }

    const size_t total_out_bytes = static_cast<size_t>(rounds) * kOutputBytes;
    if (total_out_bytes + 2 > kOutputBytes + kTensorSlackBytes) {
        fprintf(stderr,
                "rounds=%u needs %zu output bytes, available %zu; reduce --rounds\n",
                rounds, total_out_bytes + 2, kOutputBytes + kTensorSlackBytes);
        return false;
    }

    printf("quant-bias concat test: rounds=%u seed_start=0x%08x\n",
           rounds, seed);
    printf("  each output block bytes=%zu, blocks are byte-adjacent\n",
           kOutputBytes);

    memset(t.input, 0, kInputBytes + kTensorSlackBytes);
    memset(t.bias, 0, kBiasBytes + kTensorSlackBytes);
    memset(t.out, kSentinel, kOutputBytes + kTensorSlackBytes);

    uint8_t* expected_all = static_cast<uint8_t*>(
        malloc(static_cast<size_t>(rounds) * kOutputBytes));
    if (!expected_all) {
        fprintf(stderr, "failed to allocate expected output buffer\n");
        return false;
    }

    for (uint32_t r = 0; r < rounds; ++r) {
        t.ins.clear();

        const uint32_t bias_shift = rand_aligned_offset(seed, 2048, 64);
        uint8_t* bias_base = t.bias + bias_shift;
        uint8_t* out_base = t.out + static_cast<size_t>(r) * kOutputBytes;
        uint8_t* expected_block = expected_all + static_cast<size_t>(r) * kOutputBytes;

        memset(bias_base, 0, kBiasBytes);
        fill_random_bias(bias_base, expected_block, seed);

        t.ins.push(lml::instr::bias(pool.phys(bias_base), kBankStrideBytes,
                                    kBiasRowStrideBytes, kChannels, kOutH,
                                    kOutW * 4));
        t.ins.push(lml::instr::write_axi(pool.phys(out_base),
                                         kOutputChannelStrideBytes,
                                         kOutputRowStrideBytes, kChannels,
                                         kOutH, kOutW));
        t.ins.push(lml::instr::mode(0x1));
        t.ins.push(lml::instr::mode(0x0));

        try_msync(pool.virt_base(), pool.bytes(), MS_SYNC);

        printf("quant-bias concat round %u:\n", r);
        printf("  tensor ranges for AXI compare:\n");
        print_addr_range("bias", pool.phys(bias_base), kBiasBytes);
        print_addr_range("out", pool.phys(out_base), kOutputBytes);
        print_instruction_stream(t.ins);

        write_regs_and_start(regs, t.ins);
        wait_busy_done(regs, timeout_ms);
        wait_output_change(out_base, kSentinel, timeout_ms);

        try_msync(t.out, total_out_bytes + 2, MS_INVALIDATE);
        print_bytes("  output sample", out_base, 80);
        printf("  reg0=0x%08x\n", regs[kRegCtrl]);

        if (!check_output_block(out_base, expected_block, r)) {
            free(expected_all);
            return false;
        }

        for (uint32_t prev = 0; prev < r; ++prev) {
            uint8_t* prev_base = t.out + static_cast<size_t>(prev) * kOutputBytes;
            uint8_t* prev_expected = expected_all + static_cast<size_t>(prev) * kOutputBytes;
            if (!check_output_block(prev_base, prev_expected, prev)) {
                fprintf(stderr, "previous output block %u changed after round %u\n",
                        prev, r);
                free(expected_all);
                return false;
            }
        }

        if (load8(t.out + total_out_bytes) != kSentinel ||
            load8(t.out + total_out_bytes + 1) != kSentinel) {
            fprintf(stderr,
                    "post-output sentinel clobbered after adjacent blocks\n");
            free(expected_all);
            return false;
        }

        printf("  PASS\n");
    }

    free(expected_all);
    return true;
}

static bool test_partial_writeback_case(volatile uint32_t* regs,
                                        lml::exch_pool& pool, tensors& t,
                                        uint32_t timeout_ms, uint32_t H,
                                        uint32_t W, const char* name) {
    memset(t.bias, 0, kBiasBytes);
    memset(t.out, kSentinel, kOutputBytes + kTensorSlackBytes);

    for (uint32_t c = 0; c < kChannels; ++c) {
        for (uint32_t y = 0; y < kOutH; ++y) {
            for (uint32_t x = 0; x < kOutW; ++x) {
                const int32_t acc = static_cast<int32_t>(
                    (c * 40u + y * kOutW + x - 64) * 256);
                store32_le(t.bias + bias_ddr_offset(c, y, x),
                           static_cast<uint32_t>(acc));
            }
        }
    }

    t.ins.clear();
    t.ins.push(lml::instr::bias(pool.phys(t.bias), kBankStrideBytes,
                                kBiasRowStrideBytes, kChannels,
                                static_cast<uint8_t>(H),
                                static_cast<uint16_t>(W * 4)));
    t.ins.push(lml::instr::write_axi(pool.phys(t.out),
                                     kOutputChannelStrideBytes,
                                     kOutputRowStrideBytes, kChannels,
                                     static_cast<uint8_t>(H),
                                     static_cast<uint16_t>(W)));
    t.ins.push(lml::instr::mode(0x1));
    t.ins.push(lml::instr::mode(0x0));

    try_msync(pool.virt_base(), pool.bytes(), MS_SYNC);
    write_regs_and_start(regs, t.ins);
    wait_busy_done(regs, timeout_ms);
    if (!wait_output_change(t.out, kSentinel, timeout_ms)) {
        fprintf(stderr, "partial-writeback %s timeout reg0=0x%08x\n",
                name, regs[kRegCtrl]);
        return false;
    }
    try_msync(t.out, kOutputBytes + kTensorSlackBytes, MS_INVALIDATE);

    for (uint32_t c = 0; c < kChannels; ++c) {
        for (uint32_t y = 0; y < kOutH; ++y) {
            for (uint32_t x = 0; x < kOutW; ++x) {
                const uint32_t off = output_offset(c, y, x);
                const uint8_t got = load8(t.out + off);
                const bool should_write = (y < H && x < W);
                if (should_write) {
                    const int32_t acc = static_cast<int32_t>(
                        (c * 40u + y * kOutW + x - 64) * 256);
                    const uint8_t exp = quant_from_acc(acc);
                    if (got != exp) {
                        bool found = false;
                        uint32_t src_y = 0;
                        uint32_t src_x = 0;
                        for (uint32_t yy = 0; yy < kOutH && !found; ++yy) {
                            for (uint32_t xx = 0; xx < kOutW; ++xx) {
                                const int32_t src_acc = static_cast<int32_t>(
                                    (c * 40u + yy * kOutW + xx - 64) * 256);
                                if (quant_from_acc(src_acc) == got) {
                                    src_y = yy;
                                    src_x = xx;
                                    found = true;
                                    break;
                                }
                            }
                        }
                        fprintf(stderr,
                                "partial-writeback %s value mismatch c=%u y=%u x=%u off=%u got=0x%02x exp=0x%02x%s%u%s%u%s\n",
                                name, c, y, x, off, got, exp,
                                found ? " got_matches_src_y=" : "",
                                found ? src_y : 0,
                                found ? " src_x=" : "",
                                found ? src_x : 0,
                                found ? "" : " got_has_no_source_match");
                        return false;
                    }
                } else if (got != kSentinel) {
                    fprintf(stderr,
                            "partial-writeback %s clobber c=%u y=%u x=%u off=%u got=0x%02x sentinel=0x%02x\n",
                            name, c, y, x, off, got, kSentinel);
                    return false;
                }
            }
        }
    }

    printf("  %-14s H=%u W=%u PASS\n", name, H, W);
    return true;
}

static bool test_partial_writeback(volatile uint32_t* regs, lml::exch_pool& pool,
                                   tensors& t, uint32_t timeout_ms) {
    printf("partial-writeback test:\n");
    bool ok = true;
    ok = test_partial_writeback_case(regs, pool, t, timeout_ms, 30, 30,
                                     "full_30x30") && ok;
    ok = test_partial_writeback_case(regs, pool, t, timeout_ms, 30, 8,
                                     "right_30x8") && ok;
    ok = test_partial_writeback_case(regs, pool, t, timeout_ms, 8, 30,
                                     "bottom_8x30") && ok;
    ok = test_partial_writeback_case(regs, pool, t, timeout_ms, 8, 8,
                                     "corner_8x8") && ok;
    ok = test_partial_writeback_case(regs, pool, t, timeout_ms, 28, 28,
                                     "p3_edge_28") && ok;
    if (ok) printf("  PASS\n");
    return ok;
}

static bool test_writeback_basic(volatile uint32_t* regs, lml::exch_pool& pool,
                                 tensors& t, uint32_t timeout_ms) {
    memset(t.bias, 0, kBiasBytes);
    memset(t.out, kSentinel, kOutputBytes);
    for (uint32_t c = 0; c < kChannels; ++c) {
        for (uint32_t y = 0; y < kOutH; ++y) {
            for (uint32_t x = 0; x < kOutW; ++x) {
                const int32_t acc = static_cast<int32_t>(
                    (static_cast<int32_t>(c * 10 + y + x) - 16) * 256);
                store32_le(t.bias + bias_ddr_offset(c, y, x),
                           static_cast<uint32_t>(acc));
            }
        }
    }

    t.ins.clear();
    t.ins.push(lml::instr::bias(pool.phys(t.bias), kBankStrideBytes,
                                kBiasRowStrideBytes, kChannels, kOutH,
                                kOutW * 4));
    t.ins.push(lml::instr::write_axi(pool.phys(t.out),
                                     kOutputChannelStrideBytes,
                                     kOutputRowStrideBytes, kChannels,
                                     kOutH, kOutW));
    t.ins.push(lml::instr::mode(0x1));
    t.ins.push(lml::instr::mode(0x0));

    try_msync(pool.virt_base(), pool.bytes(), MS_SYNC);
    printf("writeback-basic:\n");
    printf("  reg0_before=0x%08x\n", regs[kRegCtrl]);
    print_addr_range("bias", pool.phys(t.bias), kBiasBytes);
    print_addr_range("out", pool.phys(t.out), kOutputBytes);
    print_instruction_stream(t.ins);

    write_regs_and_start(regs, t.ins);
    if (!wait_busy_done(regs, timeout_ms)) {
        fprintf(stderr, "writeback-basic timeout reg0=0x%08x\n", regs[kRegCtrl]);
        return false;
    }
    try_msync(t.out, kOutputBytes, MS_INVALIDATE);
    printf("  reg0_after=0x%08x\n", regs[kRegCtrl]);
    print_bytes("  output sample", t.out, 80);

    for (uint32_t c = 0; c < kChannels; ++c) {
        for (uint32_t y = 0; y < kOutH; ++y) {
            for (uint32_t x = 0; x < kOutW; ++x) {
                const uint32_t off = output_offset(c, y, x);
                const int32_t acc = static_cast<int32_t>(
                    (static_cast<int32_t>(c * 10 + y + x) - 16) * 256);
                const uint8_t exp = quant_from_acc(acc);
                const uint8_t got = load8(t.out + off);
                if (got != exp) {
                    fprintf(stderr,
                            "writeback-basic mismatch c=%u y=%u x=%u off=%u got=0x%02x exp=0x%02x\n",
                            c, y, x, off, got, exp);
                    return false;
                }
            }
        }
    }
    printf("  PASS\n");
    return true;
}

static bool dump_conv_case(const char* dir, lml::exch_pool& pool, tensors& t,
                           const int8_t kernel[9],
                           const uint8_t* expected,
                           uint32_t seed, uint32_t reg0) {
    if (!dir) {
        return true;
    }
    if (mkdir(dir, 0777) != 0 && errno != EEXIST) {
        perror(dir);
        return false;
    }

    if (!dump_file(dir, "input.bin", t.input, kInputBytes)) return false;
    if (!dump_file(dir, "bias.bin", t.bias, kBiasBytes)) return false;
    if (!dump_file(dir, "kernel.bin", kernel, 9)) return false;
    if (!dump_file(dir, "output_hw.bin", t.out, kOutputBytes)) return false;
    if (!dump_file(dir, "output_expected.bin", expected, kOutputBytes)) return false;
    if (!dump_file(dir, "instructions.bin", t.ins.data(),
                   t.ins.size() * sizeof(lml::instruction))) return false;

    char meta_path[512];
    if (!path_join(meta_path, sizeof(meta_path), dir, "meta.txt")) {
        fprintf(stderr, "dump meta path too long\n");
        return false;
    }
    FILE* f = fopen(meta_path, "w");
    if (!f) {
        perror(meta_path);
        return false;
    }

    fprintf(f, "seed=0x%08x\n", seed);
    fprintf(f, "reg0=0x%08x\n", reg0);
    fprintf(f, "input_shape=4x%ux%u int8\n", kInH, kInW);
    fprintf(f, "bias_shape=4x%ux%u int32\n", kOutH, kOutW);
    fprintf(f, "output_shape=4x%ux%u int8\n", kOutH, kOutW);
    fprintf(f, "input_phys=0x%08x bytes=%zu\n", pool.phys(t.input), kInputBytes);
    fprintf(f, "bias_phys=0x%08x bytes=%zu\n", pool.phys(t.bias), kBiasBytes);
    fprintf(f, "out_phys=0x%08x bytes=%zu\n", pool.phys(t.out), kOutputBytes);
    fprintf(f, "instruction_phys=0x%08x count=%zu bytes=%zu\n",
            t.ins.phys_addr(), t.ins.size(),
            t.ins.size() * sizeof(lml::instruction));
    fprintf(f, "kernel=");
    for (uint32_t i = 0; i < 9; ++i) {
        fprintf(f, "%s%d", i ? " " : "", static_cast<int>(kernel[i]));
    }
    fprintf(f, "\n");
    for (size_t i = 0; i < t.ins.size(); ++i) {
        fprintf(f, "ins[%02zu].phys=0x%08x hi=0x%016" PRIx64 " lo=0x%016" PRIx64 "\n",
                i,
                t.ins.phys_addr() + static_cast<uint32_t>(i * sizeof(lml::instruction)),
                t.ins[i].hi,
                t.ins[i].lo);
    }

    if (fclose(f) != 0) {
        perror(meta_path);
        return false;
    }
    printf("  dumped conv files to %s\n", dir);
    return true;
}

static bool test_random_conv(volatile uint32_t* regs, lml::exch_pool& pool,
                             tensors& t, uint32_t timeout_ms,
                             uint32_t& seed, const char* dump_dir) {
    t.ins.clear();
    memset(t.input, 0, kInputBytes);
    memset(t.bias, 0, kBiasBytes);
    memset(t.out, 0xa5, kOutputBytes);

    int8_t kernel[9];
    for (uint32_t c = 0; c < kChannels; ++c) {
        for (uint32_t y = 0; y < kInH; ++y) {
            for (uint32_t x = 0; x < kInW; ++x) {
                const uint8_t v = rand_u8(seed, 8);
                t.input[input_offset(c, y, x)] = v;
            }
        }
    }
    for (uint32_t i = 0; i < 9; ++i) {
        kernel[i] = rand_i8_small(seed);
    }
    for (uint32_t c = 0; c < kChannels; ++c) {
        for (uint32_t y = 0; y < kOutH; ++y) {
            for (uint32_t x = 0; x < kOutW; ++x) {
                store32_le(t.bias + bias_ddr_offset(c, y, x), 0);
            }
        }
    }

    lml::instruction kernel_regs[kChannels * kChannels];
    fill_kernel_reg_pack(kernel_regs, kernel);
    lml::conv::tile_task task;
    task.input_phys = pool.phys(t.input);
    task.bias_phys = pool.phys(t.bias);
    task.output_phys = pool.phys(t.out);
    task.input_layout = {kBankStrideBytes, kInputRowStrideBytes};
    task.bias_layout = {kBankStrideBytes, kBiasRowStrideBytes};
    task.output_layout = {kOutputChannelStrideBytes, kOutputRowStrideBytes};
    task.shape = {kChannels, kInH, kInW, kOutH, kOutW};
    task.write_cfg = {1, 8, 0};
    task.kernels = {kernel_regs, kChannels * kChannels};
    task.sync_after = false;

    lml::conv::tile_program_builder builder(t.ins);
    builder.clear();
    if (!builder.add_tile(task) || !builder.finish()) {
        fprintf(stderr, "conv tile instruction buffer overflow\n");
        return false;
    }

    try_msync(pool.virt_base(), pool.bytes(), MS_SYNC);

    printf("random-conv:\n");
    printf("  tensor ranges for AXI compare:\n");
    print_addr_range("input", pool.phys(t.input), kInputBytes);
    print_addr_range("bias", pool.phys(t.bias), kBiasBytes);
    print_addr_range("out", pool.phys(t.out), kOutputBytes);
    print_instruction_stream(t.ins);
    printf("  kernel:");
    for (uint32_t i = 0; i < 9; ++i) {
        printf(" %d", static_cast<int>(kernel[i]));
    }
    printf("\n");

    write_regs_and_start(regs, t.ins);
    wait_busy_done(regs, timeout_ms);
    wait_output_change(t.out, 0xa5, timeout_ms);

    try_msync(t.out, kOutputBytes, MS_INVALIDATE);
    print_bytes("  output sample", t.out, 80);
    const uint32_t reg0 = regs[kRegCtrl];
    printf("  reg0=0x%08x\n", reg0);

    uint8_t* expected = static_cast<uint8_t*>(malloc(kOutputBytes));
    if (!expected) {
        fprintf(stderr, "failed to allocate expected conv output\n");
        return false;
    }

    for (uint32_t co = 0; co < kChannels; ++co) {
        for (uint32_t y = 0; y < kOutH; ++y) {
            for (uint32_t x = 0; x < kOutW; ++x) {
                const int32_t acc = conv3x3_expected(t.input, kernel, co, y, x);
                expected[output_offset(co, y, x)] = quant_from_acc(acc);
            }
        }
    }

    for (uint32_t co = 0; co < kChannels; ++co) {
        for (uint32_t y = 0; y < kOutH; ++y) {
            for (uint32_t x = 0; x < kOutW; ++x) {
                const uint32_t off = output_offset(co, y, x);
                const uint8_t exp = expected[off];
                const uint8_t got = load8(t.out + off);
                if (got != exp) {
                    const int32_t acc = conv3x3_expected(t.input, kernel, co, y, x);
                    fprintf(stderr,
                            "conv mismatch at co=%u y=%u x=%u off=%u: got=0x%02x exp=0x%02x acc=%" PRId32 "\n",
                            co, y, x, off, got, exp, acc);
                    print_conv_mismatch_context(t.input, t.out, expected,
                                                kernel, co, y, x);
                    dump_conv_case(dump_dir, pool, t, kernel, expected, seed, reg0);
                    free(expected);
                    return false;
                }
            }
        }
    }

    if (!dump_conv_case(dump_dir, pool, t, kernel, expected, seed, reg0)) {
        free(expected);
        return false;
    }
    free(expected);
    printf("  PASS\n");
    return true;
}

static bool wait_byte_change(uint8_t* p, uint8_t sentinel,
                             uint32_t timeout_ms) {
    const uint64_t deadline = monotonic_ms() + timeout_ms;
    while (monotonic_ms() < deadline) {
        try_msync(p, 1, MS_INVALIDATE);
        if (load8(p) != sentinel) {
            return true;
        }
        usleep(1000);
    }
    return false;
}

static void fill_zero_bias(uint8_t* bias_base);

static bool run_throughput(volatile uint32_t* regs, lml::exch_pool& pool,
                           uint32_t timeout_ms, uint32_t& seed,
                           uint32_t batches, uint32_t tasks_per_batch) {
    if (tasks_per_batch == 0 || batches == 0) {
        return true;
    }
    const uint32_t max_tasks = (kMaxInstructions - 2) / 2;
    if (tasks_per_batch > max_tasks) {
        fprintf(stderr, "tasks=%u exceeds max %u for 512 instructions\n",
                tasks_per_batch, max_tasks);
        return false;
    }

    const size_t total_tasks = static_cast<size_t>(batches) * tasks_per_batch;
    const size_t total_output_bytes = total_tasks * kOutputBytes;
    const size_t out_bytes = total_output_bytes + 8;

    lml::instruction_buffer ins(pool, kMaxInstructions);
    auto* bias = static_cast<uint8_t*>(pool.alloc(kBiasBytes, 64));
    auto* out = static_cast<uint8_t*>(pool.alloc(out_bytes, 64));
    if (!ins.valid() || !bias || !out) {
        fprintf(stderr, "throughput allocation failed\n");
        return false;
    }

    uint8_t expected_first[kOutputBytes];
    memset(bias, 0, kBiasBytes);
    fill_random_bias(bias, expected_first, seed);
    memset(out, kSentinel, out_bytes);

    printf("throughput BIAS+WRITE test:\n");
    printf("  batches=%u tasks_per_batch=%u total_tasks=%zu\n",
           batches, tasks_per_batch, total_tasks);
    printf("  bytes_per_task=%zu total_output_bytes=%zu\n",
           kOutputBytes, total_output_bytes);
    print_addr_range("bias", pool.phys(bias), kBiasBytes);
    print_addr_range("out", pool.phys(out), out_bytes);

    try_msync(pool.virt_base(), pool.bytes(), MS_SYNC);

    const uint64_t t0 = monotonic_ns();
    size_t global_task = 0;

    for (uint32_t b = 0; b < batches; ++b) {
        ins.clear();
        for (uint32_t t = 0; t < tasks_per_batch; ++t) {
            uint8_t* out_base = out + global_task * kOutputBytes;
            if (!ins.push(lml::instr::bias(pool.phys(bias), kBankStrideBytes,
                                           kBiasRowStrideBytes, kChannels,
                                           kOutH, kOutW * 4)) ||
                !ins.push(lml::instr::write_axi(pool.phys(out_base),
                                                kOutputChannelStrideBytes,
                                                kOutputRowStrideBytes,
                                                kChannels, kOutH, kOutW))) {
                fprintf(stderr, "instruction buffer overflow in throughput\n");
                return false;
            }
            ++global_task;
        }
        ins.push(lml::instr::mode(0x1));
        ins.push(lml::instr::mode(0x0));

        if (b == 0 || b + 1 == batches) {
            printf("  batch %u instruction_count=%zu out_start=0x%08x out_end=0x%08x\n",
                   b, ins.size(),
                   pool.phys(out + static_cast<size_t>(b) * tasks_per_batch * kOutputBytes),
                   pool.phys(out + static_cast<size_t>(b + 1) * tasks_per_batch * kOutputBytes) - 1u);
        }

        try_msync(ins.data(), ins.size() * sizeof(lml::instruction), MS_SYNC);
        write_regs_and_start(regs, ins);

        if (!wait_busy_done(regs, timeout_ms)) {
            fprintf(stderr, "throughput timeout at batch=%u reg0=0x%08x\n",
                    b, regs[kRegCtrl]);
            return false;
        }
    }

    const uint64_t t1 = monotonic_ns();
    try_msync(out, out_bytes, MS_INVALIDATE);

    if (!check_output_block(out, expected_first, 0)) {
        fprintf(stderr, "throughput first block verification failed\n");
        return false;
    }
    if (!check_output_block(out + (total_tasks - 1) * kOutputBytes,
                            expected_first,
                            static_cast<uint32_t>(total_tasks - 1))) {
        fprintf(stderr, "throughput last block verification failed\n");
        return false;
    }
    if (load8(out + total_output_bytes) != kSentinel ||
        load8(out + total_output_bytes + 1) != kSentinel) {
        fprintf(stderr, "throughput tail sentinel clobbered\n");
        return false;
    }

    const double seconds = static_cast<double>(t1 - t0) / 1000000000.0;
    const double mib = static_cast<double>(total_output_bytes) / (1024.0 * 1024.0);
    const double mb = static_cast<double>(total_output_bytes) / 1000000.0;
    printf("  elapsed=%.6f s\n", seconds);
    printf("  throughput=%.3f MiB/s %.3f MB/s\n",
           mib / seconds, mb / seconds);
    printf("  reg0=0x%08x\n", regs[kRegCtrl]);
    printf("  PASS\n");
    return true;
}

static bool run_read_benchmark(volatile uint32_t* regs, lml::exch_pool& pool,
                               uint32_t timeout_ms, uint32_t& seed,
                               uint32_t batches, uint32_t tasks_per_batch) {
    if (batches == 0 || tasks_per_batch == 0) return true;

    const uint32_t max_tasks = kMaxInstructions - 3u;
    if (tasks_per_batch > max_tasks) {
        fprintf(stderr, "read-bench tasks=%u exceeds max %u\n",
                tasks_per_batch, max_tasks);
        return false;
    }

    const size_t total_tasks = static_cast<size_t>(batches) * tasks_per_batch;
    const size_t total_input_span = total_tasks * kInputBytes;
    const size_t total_input_payload =
        total_tasks * static_cast<size_t>(kChannels) * kInH * kInW;

    lml::instruction_buffer ins(pool, kMaxInstructions);
    auto* input = static_cast<uint8_t*>(pool.alloc(total_input_span, 64));
    if (!ins.valid() || !input) {
        fprintf(stderr, "read-bench allocation failed\n");
        return false;
    }
    for (size_t i = 0; i < total_input_span; ++i) {
        input[i] = rand_u8(seed, 251);
    }

    printf("read-bench:\n");
    printf("  batches=%u tasks_per_batch=%u total_tasks=%zu tile=%ux%u C=%u\n",
           batches, tasks_per_batch, total_tasks, kInH, kInW, kChannels);
    printf("  payload_bytes_per_task=%zu address_span_per_task=%zu\n",
           static_cast<size_t>(kChannels) * kInH * kInW, kInputBytes);
    print_addr_range("input", pool.phys(input), total_input_span);

    try_msync(input, total_input_span, MS_SYNC);
    const uint64_t t0 = monotonic_ns();
    size_t task = 0;
    for (uint32_t b = 0; b < batches; ++b) {
        ins.clear();
        ins.push(lml::instr::config_regs(kBankStrideBytes,
                                         kInputRowStrideBytes,
                                         kChannels, kInH, kInW));
        for (uint32_t t = 0; t < tasks_per_batch; ++t) {
            uint8_t* input_base = input + task * kInputBytes;
            if (!ins.push(lml::instr::read_conv(true, pool.phys(input_base),
                                                false, 0xf, 0xf,
                                                kInH, kInW))) {
                fprintf(stderr, "instruction buffer overflow in read-bench\n");
                return false;
            }
            ++task;
        }
        ins.push(lml::instr::mode(0x1));
        ins.push(lml::instr::mode(0x0));

        try_msync(ins.data(), ins.size() * sizeof(lml::instruction), MS_SYNC);
        write_regs_and_start(regs, ins);
        if (!wait_busy_done(regs, timeout_ms)) {
            fprintf(stderr, "read-bench timeout batch=%u reg0=0x%08x\n",
                    b, regs[kRegCtrl]);
            return false;
        }
    }
    const uint64_t t1 = monotonic_ns();

    const double seconds = static_cast<double>(t1 - t0) / 1000000000.0;
    printf("  elapsed=%.6f s\n", seconds);
    printf("  payload_read=%.1f MB/s address_span=%.1f MB/s tasks=%.1f/s\n",
           static_cast<double>(total_input_payload) / seconds / 1000000.0,
           static_cast<double>(total_input_span) / seconds / 1000000.0,
           static_cast<double>(total_tasks) / seconds);
    printf("  reg0=0x%08x\n", regs[kRegCtrl]);
    printf("  PASS\n");
    return true;
}

static bool run_write_benchmark(volatile uint32_t* regs, lml::exch_pool& pool,
                                uint32_t timeout_ms,
                                uint32_t batches,
                                uint32_t tasks_per_batch) {
    if (batches == 0 || tasks_per_batch == 0) return true;

    const uint32_t max_tasks = (kMaxInstructions - 3u) / 2u;
    if (tasks_per_batch > max_tasks) {
        fprintf(stderr, "write-bench tasks=%u exceeds max %u\n",
                tasks_per_batch, max_tasks);
        return false;
    }

    const size_t total_tasks = static_cast<size_t>(batches) * tasks_per_batch;
    const size_t total_output_bytes = total_tasks * kOutputBytes;

    lml::instruction_buffer ins(pool, kMaxInstructions);
    auto* bias = static_cast<uint8_t*>(pool.alloc(kBiasBytes, 64));
    auto* out = static_cast<uint8_t*>(
        pool.alloc(total_output_bytes + 8, 64));
    if (!ins.valid() || !bias || !out) {
        fprintf(stderr, "write-bench allocation failed\n");
        return false;
    }
    fill_zero_bias(bias);
    memset(out, kSentinel, total_output_bytes + 8);

    printf("write-bench:\n");
    printf("  batches=%u tasks_per_batch=%u total_tasks=%zu tile_out=%ux%u C=%u\n",
           batches, tasks_per_batch, total_tasks, kOutH, kOutW, kChannels);
    printf("  output_bytes_per_task=%zu\n", kOutputBytes);
    print_addr_range("bias", pool.phys(bias), kBiasBytes);
    print_addr_range("out", pool.phys(out), total_output_bytes + 8);

    try_msync(bias, kBiasBytes, MS_SYNC);
    try_msync(out, total_output_bytes + 8, MS_SYNC);
    const uint64_t t0 = monotonic_ns();
    size_t task = 0;
    for (uint32_t b = 0; b < batches; ++b) {
        ins.clear();
        for (uint32_t t = 0; t < tasks_per_batch; ++t) {
            uint8_t* out_base = out + task * kOutputBytes;
            if (!ins.push(lml::instr::bias(pool.phys(bias),
                                           kBankStrideBytes,
                                           kBiasRowStrideBytes,
                                           kChannels, kOutH, kOutW * 4u)) ||
                !ins.push(lml::instr::write_axi(pool.phys(out_base),
                                                kOutputChannelStrideBytes,
                                                kOutputRowStrideBytes,
                                                kChannels, kOutH, kOutW))) {
                fprintf(stderr, "instruction buffer overflow in write-bench\n");
                return false;
            }
            ++task;
        }
        ins.push(lml::instr::mode(0x1));
        ins.push(lml::instr::mode(0x0));

        try_msync(ins.data(), ins.size() * sizeof(lml::instruction), MS_SYNC);
        write_regs_and_start(regs, ins);
        if (!wait_busy_done(regs, timeout_ms)) {
            fprintf(stderr, "write-bench timeout batch=%u reg0=0x%08x\n",
                    b, regs[kRegCtrl]);
            return false;
        }
    }
    const uint64_t t1 = monotonic_ns();

    try_msync(out, total_output_bytes + 8, MS_INVALIDATE);
    if (load8(out + total_output_bytes) != kSentinel ||
        load8(out + total_output_bytes + 1) != kSentinel) {
        fprintf(stderr, "write-bench tail sentinel clobbered\n");
        return false;
    }

    const double seconds = static_cast<double>(t1 - t0) / 1000000000.0;
    printf("  elapsed=%.6f s\n", seconds);
    printf("  write_payload=%.1f MB/s tasks=%.1f/s\n",
           static_cast<double>(total_output_bytes) / seconds / 1000000.0,
           static_cast<double>(total_tasks) / seconds);
    printf("  reg0=0x%08x\n", regs[kRegCtrl]);
    printf("  PASS\n");
    return true;
}

static void fill_random_input_tile(uint8_t* input_base, uint32_t& seed) {
    for (uint32_t c = 0; c < kChannels; ++c) {
        for (uint32_t y = 0; y < kInH; ++y) {
            for (uint32_t x = 0; x < kInW; ++x) {
                input_base[input_offset(c, y, x)] = rand_u8(seed, 8);
            }
        }
    }
}

static void fill_zero_bias(uint8_t* bias_base) {
    for (uint32_t c = 0; c < kChannels; ++c) {
        for (uint32_t y = 0; y < kOutH; ++y) {
            for (uint32_t x = 0; x < kOutW; ++x) {
                store32_le(bias_base + bias_ddr_offset(c, y, x), 0);
            }
        }
    }
}

static bool should_verify_task(size_t task, size_t total_tasks,
                               uint32_t tasks_per_batch, bool verify_all) {
    if (verify_all) {
        return true;
    }
    if (task == 0 || task + 1 == total_tasks) {
        return true;
    }
    if (tasks_per_batch != 0 && (task % tasks_per_batch) == 0) {
        return true;
    }
    if (tasks_per_batch != 0 && ((task + 1) % tasks_per_batch) == 0) {
        return true;
    }
    return false;
}

static uint8_t conv_grouped_expected_byte(const uint8_t* input,
                                          const int8_t kernel[9],
                                          size_t task,
                                          uint32_t groups_per_task,
                                          uint32_t co, uint32_t y,
                                          uint32_t x) {
    int32_t acc = 0;
    for (uint32_t g = 0; g < groups_per_task; ++g) {
        const uint8_t* input_base =
            input + (task * groups_per_task + g) * kInputBytes;
        acc += conv3x3_expected(input_base, kernel, co, y, x);
    }
    return quant_from_acc(acc);
}

static uint64_t verify_conv_throughput_outputs(const uint8_t* input,
                                               const uint8_t* out,
                                               const int8_t kernel[9],
                                               size_t total_tasks,
                                               uint32_t tasks_per_batch,
                                               uint32_t groups_per_task,
                                               bool verify_all,
                                               bool* ok) {
    uint64_t checked_bytes = 0;
    *ok = true;
    for (size_t task = 0; task < total_tasks; ++task) {
        if (!should_verify_task(task, total_tasks, tasks_per_batch, verify_all)) {
            continue;
        }
        for (uint32_t co = 0; co < kChannels; ++co) {
            for (uint32_t y = 0; y < kOutH; ++y) {
                for (uint32_t x = 0; x < kOutW; ++x) {
                    const uint32_t off = output_offset(co, y, x);
                    const uint8_t exp = conv_grouped_expected_byte(
                        input, kernel, task, groups_per_task, co, y, x);
                    const uint8_t got = load8(out + task * kOutputBytes + off);
                    ++checked_bytes;
                    if (got != exp) {
                        fprintf(stderr,
                                "conv-throughput mismatch task=%zu co=%u y=%u x=%u off=%u: got=0x%02x exp=0x%02x\n",
                                task, co, y, x, off, got, exp);
                        *ok = false;
                        return checked_bytes;
                    }
                }
            }
        }
    }
    return checked_bytes;
}

static void arm_conv_reference_benchmark(const uint8_t* input,
                                         uint8_t* arm_out,
                                         const int8_t kernel[9],
                                         size_t total_tasks,
                                         uint32_t tasks_per_batch,
                                         uint32_t groups_per_task,
                                         bool verify_all,
                                         uint64_t* elapsed_ns,
                                         uint32_t* checked_tasks) {
    const uint64_t t0 = monotonic_ns();
    uint32_t tasks = 0;
    for (size_t task = 0; task < total_tasks; ++task) {
        if (!should_verify_task(task, total_tasks, tasks_per_batch,
                                verify_all)) {
            continue;
        }
        uint8_t* out_base = arm_out + task * kOutputBytes;
        for (uint32_t co = 0; co < kChannels; ++co) {
            for (uint32_t y = 0; y < kOutH; ++y) {
                for (uint32_t x = 0; x < kOutW; ++x) {
                    out_base[output_offset(co, y, x)] =
                        conv_grouped_expected_byte(input, kernel, task,
                                                   groups_per_task, co, y, x);
                }
            }
        }
        ++tasks;
    }
    *elapsed_ns = monotonic_ns() - t0;
    if (checked_tasks) {
        *checked_tasks = tasks;
    }
}

struct conv_throughput_workspace {
    lml::instruction_buffer ins;
    uint8_t* input = nullptr;
    uint8_t* bias = nullptr;
    uint8_t* out = nullptr;
    uint8_t* arm_out = nullptr;
};

struct conv_throughput_plan {
    uint32_t tasks_per_batch = 0;
    size_t total_tasks = 0;
    size_t input_bytes = 0;
    size_t bias_bytes = 0;
    size_t output_bytes = 0;
};

static uint32_t conv_throughput_instructions_per_task(uint32_t groups_per_task) {
    if (groups_per_task == 0) return 0;
    if (groups_per_task == 1) {
        return 5u;  // bias, read, conv, write, bias-reset
    }
    return 2u * groups_per_task + 4u;
}

static bool make_conv_throughput_plan(uint32_t batches, uint32_t groups_per_task,
                                      conv_throughput_plan* plan) {
    if (!plan || batches == 0 || groups_per_task == 0) {
        return false;
    }

    const uint32_t instructions_per_task =
        conv_throughput_instructions_per_task(groups_per_task);
    const uint32_t fixed_instructions = kChannels * kChannels + 1u + 2u;
    if (fixed_instructions + instructions_per_task > kMaxInstructions) {
        return false;
    }

    plan->tasks_per_batch =
        (kMaxInstructions - fixed_instructions) / instructions_per_task;
    if (plan->tasks_per_batch == 0) {
        return false;
    }

    plan->total_tasks = static_cast<size_t>(batches) * plan->tasks_per_batch;
    plan->input_bytes = plan->total_tasks * groups_per_task * kInputBytes;
    plan->bias_bytes = plan->total_tasks * kBiasBytes;
    plan->output_bytes = plan->total_tasks * kOutputBytes;
    return true;
}

static bool run_conv_throughput_core(volatile uint32_t* regs,
                                     lml::exch_pool& pool,
                                     uint32_t timeout_ms, uint32_t& seed,
                                     uint32_t batches,
                                     uint32_t groups_per_task,
                                     bool verify_all,
                                     conv_throughput_workspace& ws) {
    if (batches == 0 || groups_per_task == 0) {
        return true;
    }

    conv_throughput_plan plan;
    if (!make_conv_throughput_plan(batches, groups_per_task, &plan)) {
        fprintf(stderr, "no conv throughput plan fits groups=%u batches=%u\n",
                groups_per_task, batches);
        return false;
    }

    if (!ws.ins.valid() || !ws.input || !ws.bias || !ws.out || !ws.arm_out) {
        fprintf(stderr, "conv-throughput workspace invalid\n");
        return false;
    }

    int8_t kernel[9];
    for (uint32_t i = 0; i < 9; ++i) {
        kernel[i] = rand_i8_small(seed);
    }
    for (size_t task = 0; task < plan.total_tasks; ++task) {
        fill_zero_bias(ws.bias + task * kBiasBytes);
        for (uint32_t g = 0; g < groups_per_task; ++g) {
            fill_random_input_tile(ws.input + (task * groups_per_task + g) * kInputBytes,
                                   seed);
        }
    }
    memset(ws.out, kSentinel, plan.output_bytes + 8);

    printf("conv-throughput tiled test:\n");
    printf("  batches=%u tasks_per_batch=%u total_tasks=%zu groups_per_task=%u\n",
           batches, plan.tasks_per_batch, plan.total_tasks, groups_per_task);
    printf("  read_conv_overlap=%s\n",
           groups_per_task > 1u ? "on" : "off");
    printf("  one task: 4out x %uin x %ux%u x 3x3 MACs\n",
           groups_per_task * kChannels, kOutH, kOutW);
    printf("  instruction_count_per_batch=%u max=%u\n",
           1u + kChannels * kChannels +
               plan.tasks_per_batch *
                   conv_throughput_instructions_per_task(groups_per_task) +
               2u,
           kMaxInstructions);
    printf("  verify=%s\n", verify_all ? "all outputs" : "sampled first/last/per-batch outputs");
    print_addr_range("input", pool.phys(ws.input), plan.input_bytes);
    print_addr_range("bias", pool.phys(ws.bias), plan.bias_bytes);
    print_addr_range("out", pool.phys(ws.out), plan.output_bytes + 8);
    printf("  kernel:");
    for (uint32_t i = 0; i < 9; ++i) {
        printf(" %d", static_cast<int>(kernel[i]));
    }
    printf("\n");

    try_msync(pool.virt_base(), pool.bytes(), MS_SYNC);

    size_t global_task = 0;
    const uint64_t t0 = monotonic_ns();
    for (uint32_t b = 0; b < batches; ++b) {
        ws.ins.clear();
        push_common_regs(ws.ins, kernel);

        const size_t batch_first_task = global_task;
        for (uint32_t t = 0; t < plan.tasks_per_batch; ++t) {
            uint8_t* bias_base = ws.bias + global_task * kBiasBytes;
            uint8_t* out_base = ws.out + global_task * kOutputBytes;

            ws.ins.push(lml::instr::bias(pool.phys(bias_base), kBankStrideBytes,
                                      kBiasRowStrideBytes, kChannels, kOutH,
                                      kOutW * 4));
            if (groups_per_task == 1u) {
                uint8_t* input_base =
                    ws.input + global_task * groups_per_task * kInputBytes;
                ws.ins.push(lml::instr::read_conv(true, pool.phys(input_base),
                                               false, 0xf, 0xf, kInH, kInW));
                ws.ins.push(lml::instr::read_conv(false, pool.phys(input_base),
                                               true, 0xf, 0xf, kInH, kInW));
            } else {
                uint8_t* first_input_base =
                    ws.input + global_task * groups_per_task * kInputBytes;
                ws.ins.push(lml::instr::read_conv(true, pool.phys(first_input_base),
                                               false, 0xf, 0xf, kInH, kInW));
                for (uint32_t g = 1; g < groups_per_task; ++g) {
                    uint8_t* next_input_base =
                        ws.input + (global_task * groups_per_task + g) * kInputBytes;
                    ws.ins.push(lml::instr::read_conv(
                        false, pool.phys(next_input_base), true, 0xf, 0xf,
                        kInH, kInW));
                    ws.ins.push(lml::instr::read_conv(
                        true, pool.phys(next_input_base), false, 0xf, 0xf,
                        kInH, kInW));
                }
                uint8_t* last_input_base =
                    ws.input + (global_task * groups_per_task +
                             (groups_per_task - 1u)) * kInputBytes;
                ws.ins.push(lml::instr::read_conv(false, pool.phys(last_input_base),
                                               true, 0xf, 0xf, kInH, kInW));
                ws.ins.push(lml::instr::mode(0x1));
            }
            ws.ins.push(lml::instr::write_axi(pool.phys(out_base),
                                           kOutputChannelStrideBytes,
                                           kOutputRowStrideBytes, kChannels,
                                           kOutH, kOutW));
            ws.ins.push(lml::instr::bias(pool.phys(bias_base), kBankStrideBytes,
                                      kBiasRowStrideBytes, kChannels, kOutH,
                                      kOutW * 4));
            ++global_task;
        }
        ws.ins.push(lml::instr::mode(0x1));
        ws.ins.push(lml::instr::mode(0x0));

        if (b == 0 || b + 1 == batches) {
            printf("  batch %u instruction_count=%zu task_range=[%zu,%zu] out_start=0x%08x out_end=0x%08x\n",
                   b, ws.ins.size(), batch_first_task, global_task - 1,
                   pool.phys(ws.out + batch_first_task * kOutputBytes),
                   pool.phys(ws.out + global_task * kOutputBytes) - 1u);
        }

        try_msync(ws.ins.data(), ws.ins.size() * sizeof(lml::instruction), MS_SYNC);
        write_regs_and_start(regs, ws.ins);

        if (!wait_busy_done(regs, timeout_ms)) {
            fprintf(stderr, "conv-throughput timeout at batch=%u reg0=0x%08x\n",
                    b, regs[kRegCtrl]);
            return false;
        }
    }
    const uint64_t t1 = monotonic_ns();

    try_msync(ws.out, plan.output_bytes + 8, MS_INVALIDATE);
    if (load8(ws.out + plan.output_bytes) != kSentinel ||
        load8(ws.out + plan.output_bytes + 1) != kSentinel) {
        fprintf(stderr, "conv-throughput tail sentinel clobbered\n");
        return false;
    }

    const uint64_t macs_per_task =
        static_cast<uint64_t>(kChannels) * kChannels * groups_per_task *
        kOutH * kOutW * 9u;
    const uint64_t total_macs = macs_per_task * plan.total_tasks;
    const double seconds = static_cast<double>(t1 - t0) / 1000000000.0;
    const double gmacs = static_cast<double>(total_macs) / 1000000000.0;
    const double gops = gmacs * 2.0;
    const uint64_t input_bytes_per_task =
        static_cast<uint64_t>(groups_per_task) * kChannels * kInH * kInW;
    const uint64_t input_span_bytes_per_task =
        static_cast<uint64_t>(groups_per_task) * kInputBytes;
    const uint64_t bias_bytes_per_task =
        static_cast<uint64_t>(kChannels) * kOutH * kBiasRowStrideBytes;
    const uint64_t out_bytes_per_task = kOutputBytes;
    const uint64_t total_bytes_per_task =
        input_bytes_per_task + bias_bytes_per_task + out_bytes_per_task;
    const uint64_t total_span_bytes_per_task =
        input_span_bytes_per_task + kBiasBytes + out_bytes_per_task;
    const uint64_t total_input_bytes =
        input_bytes_per_task * plan.total_tasks;
    const uint64_t total_axi_payload_bytes =
        total_bytes_per_task * plan.total_tasks;
    const uint64_t total_span_bytes =
        total_span_bytes_per_task * plan.total_tasks;
    const double tasks_per_s =
        static_cast<double>(plan.total_tasks) / seconds;
    const double us_per_task = seconds * 1000000.0 /
                               static_cast<double>(plan.total_tasks);
    const double cycles_per_task_100m = us_per_task * 100.0;
    const double cycles_per_out_pixel_100m =
        cycles_per_task_100m /
        static_cast<double>(kChannels * kOutH * kOutW);
    const double effective_total_mb_s =
        static_cast<double>(total_axi_payload_bytes) / seconds / 1000000.0;
    const double effective_input_mb_s =
        static_cast<double>(total_input_bytes) / seconds / 1000000.0;
    const double effective_span_mb_s =
        static_cast<double>(total_span_bytes) / seconds / 1000000.0;
    const double operational_intensity =
        static_cast<double>(macs_per_task) /
        static_cast<double>(total_bytes_per_task);

    bool verify_ok = false;
    const uint64_t checked_bytes = verify_conv_throughput_outputs(
        ws.input, ws.out, kernel, plan.total_tasks, plan.tasks_per_batch, groups_per_task,
        verify_all, &verify_ok);
    if (!verify_ok) {
        return false;
    }

    uint64_t arm_elapsed_ns = 0;
    uint32_t arm_checked_tasks = 0;
    arm_conv_reference_benchmark(ws.input, ws.arm_out, kernel, plan.total_tasks,
                                 plan.tasks_per_batch, groups_per_task,
                                 verify_all, &arm_elapsed_ns,
                                 &arm_checked_tasks);
    bool arm_match = true;
    for (size_t task = 0; task < plan.total_tasks && arm_match; ++task) {
        if (!should_verify_task(task, plan.total_tasks, plan.tasks_per_batch,
                                verify_all)) {
            continue;
        }
        const uint8_t* hw_base = ws.out + task * kOutputBytes;
        const uint8_t* arm_base = ws.arm_out + task * kOutputBytes;
        for (uint32_t i = 0; i < kOutputBytes; ++i) {
            if (load8(hw_base + i) != arm_base[i]) {
                fprintf(stderr,
                        "ARM reference mismatch task=%zu byte=%u: hw=0x%02x arm=0x%02x\n",
                        task, i, load8(hw_base + i), arm_base[i]);
                arm_match = false;
                break;
            }
        }
    }
    if (!arm_match) {
        return false;
    }

    const double arm_seconds = static_cast<double>(arm_elapsed_ns) / 1000000000.0;
    const double hw_gmac_s = gmacs / seconds;
    const double mac_per_cycle_100m = hw_gmac_s * 10.0;
    const double util_16mac_cycle = mac_per_cycle_100m / 16.0 * 100.0;
    const uint64_t arm_macs =
        macs_per_task * static_cast<uint64_t>(arm_checked_tasks);
    const double arm_gmac_s =
        (arm_seconds > 0.0) ?
        (static_cast<double>(arm_macs) / 1000000000.0 / arm_seconds) : 0.0;
    const double speedup = arm_seconds / seconds;

    printf("  macs_per_task=%" PRIu64 " total_macs=%" PRIu64 "\n",
           macs_per_task, total_macs);
    printf("  bytes_per_task payload input=%" PRIu64 " bias=%" PRIu64
           " out=%" PRIu64 " total=%" PRIu64 "\n",
           input_bytes_per_task, bias_bytes_per_task, out_bytes_per_task,
           total_bytes_per_task);
    printf("  bytes_per_task address_span input=%" PRIu64
           " bias=%zu out=%" PRIu64 " total=%" PRIu64 "\n",
           input_span_bytes_per_task, kBiasBytes, out_bytes_per_task,
           total_span_bytes_per_task);
    printf("  effective_axi_payload=%.1f MB/s input_only=%.1f MB/s\n",
           effective_total_mb_s, effective_input_mb_s);
    printf("  effective_address_span=%.1f MB/s\n", effective_span_mb_s);
    printf("  task_rate=%.1f task/s us_per_task=%.3f cycles_per_task@100MHz=%.0f cycles_per_out_pixel@100MHz=%.2f\n",
           tasks_per_s, us_per_task, cycles_per_task_100m,
           cycles_per_out_pixel_100m);
    printf("  compute_cycles mac_per_cycle@100MHz=%.2f util_vs_16mac/cycle=%.1f%%\n",
           mac_per_cycle_100m, util_16mac_cycle);
    printf("  operational_intensity=%.2f MAC/byte\n",
           operational_intensity);
    printf("  verified_output_bytes=%" PRIu64 "\n", checked_bytes);
    printf("  hw_elapsed=%.6f s\n", seconds);
    printf("  hw_compute=%.3f GMAC/s %.3f GOPS\n",
           hw_gmac_s, gops / seconds);
    printf("  arm_reference=%s checked_tasks=%u elapsed=%.6f s\n",
           verify_all ? "all" : "sampled",
           arm_checked_tasks, arm_seconds);
    printf("  arm_compute=%.3f GMAC/s %.3f GOPS\n",
           arm_gmac_s, arm_gmac_s * 2.0);
    printf("  speedup_vs_arm=%.2fx\n", speedup);
    printf("  reg0=0x%08x\n", regs[kRegCtrl]);
    printf("  PASS\n");
    return true;
}

static bool run_conv_throughput(volatile uint32_t* regs, lml::exch_pool& pool,
                                uint32_t timeout_ms, uint32_t& seed,
                                uint32_t batches, uint32_t groups_per_task,
                                bool verify_all) {
    conv_throughput_plan plan;
    if (!make_conv_throughput_plan(batches, groups_per_task, &plan)) {
        fprintf(stderr, "conv-throughput allocation failed\n");
        return false;
    }
    conv_throughput_workspace ws{
        lml::instruction_buffer(pool, kMaxInstructions),
        static_cast<uint8_t*>(pool.alloc(plan.input_bytes, 64)),
        static_cast<uint8_t*>(pool.alloc(plan.bias_bytes, 64)),
        static_cast<uint8_t*>(pool.alloc(plan.output_bytes + 8, 64)),
        static_cast<uint8_t*>(malloc(plan.output_bytes))};
    if (!ws.ins.valid() || !ws.input || !ws.bias || !ws.out || !ws.arm_out) {
        fprintf(stderr, "conv-throughput allocation failed\n");
        free(ws.arm_out);
        return false;
    }
    const bool ok = run_conv_throughput_core(regs, pool, timeout_ms, seed,
                                              batches, groups_per_task,
                                              verify_all, ws);
    free(ws.arm_out);
    return ok;
}

static bool run_conv_throughput_sweep(volatile uint32_t* regs,
                                      lml::exch_pool& pool,
                                      uint32_t timeout_ms, uint32_t& seed,
                                      uint32_t batches, bool verify_all) {
    const uint32_t candidates[] = {1u, 2u, 4u, 8u, 12u, 16u};
    bool ok = true;
    uint32_t max_groups = 0;
    uint32_t max_total_tasks = 0;
    uint32_t max_input_groups = 0;
    uint32_t max_bias_tasks = 0;
    uint32_t max_output_tasks = 0;
    for (uint32_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        const uint32_t groups = candidates[i];
        const uint32_t instructions_per_task =
            conv_throughput_instructions_per_task(groups);
        const uint32_t fixed_instructions = kChannels * kChannels + 1u + 2u;
        const uint32_t tasks_per_batch =
            (kMaxInstructions - fixed_instructions) / instructions_per_task;
        const uint32_t total_tasks = batches * tasks_per_batch;
        if (groups > max_groups) max_groups = groups;
        if (total_tasks > max_total_tasks) max_total_tasks = total_tasks;
        if (total_tasks * groups > max_input_groups) max_input_groups = total_tasks * groups;
        if (total_tasks > max_bias_tasks) max_bias_tasks = total_tasks;
        if (total_tasks > max_output_tasks) max_output_tasks = total_tasks;
    }
    conv_throughput_workspace ws{
        lml::instruction_buffer(pool, kMaxInstructions),
        static_cast<uint8_t*>(pool.alloc(static_cast<size_t>(max_input_groups) * kInputBytes, 64)),
        static_cast<uint8_t*>(pool.alloc(static_cast<size_t>(max_bias_tasks) * kBiasBytes, 64)),
        static_cast<uint8_t*>(pool.alloc(static_cast<size_t>(max_output_tasks) * kOutputBytes + 8, 64)),
        static_cast<uint8_t*>(malloc(static_cast<size_t>(max_output_tasks) * kOutputBytes))};
    if (!ws.ins.valid() || !ws.input || !ws.bias || !ws.out || !ws.arm_out) {
        fprintf(stderr, "conv-throughput sweep allocation failed\n");
        free(ws.arm_out);
        return false;
    }
    printf("conv-throughput sweep:\n");
    for (uint32_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        uint32_t groups = candidates[i];
        printf("  case groups=%u\n", groups);
        uint32_t case_seed = seed;
        ok = run_conv_throughput_core(regs, pool, timeout_ms, case_seed,
                                      batches, groups, verify_all, ws) && ok;
    }
    if (ok) {
        printf("conv-throughput sweep PASS\n");
    }
    free(ws.arm_out);
    return ok;
}

struct acc_device {
    lml::accelerator acc;
    lml::exch_pool* pool = nullptr;

    acc_device(volatile uint32_t* regs, lml::exch_pool* p, uint32_t timeout_ms)
        : acc(regs, p ? p->virt_base() : nullptr,
              p ? p->phys_base() : 0, timeout_ms),
          pool(p) {}

    uint32_t phys(const void* ptr) const {
        return acc.phys(ptr);
    }

    bool run(lml::instruction_buffer& ins, uint8_t* wait_ptr,
             uint8_t sentinel) {
        try_msync(ins.data(), ins.size() * sizeof(lml::instruction), MS_SYNC);
        acc.start(ins);
        if (!acc.wait_done()) {
            fprintf(stderr, "accelerator timeout reg0=0x%08x\n", acc.ctrl());
            return false;
        }
        (void)wait_ptr;
        (void)sentinel;
        return true;
    }
};

struct yolo_layer_desc {
    const char* name;
    uint32_t in_h;
    uint32_t in_w;
    uint32_t cin;
    uint32_t cout;
};

struct yolo_tile_job {
    uint32_t oc;
    uint32_t oy;
    uint32_t ox;
    uint32_t valid_oh;
    uint32_t valid_ow;
};

struct yolo_batch_profile {
    uint64_t pack_ns = 0;
    uint64_t build_ns = 0;
    uint64_t sync_ns = 0;
    uint64_t fpga_wait_ns = 0;
    uint64_t invalidate_ns = 0;
    uint64_t copy_ns = 0;
    uint64_t batches = 0;
    uint64_t tiles = 0;
    uint64_t instructions = 0;
};

static uint8_t random_feature(uint32_t& seed) {
    return static_cast<uint8_t>(static_cast<int8_t>(
        static_cast<int>(rand_u8(seed, 17)) - 8));
}

static void fill_random_feature(std::vector<uint8_t>& data, uint32_t& seed) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = random_feature(seed);
    }
}

static void fill_random_weights(std::vector<int8_t>& weights, uint32_t& seed) {
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = rand_i8_small(seed);
    }
}

static bool run_acc_conv_tile(acc_device& dev, const uint8_t* input,
                              uint8_t* output, const int8_t* weights,
                              uint32_t in_h, uint32_t in_w,
                              uint32_t cin, uint32_t cout,
                              uint32_t oc_base, uint32_t oy0,
                              uint32_t ox0, uint32_t valid_oh,
                              uint32_t valid_ow, lml::instruction_buffer& ins,
                              uint8_t* tile_inputs, uint8_t* tile_bias,
                              uint8_t* tile_out) {
    memset(tile_bias, 0, kBiasBytes);
    memset(tile_out, kSentinel, kOutputBytes);
    fill_zero_bias(tile_bias);

    ins.clear();
    ins.push(lml::instr::config_regs(kBankStrideBytes, kInputRowStrideBytes,
                                     kChannels, kInH, kInW));
    ins.push(lml::instr::bias(dev.phys(tile_bias), kBankStrideBytes,
                              kBiasRowStrideBytes, kChannels, kOutH,
                              kOutW * 4));

    for (uint32_t ic_base = 0; ic_base < cin; ic_base += kChannels) {
        uint8_t* tile_input = tile_inputs + (ic_base / kChannels) * kInputBytes;
        memset(tile_input, 0, kInputBytes);
        for (uint32_t ci = 0; ci < kChannels; ++ci) {
            for (uint32_t y = 0; y < kInH; ++y) {
                for (uint32_t x = 0; x < kInW; ++x) {
                    const uint32_t iy = oy0 + y;
                    const uint32_t ix = ox0 + x;
                    if (iy < in_h && ix < in_w) {
                        tile_input[input_offset(ci, y, x)] =
                            input[lml::yolo::nchw_offset(ic_base + ci, iy,
                                                         ix, in_h, in_w)];
                    }
                }
            }
        }
        push_kernel_regs(ins, weights, cin, oc_base, ic_base);
        ins.push(lml::instr::read_conv(true, dev.phys(tile_input),
                                       false, 0xf, 0xf, kInH, kInW));
        ins.push(lml::instr::read_conv(false, dev.phys(tile_input),
                                       true, 0xf, 0xf, kInH, kInW));
        if (ic_base + kChannels < cin) {
            ins.push(lml::instr::mode(0x1));
        }
    }

    ins.push(lml::instr::write_axi(dev.phys(tile_out),
                                   kOutputChannelStrideBytes,
                                   kOutputRowStrideBytes, kChannels,
                                   kOutH, kOutW));
    ins.push(lml::instr::bias(dev.phys(tile_bias), kBankStrideBytes,
                              kBiasRowStrideBytes, kChannels, kOutH,
                              kOutW * 4));
    ins.push(lml::instr::mode(0x1));
    ins.push(lml::instr::mode(0x0));

    try_msync(tile_inputs, static_cast<size_t>(cin / kChannels) * kInputBytes,
              MS_SYNC);
    try_msync(tile_bias, kBiasBytes, MS_SYNC);
    try_msync(tile_out, kOutputBytes, MS_SYNC);
    if (!dev.run(ins, tile_out, kSentinel)) {
        return false;
    }
    try_msync(tile_out, kOutputBytes, MS_INVALIDATE);

    const uint32_t out_h = in_h - 2;
    const uint32_t out_w = in_w - 2;
    for (uint32_t co = 0; co < kChannels; ++co) {
        for (uint32_t y = 0; y < valid_oh; ++y) {
            for (uint32_t x = 0; x < valid_ow; ++x) {
                output[lml::yolo::nchw_offset(oc_base + co, oy0 + y,
                                               ox0 + x, out_h, out_w)] =
                    load8(tile_out + output_offset(co, y, x));
            }
        }
    }
    (void)cout;
    return true;
}

static void pack_tile_input(const uint8_t* input, uint8_t* tile_input,
                            uint32_t in_h, uint32_t in_w, uint32_t ic_base,
                            uint32_t oy0, uint32_t ox0) {
    memset(tile_input, 0, kInputBytes);
    for (uint32_t ci = 0; ci < kChannels; ++ci) {
        for (uint32_t y = 0; y < kInH; ++y) {
            const uint32_t iy = oy0 + y;
            if (iy >= in_h || ox0 >= in_w) {
                continue;
            }
            const uint32_t copy_w = std::min(kInW, in_w - ox0);
            memcpy(tile_input + input_offset(ci, y, 0),
                   input + lml::yolo::nchw_offset(ic_base + ci, iy, ox0,
                                                  in_h, in_w),
                   copy_w);
        }
    }
}

static bool run_acc_conv_tile_batch(acc_device& dev, const uint8_t* input,
                                    uint8_t* output, const int8_t* weights,
                                    uint32_t in_h, uint32_t in_w,
                                    uint32_t cin, uint32_t cout,
                                    uint32_t oc_base,
                                    const std::vector<yolo_tile_job>& jobs,
                                    size_t job_begin, size_t job_count,
                                    lml::instruction_buffer& ins,
                                    uint8_t* batch_inputs,
                                    uint8_t* batch_bias,
                                    uint8_t* batch_out,
                                    yolo_batch_profile* profile,
                                    bool overlap_read_conv) {
    if (job_count == 0) return true;

    const uint32_t groups = cin / kChannels;
    const uint64_t pack0 = monotonic_ns();
    memset(batch_bias, 0, kBiasBytes);
    fill_zero_bias(batch_bias);
    memset(batch_out, kSentinel, job_count * kOutputBytes);

    for (size_t j = 0; j < job_count; ++j) {
        const yolo_tile_job& job = jobs[job_begin + j];
        for (uint32_t g = 0; g < groups; ++g) {
            uint8_t* tile_input =
                batch_inputs + (j * groups + g) * kInputBytes;
            pack_tile_input(input, tile_input, in_h, in_w, g * kChannels,
                            job.oy, job.ox);
        }
    }
    const uint64_t pack1 = monotonic_ns();

    const uint64_t build0 = monotonic_ns();
    ins.clear();
    auto push_checked = [&](lml::instruction x) -> bool {
        if (ins.push(x)) return true;
        fprintf(stderr,
                "instruction buffer overflow: size=%zu capacity=%zu job_count=%zu cin=%u oc=%u\n",
                ins.size(), ins.capacity(), job_count, cin, oc_base);
        return false;
    };
    if (!push_checked(lml::instr::config_regs(kBankStrideBytes,
                                              kInputRowStrideBytes,
                                              kChannels, kInH, kInW))) {
        return false;
    }

    uint32_t last_ic_base = UINT32_MAX;
    for (size_t j = 0; j < job_count; ++j) {
        if (!push_checked(lml::instr::bias(dev.phys(batch_bias),
                                           kBankStrideBytes,
                                           kBiasRowStrideBytes, kChannels,
                                           kOutH, kOutW * 4))) {
            return false;
        }
        auto push_kernel_group = [&](uint32_t g) -> bool {
            const uint32_t ic_base = g * kChannels;
            if (ic_base == last_ic_base) return true;
            for (uint32_t co = 0; co < kChannels; ++co) {
                for (uint32_t ci = 0; ci < kChannels; ++ci) {
                    const int8_t* k =
                        weights + ((oc_base + co) * cin + ic_base + ci) * 9;
                    if (!push_checked(lml::instr::regs(
                            static_cast<uint8_t>(co * kChannels + ci),
                            pack_kernel_lo(k), pack_kernel_hi(k)))) {
                        return false;
                    }
                }
            }
            last_ic_base = ic_base;
            return true;
        };

        if (!overlap_read_conv) {
            for (uint32_t g = 0; g < groups; ++g) {
                if (!push_kernel_group(g)) return false;
                uint8_t* tile_input =
                    batch_inputs + (j * groups + g) * kInputBytes;
                if (!push_checked(lml::instr::read_conv(true, dev.phys(tile_input),
                                                        false, 0xf, 0xf,
                                                        kInH, kInW)) ||
                    !push_checked(lml::instr::read_conv(false,
                                                        dev.phys(tile_input),
                                                        true, 0xf, 0xf,
                                                        kInH, kInW))) {
                    return false;
                }
                if (g + 1u < groups) {
                    if (!push_checked(lml::instr::mode(0x1))) {
                        return false;
                    }
                }
            }
        } else {
            if (groups == 0) {
                return false;
            }
            if (!push_kernel_group(0)) return false;
            uint8_t* first_input =
                batch_inputs + (j * groups + 0u) * kInputBytes;
            if (!push_checked(lml::instr::read_conv(true, dev.phys(first_input),
                                                    false, 0xf, 0xf,
                                                    kInH, kInW))) {
                return false;
            }
            for (uint32_t g = 1; g < groups; ++g) {
                uint8_t* tile_input =
                    batch_inputs + (j * groups + g) * kInputBytes;
                if (!push_checked(lml::instr::read_conv(false,
                                                        dev.phys(tile_input),
                                                        true, 0xf, 0xf,
                                                        kInH, kInW)) ||
                    !push_checked(lml::instr::read_conv(true,
                                                        dev.phys(tile_input),
                                                        false, 0xf, 0xf,
                                                        kInH, kInW))) {
                    return false;
                }
                if (!push_kernel_group(g)) return false;
            }
            uint8_t* last_tile_input =
                batch_inputs + (j * groups + (groups - 1u)) * kInputBytes;
            if (!push_checked(lml::instr::read_conv(false,
                                                    dev.phys(last_tile_input),
                                                    true, 0xf, 0xf,
                                                    kInH, kInW))) {
                return false;
            }
            if (groups > 1u) {
                if (!push_checked(lml::instr::mode(0x1))) {
                    return false;
                }
            }
        }
        uint8_t* tile_out = batch_out + j * kOutputBytes;
        if (!push_checked(lml::instr::write_axi(dev.phys(tile_out),
                                                kOutputChannelStrideBytes,
                                                kOutputRowStrideBytes,
                                                kChannels, kOutH, kOutW)) ||
            !push_checked(lml::instr::bias(dev.phys(batch_bias),
                                           kBankStrideBytes,
                                           kBiasRowStrideBytes, kChannels,
                                           kOutH, kOutW * 4))) {
            return false;
        }
    }
    if (!push_checked(lml::instr::mode(0x1)) ||
        !push_checked(lml::instr::mode(0x0))) {
        return false;
    }
    const uint64_t build1 = monotonic_ns();

    const uint64_t sync0 = monotonic_ns();
    try_msync(batch_inputs, job_count * groups * kInputBytes, MS_SYNC);
    try_msync(batch_bias, kBiasBytes, MS_SYNC);
    try_msync(batch_out, job_count * kOutputBytes, MS_SYNC);
    try_msync(ins.data(), ins.size() * sizeof(lml::instruction), MS_SYNC);
    const uint64_t sync1 = monotonic_ns();

    const uint64_t fpga0 = monotonic_ns();
    dev.acc.start(ins);
    if (!dev.acc.wait_done()) {
        fprintf(stderr, "accelerator timeout reg0=0x%08x\n",
                dev.acc.ctrl());
        return false;
    }
    const uint64_t fpga1 = monotonic_ns();

    const uint64_t inv0 = monotonic_ns();
    try_msync(batch_out, job_count * kOutputBytes, MS_INVALIDATE);
    const uint64_t inv1 = monotonic_ns();

    const uint64_t copy0 = monotonic_ns();
    const uint32_t out_h = in_h - 2;
    const uint32_t out_w = in_w - 2;
    for (size_t j = 0; j < job_count; ++j) {
        const yolo_tile_job& job = jobs[job_begin + j];
        const uint8_t* tile_out = batch_out + j * kOutputBytes;
        for (uint32_t co = 0; co < kChannels; ++co) {
            for (uint32_t y = 0; y < job.valid_oh; ++y) {
                memcpy(output + lml::yolo::nchw_offset(oc_base + co,
                                                       job.oy + y, job.ox,
                                                       out_h, out_w),
                       tile_out + output_offset(co, y, 0),
                       job.valid_ow);
            }
        }
    }
    const uint64_t copy1 = monotonic_ns();

    if (profile) {
        profile->pack_ns += pack1 - pack0;
        profile->build_ns += build1 - build0;
        profile->sync_ns += sync1 - sync0;
        profile->fpga_wait_ns += fpga1 - fpga0;
        profile->invalidate_ns += inv1 - inv0;
        profile->copy_ns += copy1 - copy0;
        profile->batches += 1;
        profile->tiles += job_count;
        profile->instructions += ins.size();
    }

    (void)cout;
    return true;
}

static bool run_yolo_cpp_check(uint32_t& seed) {
    const yolo_layer_desc layers[] = {
        {"p3", 160, 160, 4, 16},
        {"p4", 80, 80, 16, 32},
        {"p5", 40, 40, 16, 64},
    };

    std::vector<uint8_t> current(static_cast<size_t>(layers[0].cin) *
                                 layers[0].in_h * layers[0].in_w);
    fill_random_feature(current, seed);

    printf("yolo-like C++ tiled check:\n");
    printf("  layers=3, 3x3 stride1 valid, tile=%ux%u oc=%u seed path\n",
           kOutH, kOutW, kChannels);

    for (uint32_t li = 0; li < sizeof(layers) / sizeof(layers[0]); ++li) {
        const yolo_layer_desc& l = layers[li];
        const uint32_t out_h = l.in_h - 2;
        const uint32_t out_w = l.in_w - 2;
        const lml::yolo::conv3x3_shape shape{l.in_h, l.in_w, l.cin, l.cout};
        std::vector<int8_t> weights(static_cast<size_t>(l.cout) * l.cin * 9);
        fill_random_weights(weights, seed);
        std::vector<uint8_t> out_full(static_cast<size_t>(l.cout) * out_h * out_w);
        std::vector<uint8_t> out_tiled(static_cast<size_t>(l.cout) * out_h * out_w);

        lml::yolo::conv3x3_stride1_valid(current.data(), out_full.data(),
                                         weights.data(), shape);
        if (!lml::yolo::conv3x3_stride1_valid_tiled_cpp(
                current.data(), out_tiled.data(), weights.data(), shape,
                kOutH, kOutW, kChannels)) {
            fprintf(stderr, "yolo-like C++ tiled check invalid shape/tile\n");
            return false;
        }
        const ptrdiff_t mismatch = lml::yolo::first_mismatch(
            out_full.data(), out_tiled.data(), out_full.size());
        if (mismatch >= 0) {
            fprintf(stderr,
                    "yolo-like C++ tiled mismatch layer=%s byte=%td full=0x%02x tiled=0x%02x\n",
                    l.name, mismatch, out_full[static_cast<size_t>(mismatch)],
                    out_tiled[static_cast<size_t>(mismatch)]);
            return false;
        }

        printf("  %-6s in=%ux%ux%u out=%ux%ux%u bytes=%zu PASS\n",
               l.name, l.cin, l.in_h, l.in_w, l.cout, out_h, out_w,
               out_full.size());
        current.swap(out_tiled);
    }

    printf("  PASS\n");
    return true;
}

static bool run_overlap_compare(acc_device& dev, uint32_t& seed) {
    constexpr uint32_t kTestInH = 32;
    constexpr uint32_t kTestInW = 32;
    constexpr uint32_t kTestCin = 16;
    constexpr uint32_t kTestCout = 4;
    const lml::yolo::conv3x3_shape shape{
        kTestInH, kTestInW, kTestCin, kTestCout};

    std::vector<uint8_t> input(static_cast<size_t>(kTestCin) *
                               kTestInH * kTestInW);
    std::vector<int8_t> weights(static_cast<size_t>(kTestCout) *
                                kTestCin * 9);
    std::vector<uint8_t> baseline(static_cast<size_t>(kTestCout) *
                                  kOutH * kOutW);
    std::vector<uint8_t> overlap(static_cast<size_t>(kTestCout) *
                                 kOutH * kOutW);
    fill_random_feature(input, seed);
    std::vector<int8_t> kernel(9);
    fill_random_weights(kernel, seed);
    for (uint32_t co = 0; co < kTestCout; ++co) {
        for (uint32_t ci = 0; ci < kTestCin; ++ci) {
            int8_t* k = weights.data() + (co * kTestCin + ci) * 9;
            for (uint32_t i = 0; i < 9; ++i) {
                k[i] = kernel[i];
            }
        }
    }

    lml::instruction_buffer ins(*dev.pool, kMaxInstructions);
    auto* batch_inputs = static_cast<uint8_t*>(
        dev.pool->alloc(static_cast<size_t>(kTestCin / kChannels) *
                        kInputBytes, 64));
    auto* batch_bias = static_cast<uint8_t*>(dev.pool->alloc(kBiasBytes, 64));
    auto* batch_out = static_cast<uint8_t*>(dev.pool->alloc(kOutputBytes, 64));
    if (!ins.valid() || !batch_inputs || !batch_bias || !batch_out) {
        fprintf(stderr, "overlap-compare allocation failed\n");
        return false;
    }

    std::vector<yolo_tile_job> jobs;
    jobs.push_back(yolo_tile_job{0, 0, 0, kOutH, kOutW});
    yolo_batch_profile base_profile;
    yolo_batch_profile overlap_profile;

    printf("overlap-compare:\n");
    printf("  shape in=%ux%ux%u out=%ux%ux%u groups=%u fixed-kernel\n",
           kTestCin, kTestInH, kTestInW, kTestCout, kOutH, kOutW,
           kTestCin / kChannels);

    if (!run_acc_conv_tile_batch(dev, input.data(), baseline.data(),
                                 weights.data(), kTestInH, kTestInW,
                                 kTestCin, kTestCout, 0, jobs, 0, 1,
                                 ins, batch_inputs, batch_bias, batch_out,
                                 &base_profile, false)) {
        return false;
    }
    if (!run_acc_conv_tile_batch(dev, input.data(), overlap.data(),
                                 weights.data(), kTestInH, kTestInW,
                                 kTestCin, kTestCout, 0, jobs, 0, 1,
                                 ins, batch_inputs, batch_bias, batch_out,
                                 &overlap_profile, true)) {
        return false;
    }

    const ptrdiff_t mismatch = lml::yolo::first_mismatch(
        baseline.data(), overlap.data(), baseline.size());
    printf("  baseline_ins=%" PRIu64 " overlap_ins=%" PRIu64 "\n",
           base_profile.instructions, overlap_profile.instructions);
    printf("  baseline_fpga=%.6f s overlap_fpga=%.6f s\n",
           base_profile.fpga_wait_ns / 1000000000.0,
           overlap_profile.fpga_wait_ns / 1000000000.0);
    if (mismatch >= 0) {
        fprintf(stderr,
                "overlap-compare mismatch byte=%td baseline=0x%02x overlap=0x%02x\n",
                mismatch, baseline[static_cast<size_t>(mismatch)],
                overlap[static_cast<size_t>(mismatch)]);
        return false;
    }
    printf("  PASS overlap matches baseline\n");
    return true;
}

static bool run_yolo_like_benchmark(acc_device& dev, uint32_t& seed) {
    const yolo_layer_desc layers[] = {
        {"p3", 160, 160, 4, 16},
        {"p4", 80, 80, 16, 32},
        {"p5", 40, 40, 16, 64},
    };

    std::vector<uint8_t> current(static_cast<size_t>(layers[0].cin) *
                                 layers[0].in_h * layers[0].in_w);
    fill_random_feature(current, seed);
    std::vector<uint8_t> current_cpu = current;

    uint32_t max_cin = 0;
    for (uint32_t i = 0; i < sizeof(layers) / sizeof(layers[0]); ++i) {
        if (layers[i].cin > max_cin) max_cin = layers[i].cin;
    }

    const uint32_t max_groups = max_cin / kChannels;
    const uint32_t max_theoretical_batch_tiles = 24;

    lml::instruction_buffer ins(*dev.pool, kMaxInstructions);
    auto* batch_inputs = static_cast<uint8_t*>(
        dev.pool->alloc(static_cast<size_t>(max_theoretical_batch_tiles) * max_groups *
                        kInputBytes, 64));
    auto* batch_bias = static_cast<uint8_t*>(dev.pool->alloc(kBiasBytes, 64));
    auto* batch_out = static_cast<uint8_t*>(
        dev.pool->alloc(static_cast<size_t>(max_theoretical_batch_tiles) *
                        kOutputBytes + 8, 64));
    if (!ins.valid() || !batch_inputs || !batch_bias || !batch_out) {
        fprintf(stderr, "yolo-like benchmark allocation failed\n");
        return false;
    }

    uint64_t total_macs = 0;
    uint64_t fpga_macs = 0;
    uint64_t cpu_edge_macs = 0;
    uint64_t cpu_baseline_ns = 0;
    uint64_t hybrid_ns = 0;
    yolo_batch_profile total_profile;

    printf("yolo-like benchmark:\n");
    printf("  layers=3, YOLO-like P3/P4/P5 3x3 stride1 valid, batched tiles, full tile writeback\n");
    printf("  read_conv_overlap=off\n");
    printf("  max_batch_tiles=%u max_instructions=%u\n",
           max_theoretical_batch_tiles, kMaxInstructions);

    for (uint32_t li = 0; li < sizeof(layers) / sizeof(layers[0]); ++li) {
        const yolo_layer_desc& l = layers[li];
        const uint32_t out_h = l.in_h - 2;
        const uint32_t out_w = l.in_w - 2;
        const lml::yolo::conv3x3_shape shape{l.in_h, l.in_w, l.cin, l.cout};
        std::vector<int8_t> weights(static_cast<size_t>(l.cout) * l.cin * 9);
        fill_random_weights(weights, seed);
        std::vector<uint8_t> out_cpu(static_cast<size_t>(l.cout) * out_h * out_w);
        std::vector<uint8_t> out_hybrid(static_cast<size_t>(l.cout) * out_h * out_w);

        const uint64_t layer_macs =
            static_cast<uint64_t>(l.cout) * l.cin * out_h * out_w * 9u;
        total_macs += layer_macs;

        const uint64_t c0 = monotonic_ns();
        lml::yolo::conv3x3_stride1_valid(current_cpu.data(), out_cpu.data(),
                                         weights.data(), shape);
        const uint64_t c1 = monotonic_ns();
        cpu_baseline_ns += c1 - c0;

        const uint64_t h0 = monotonic_ns();
        const uint32_t layer_groups = l.cin / kChannels;
        const bool overlap_read_conv = false;
        const uint32_t per_tile_instr =
            1u + layer_groups * (kChannels * kChannels + 3u) + 2u;
        uint32_t layer_batch_tiles = 1;
        if (kMaxInstructions > 3u + per_tile_instr) {
            layer_batch_tiles = (kMaxInstructions - 3u) / per_tile_instr;
        }
        if (layer_batch_tiles > max_theoretical_batch_tiles) {
            layer_batch_tiles = max_theoretical_batch_tiles;
        }
        if (layer_batch_tiles == 0) layer_batch_tiles = 1;
        yolo_batch_profile layer_profile;
        const uint32_t tile_h_covered = ((out_h + kOutH - 1u) / kOutH) * kOutH;
        const uint32_t tile_w_covered = ((out_w + kOutW - 1u) / kOutW) * kOutW;
        for (uint32_t oc = 0; oc < l.cout; oc += kChannels) {
            std::vector<yolo_tile_job> jobs;
            for (uint32_t oy = 0; oy < out_h; oy += kOutH) {
                for (uint32_t ox = 0; ox < out_w; ox += kOutW) {
                    const uint32_t valid_oh =
                        (oy + kOutH <= out_h) ? kOutH : (out_h - oy);
                    const uint32_t valid_ow =
                        (ox + kOutW <= out_w) ? kOutW : (out_w - ox);
                    jobs.push_back(yolo_tile_job{oc, oy, ox, valid_oh, valid_ow});
                }
            }
            for (size_t jb = 0; jb < jobs.size(); jb += layer_batch_tiles) {
                const size_t count =
                    std::min(static_cast<size_t>(layer_batch_tiles),
                             jobs.size() - jb);
                if (!run_acc_conv_tile_batch(dev, current.data(), out_hybrid.data(),
                                             weights.data(), l.in_h, l.in_w,
                                             l.cin, l.cout, oc, jobs, jb, count,
                                             ins, batch_inputs, batch_bias,
                                             batch_out, &layer_profile,
                                             overlap_read_conv)) {
                        return false;
                }
            }
            for (const yolo_tile_job& job : jobs) {
                fpga_macs += static_cast<uint64_t>(kChannels) * l.cin *
                             job.valid_oh * job.valid_ow * 9u;
            }
        }
        const uint64_t h1 = monotonic_ns();
        hybrid_ns += h1 - h0;

        if (memcmp(out_cpu.data(), out_hybrid.data(), out_cpu.size()) != 0) {
            for (size_t i = 0; i < out_cpu.size(); ++i) {
                if (out_cpu[i] != out_hybrid[i]) {
                    fprintf(stderr,
                            "yolo-like mismatch layer=%s byte=%zu cpu=0x%02x hybrid=0x%02x\n",
                            l.name, i, out_cpu[i], out_hybrid[i]);
                    return false;
                }
            }
        }

        printf("  %-6s in=%ux%ux%u out=%ux%ux%u macs=%" PRIu64
               " tiled=%ux%u batch_tiles=%u batches=%" PRIu64
               " avg_ins=%.1f edge_cpu=%s\n",
               l.name, l.cin, l.in_h, l.in_w, l.cout, out_h, out_w,
               layer_macs, tile_h_covered, tile_w_covered, layer_batch_tiles,
               layer_profile.batches,
               layer_profile.batches ? static_cast<double>(layer_profile.instructions) /
                                        static_cast<double>(layer_profile.batches) : 0.0,
               "no");
        total_profile.pack_ns += layer_profile.pack_ns;
        total_profile.build_ns += layer_profile.build_ns;
        total_profile.sync_ns += layer_profile.sync_ns;
        total_profile.fpga_wait_ns += layer_profile.fpga_wait_ns;
        total_profile.invalidate_ns += layer_profile.invalidate_ns;
        total_profile.copy_ns += layer_profile.copy_ns;
        total_profile.batches += layer_profile.batches;
        total_profile.tiles += layer_profile.tiles;
        total_profile.instructions += layer_profile.instructions;
        current.swap(out_hybrid);
        current_cpu.swap(out_cpu);
    }

    const double cpu_s = static_cast<double>(cpu_baseline_ns) / 1000000000.0;
    const double hybrid_s = static_cast<double>(hybrid_ns) / 1000000000.0;
    const double total_gmac = static_cast<double>(total_macs) / 1000000000.0;
    printf("  total_macs=%" PRIu64 " fpga_tile_macs=%" PRIu64
           " cpu_edge_macs=%" PRIu64 "\n",
           total_macs, fpga_macs, cpu_edge_macs);
    printf("  cpu_elapsed=%.6f s cpu_compute=%.3f GMAC/s %.3f GOPS\n",
           cpu_s, total_gmac / cpu_s, total_gmac * 2.0 / cpu_s);
    printf("  hybrid_elapsed=%.6f s hybrid_compute=%.3f GMAC/s %.3f GOPS\n",
           hybrid_s, total_gmac / hybrid_s, total_gmac * 2.0 / hybrid_s);
    const double ns_to_s = 1.0 / 1000000000.0;
    printf("  profile batches=%" PRIu64 " tiles=%" PRIu64
           " avg_ins=%.1f\n",
           total_profile.batches, total_profile.tiles,
           total_profile.batches ? static_cast<double>(total_profile.instructions) /
                                    static_cast<double>(total_profile.batches) : 0.0);
    printf("  profile pack=%.6f build=%.6f sync=%.6f fpga_wait=%.6f invalidate=%.6f copy=%.6f s\n",
           total_profile.pack_ns * ns_to_s,
           total_profile.build_ns * ns_to_s,
           total_profile.sync_ns * ns_to_s,
           total_profile.fpga_wait_ns * ns_to_s,
           total_profile.invalidate_ns * ns_to_s,
           total_profile.copy_ns * ns_to_s);
    printf("  speedup_vs_cpu=%.2fx\n", cpu_s / hybrid_s);
    printf("  PASS\n");
    return true;
}

} // namespace

int main(int argc, char** argv) {
    bool run_quant = true;
    bool run_conv = false;
    bool run_tp = false;
    bool run_read_bench = false;
    bool run_write_bench = false;
    bool run_conv_tp = false;
    bool run_conv_tp_sweep = false;
    bool run_writeback_basic = false;
    bool run_partial_writeback = false;
    bool run_overlap_cmp = false;
    bool run_yolo_cpp = false;
    bool run_yolo = false;
    uint32_t timeout_ms = 5000;
    uint32_t seed = 0x12345678u;
    uint32_t rounds = kDefaultQuantRounds;
    uint32_t batches = kDefaultThroughputBatches;
    uint32_t tasks = kDefaultThroughputTasks;
    uint32_t groups = kDefaultConvThroughputGroups;
    bool verify_all = true;
    const char* dump_dir = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--quant") == 0) {
            run_quant = true;
            run_conv = false;
        } else if (strcmp(argv[i], "--conv") == 0) {
            run_quant = false;
            run_conv = true;
            run_tp = false;
        } else if (strcmp(argv[i], "--all") == 0) {
            run_quant = true;
            run_conv = true;
            run_tp = false;
        } else if (strcmp(argv[i], "--throughput") == 0) {
            run_quant = false;
            run_conv = false;
            run_tp = true;
            run_read_bench = false;
            run_write_bench = false;
            run_conv_tp = false;
        } else if (strcmp(argv[i], "--read-bench") == 0) {
            run_quant = false;
            run_conv = false;
            run_tp = false;
            run_read_bench = true;
            run_write_bench = false;
            run_conv_tp = false;
            run_conv_tp_sweep = false;
            run_writeback_basic = false;
            run_partial_writeback = false;
            run_overlap_cmp = false;
            run_yolo_cpp = false;
            run_yolo = false;
        } else if (strcmp(argv[i], "--write-bench") == 0) {
            run_quant = false;
            run_conv = false;
            run_tp = false;
            run_read_bench = false;
            run_write_bench = true;
            run_conv_tp = false;
            run_conv_tp_sweep = false;
            run_writeback_basic = false;
            run_partial_writeback = false;
            run_overlap_cmp = false;
            run_yolo_cpp = false;
            run_yolo = false;
        } else if (strcmp(argv[i], "--conv-throughput") == 0) {
            run_quant = false;
            run_conv = false;
            run_tp = false;
            run_read_bench = false;
            run_write_bench = false;
            run_conv_tp = true;
            run_conv_tp_sweep = false;
            run_writeback_basic = false;
            run_partial_writeback = false;
            run_overlap_cmp = false;
            run_yolo_cpp = false;
            run_yolo = false;
            batches = kDefaultConvThroughputBatches;
        } else if (strcmp(argv[i], "--conv-throughput-sweep") == 0) {
            run_quant = false;
            run_conv = false;
            run_tp = false;
            run_read_bench = false;
            run_write_bench = false;
            run_conv_tp = false;
            run_conv_tp_sweep = true;
            run_writeback_basic = false;
            run_partial_writeback = false;
            run_overlap_cmp = false;
            run_yolo_cpp = false;
            run_yolo = false;
            batches = kDefaultConvThroughputBatches;
        } else if (strcmp(argv[i], "--writeback-basic") == 0) {
            run_quant = false;
            run_conv = false;
            run_tp = false;
            run_read_bench = false;
            run_write_bench = false;
            run_conv_tp = false;
            run_conv_tp_sweep = false;
            run_writeback_basic = true;
            run_partial_writeback = false;
            run_overlap_cmp = false;
            run_yolo_cpp = false;
            run_yolo = false;
        } else if (strcmp(argv[i], "--partial-writeback") == 0) {
            run_quant = false;
            run_conv = false;
            run_tp = false;
            run_read_bench = false;
            run_write_bench = false;
            run_conv_tp = false;
            run_conv_tp_sweep = false;
            run_writeback_basic = false;
            run_partial_writeback = true;
            run_overlap_cmp = false;
            run_yolo_cpp = false;
            run_yolo = false;
        } else if (strcmp(argv[i], "--overlap-compare") == 0) {
            run_quant = false;
            run_conv = false;
            run_tp = false;
            run_read_bench = false;
            run_write_bench = false;
            run_conv_tp = false;
            run_conv_tp_sweep = false;
            run_writeback_basic = false;
            run_partial_writeback = false;
            run_overlap_cmp = true;
            run_yolo_cpp = false;
            run_yolo = false;
        } else if (strcmp(argv[i], "--yolo-cpp") == 0) {
            run_quant = false;
            run_conv = false;
            run_tp = false;
            run_read_bench = false;
            run_write_bench = false;
            run_conv_tp = false;
            run_conv_tp_sweep = false;
            run_writeback_basic = false;
            run_partial_writeback = false;
            run_overlap_cmp = false;
            run_yolo_cpp = true;
            run_yolo = false;
        } else if (strcmp(argv[i], "--yolo-bench") == 0) {
            run_quant = false;
            run_conv = false;
            run_tp = false;
            run_read_bench = false;
            run_write_bench = false;
            run_conv_tp = false;
            run_conv_tp_sweep = false;
            run_writeback_basic = false;
            run_partial_writeback = false;
            run_overlap_cmp = false;
            run_yolo_cpp = true;
            run_yolo = true;
        } else if (strcmp(argv[i], "--timeout-ms") == 0 && i + 1 < argc) {
            timeout_ms = static_cast<uint32_t>(strtoul(argv[++i], nullptr, 0));
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = static_cast<uint32_t>(strtoul(argv[++i], nullptr, 0));
        } else if (strcmp(argv[i], "--rounds") == 0 && i + 1 < argc) {
            rounds = static_cast<uint32_t>(strtoul(argv[++i], nullptr, 0));
        } else if (strcmp(argv[i], "--batches") == 0 && i + 1 < argc) {
            batches = static_cast<uint32_t>(strtoul(argv[++i], nullptr, 0));
        } else if (strcmp(argv[i], "--tasks") == 0 && i + 1 < argc) {
            tasks = static_cast<uint32_t>(strtoul(argv[++i], nullptr, 0));
        } else if (strcmp(argv[i], "--groups") == 0 && i + 1 < argc) {
            groups = static_cast<uint32_t>(strtoul(argv[++i], nullptr, 0));
        } else if (strcmp(argv[i], "--sample-verify") == 0) {
            verify_all = false;
        } else if (strcmp(argv[i], "--verify-all") == 0) {
            verify_all = true;
        } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (run_yolo_cpp && !run_yolo) {
        return run_yolo_cpp_check(seed) ? 0 : 1;
    }

    const int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem");
        return 1;
    }

    mmio_region exch;
    if (!exch.map(fd, kExchPhysAddr, kExchSize)) {
        perror("mmap exchange DDR");
        close(fd);
        return 1;
    }

    mmio_region ctrl;
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

    tensors t(pool);
    if (!t.valid()) {
        fprintf(stderr, "tensor allocation failed\n");
        close(fd);
        return 1;
    }

    auto* regs = static_cast<volatile uint32_t*>(ctrl.addr);
    bool ok = true;

    if (run_quant) {
        ok = test_quant_writeback(regs, pool, t, timeout_ms, seed, rounds) && ok;
    }
    if (run_conv) {
        ok = test_random_conv(regs, pool, t, timeout_ms, seed, dump_dir) && ok;
    }
    if (run_tp) {
        ok = run_throughput(regs, pool, timeout_ms, seed, batches, tasks) && ok;
    }
    if (run_read_bench) {
        ok = run_read_benchmark(regs, pool, timeout_ms, seed, batches, tasks) && ok;
    }
    if (run_write_bench) {
        ok = run_write_benchmark(regs, pool, timeout_ms, batches, tasks) && ok;
    }
    if (run_conv_tp) {
        ok = run_conv_throughput(regs, pool, timeout_ms, seed, batches, groups,
                                 verify_all) && ok;
    }
    if (run_conv_tp_sweep) {
        ok = run_conv_throughput_sweep(regs, pool, timeout_ms, seed, batches,
                                       verify_all) && ok;
    }
    if (run_writeback_basic) {
        ok = test_writeback_basic(regs, pool, t, timeout_ms) && ok;
    }
    if (run_partial_writeback) {
        ok = test_partial_writeback(regs, pool, t, timeout_ms) && ok;
    }
    if (run_overlap_cmp) {
        acc_device dev{regs, &pool, timeout_ms};
        ok = run_overlap_compare(dev, seed) && ok;
    }
    if (run_yolo) {
        uint32_t cpp_seed = seed;
        ok = run_yolo_cpp_check(cpp_seed) && ok;
        acc_device dev{regs, &pool, timeout_ms};
        ok = run_yolo_like_benchmark(dev, seed) && ok;
    }

    close(fd);
    if (!ok) return 1;
    printf("PASS: acc_conv tests completed\n");
    return 0;
}
