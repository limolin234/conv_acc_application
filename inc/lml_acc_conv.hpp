#ifndef LML_ACC_CONV_HPP
#define LML_ACC_CONV_HPP

#include <stddef.h>
#include <stdint.h>
#include "tlsf.h"

#ifndef LML_ACC_CONV_HAS_TLSF_HEADER
extern "C" {
typedef void* tlsf_t;
tlsf_t tlsf_create_with_pool(void* mem, size_t bytes);
void* tlsf_memalign(tlsf_t tlsf, size_t align, size_t bytes);
}
#endif

namespace lml {

static constexpr uint32_t kDefaultExchPhysAddr = 0x30000000u;
static constexpr uint8_t kDefaultConvInputH = 32;
static constexpr uint8_t kDefaultConvInputW = 32;
static constexpr uint32_t kDefaultTensorChannelStrideBytes = 512 * 16;

struct instruction {
    uint64_t lo;
    uint64_t hi;

    instruction() : lo(0), hi(0) {}
    instruction(uint64_t hi_, uint64_t lo_) : lo(lo_), hi(hi_) {}
};

static_assert(sizeof(instruction) == 16, "instruction must be exactly 128 bits");

class exch_pool {
public:
    exch_pool(void* virt_base, uint32_t phys_base, size_t bytes)
        : virt_base_(static_cast<uint8_t*>(virt_base)),
          phys_base_(phys_base),
          bytes_(bytes),
          tlsf_(tlsf_create_with_pool(virt_base, bytes)) {}

    bool valid() const {
        return tlsf_ != nullptr;
    }

    void* alloc(size_t bytes, size_t align = 64) {
        if (!tlsf_) return nullptr;
        return tlsf_memalign(tlsf_, align, bytes);
    }

    uint32_t phys(const void* ptr) const {
        return phys_base_ + static_cast<uint32_t>(
            static_cast<const uint8_t*>(ptr) - virt_base_);
    }

    bool owns(const void* ptr) const {
        const uint8_t* p = static_cast<const uint8_t*>(ptr);
        return p >= virt_base_ && p < virt_base_ + bytes_;
    }

    void* virt_base() const { return virt_base_; }
    uint32_t phys_base() const { return phys_base_; }
    size_t bytes() const { return bytes_; }

private:
    uint8_t* virt_base_;
    uint32_t phys_base_;
    size_t bytes_;
    tlsf_t tlsf_;
};

class instruction_buffer {
public:
    instruction_buffer(exch_pool& pool, size_t capacity, size_t align = 64)
        : pool_(&pool),
          data_(static_cast<instruction*>(
              pool.alloc(sizeof(instruction) * capacity, align))),
          size_(0),
          capacity_(data_ ? capacity : 0) {}

    bool valid() const {
        return data_ != nullptr;
    }

    bool push(instruction ins) {
        if (size_ >= capacity_) return false;
        data_[size_++] = ins;
        return true;
    }

    void clear() {
        size_ = 0;
    }

    instruction& operator[](size_t index) {
        return data_[index];
    }

    const instruction& operator[](size_t index) const {
        return data_[index];
    }

    instruction* data() { return data_; }
    const instruction* data() const { return data_; }

    uint32_t phys_addr() const {
        return pool_->phys(data_);
    }

    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }

private:
    exch_pool* pool_;
    instruction* data_;
    size_t size_;
    size_t capacity_;
};

namespace instr {

static inline instruction raw(uint64_t hi, uint64_t lo) {
    return instruction(hi, lo);
}

static inline void set_bit(uint64_t& hi, uint64_t& lo, uint8_t bit,
                           uint64_t value) {
    if ((value & 1u) == 0) return;
    if (bit >= 64) {
        hi |= uint64_t(1) << (bit - 64);
    } else {
        lo |= uint64_t(1) << bit;
    }
}

static inline void set_field(uint64_t& hi, uint64_t& lo, uint8_t lsb,
                             uint8_t width, uint64_t value) {
    for (uint8_t i = 0; i < width; ++i) {
        set_bit(hi, lo, lsb + i, value >> i);
    }
}

static inline instruction mode(uint8_t op) {
    uint64_t hi = 0;
    uint64_t lo = 0;
    set_field(hi, lo, 120, 8, op);
    return raw(hi, lo);
}

static inline instruction nop() {
    return mode(0x00u);
}

static inline instruction sync() {
    return mode(0x01u);
}

static inline instruction regs(uint8_t addr, uint64_t data_lo,
                               uint16_t data_hi = 0) {
    uint64_t hi = 0;
    uint64_t lo = 0;
    set_field(hi, lo, 120, 8, 0x02u);
    set_field(hi, lo, 112, 8, static_cast<uint8_t>(addr + 1u));
    set_field(hi, lo, 40, 64, data_lo);
    set_field(hi, lo, 104, 8, data_hi & 0x00ffu);
    return raw(hi, lo);
}

static inline instruction write_back_config(int32_t q_multiplier,
                                            uint8_t q_shift,
                                            int16_t q_zero_point) {
    uint64_t hi = 0;
    uint64_t lo = 0;
    set_field(hi, lo, 120, 8, 0x02u);
    set_field(hi, lo, 112, 8, 17u);
    set_field(hi, lo, 80, 32, static_cast<uint32_t>(q_multiplier));
    set_field(hi, lo, 74, 6, q_shift & 0x3fu);
    set_field(hi, lo, 58, 16, static_cast<uint16_t>(q_zero_point));
    return raw(hi, lo);
}

static inline instruction config_regs(uint32_t c_offset, uint16_t h_offset,
                                      uint8_t c, uint8_t h, uint16_t w) {
    (void)c_offset;
    (void)h_offset;
    (void)h;
    (void)w;
    const uint8_t mask = (c >= 4u) ? 0x0fu : static_cast<uint8_t>((1u << c) - 1u);
    uint64_t hi = 0;
    uint64_t lo = 0;
    set_field(hi, lo, 120, 8, 0x02u);
    set_field(hi, lo, 112, 8, 0u);
    set_field(hi, lo, 108, 4, mask);
    set_field(hi, lo, 104, 4, mask);
    return raw(hi, lo);
}

static inline instruction conv_config(uint8_t ci_mask, uint8_t co_mask) {
    uint64_t hi = 0;
    uint64_t lo = 0;
    set_field(hi, lo, 120, 8, 0x02u);
    set_field(hi, lo, 112, 8, 0u);
    set_field(hi, lo, 108, 4, ci_mask & 0x0fu);
    set_field(hi, lo, 104, 4, co_mask & 0x0fu);
    return raw(hi, lo);
}

static inline instruction mem(uint8_t op, uint8_t cmd, uint32_t base_addr,
                              uint32_t c_offset, uint16_t h_offset,
                              uint8_t c, uint8_t h, uint16_t w) {
    uint64_t hi = 0;
    uint64_t lo = 0;
    set_field(hi, lo, 120, 8, op);
    set_field(hi, lo, 119, 1, cmd & 0x01u);
    set_field(hi, lo, 87, 32, base_addr);
    set_field(hi, lo, 55, 32, c_offset);
    set_field(hi, lo, 39, 16, h_offset);
    set_field(hi, lo, 36, 3, c & 0x07u);
    set_field(hi, lo, 30, 6, h & 0x3fu);
    set_field(hi, lo, 14, 16, w);
    return raw(hi, lo);
}

static inline instruction read_to_ipp(uint32_t base_addr, uint32_t c_offset,
                                      uint16_t h_offset, uint8_t c, uint8_t h,
                                      uint16_t w) {
    return mem(0x03u, 0u, base_addr, c_offset, h_offset, c, h, w);
}

static inline instruction read_to_opp(uint32_t base_addr, uint32_t c_offset,
                                      uint16_t h_offset, uint8_t c, uint8_t h,
                                      uint16_t w) {
    return mem(0x03u, 1u, base_addr, c_offset, h_offset, c, h, w);
}

static inline instruction bias(uint32_t base_addr, uint32_t c_offset,
                               uint16_t h_offset, uint8_t c, uint8_t h,
                               uint16_t w) {
    return read_to_opp(base_addr, c_offset, h_offset, c, h, w);
}

static inline instruction write_back(uint32_t base_addr, uint32_t c_offset,
                                     uint16_t h_offset, uint8_t c, uint8_t h,
                                     uint16_t w) {
    return mem(0x05u, 0u, base_addr, c_offset, h_offset, c, h, w);
}

static inline instruction write_axi(uint32_t base_addr, uint32_t c_offset,
                                    uint16_t h_offset, uint8_t c, uint8_t h,
                                    uint16_t w) {
    return write_back(base_addr, c_offset, h_offset, c, h, w);
}

static inline instruction conv_run(uint8_t h, uint8_t w) {
    uint64_t hi = 0;
    uint64_t lo = 0;
    set_field(hi, lo, 120, 8, 0x04u);
    set_field(hi, lo, 119, 1, 0u);
    set_field(hi, lo, 111, 8, h);
    set_field(hi, lo, 103, 8, w);
    return raw(hi, lo);
}

static inline instruction read_conv(bool read_en, uint32_t base_addr,
                                    bool conv_en, uint8_t ci_mask,
                                    uint8_t co_mask,
                                    uint8_t h = kDefaultConvInputH,
                                    uint8_t w = kDefaultConvInputW) {
    (void)ci_mask;
    (void)co_mask;
    if (read_en) {
        return read_to_ipp(base_addr, kDefaultTensorChannelStrideBytes, w,
                           4, h, w);
    }
    if (conv_en) {
        return conv_run(h, w);
    }
    return nop();
}

} // namespace instr

} // namespace lml

#endif // LML_ACC_CONV_HPP
