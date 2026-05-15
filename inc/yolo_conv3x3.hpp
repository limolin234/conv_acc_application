#ifndef LML_YOLO_CONV3X3_HPP
#define LML_YOLO_CONV3X3_HPP

#include <stddef.h>
#include <stdint.h>

namespace lml {
namespace yolo {

struct conv3x3_shape {
    uint32_t in_h;
    uint32_t in_w;
    uint32_t cin;
    uint32_t cout;
};

struct conv3x3_tile {
    uint32_t oc0;
    uint32_t oc_count;
    uint32_t oy0;
    uint32_t ox0;
    uint32_t oh;
    uint32_t ow;
};

uint8_t quantize_acc(int32_t acc);

size_t nchw_offset(uint32_t c, uint32_t h, uint32_t w,
                   uint32_t H, uint32_t W);

void conv3x3_stride1_valid_region(const uint8_t* input, uint8_t* output,
                                  const int8_t* weights,
                                  const conv3x3_shape& shape,
                                  const conv3x3_tile& tile);

void conv3x3_stride1_valid(const uint8_t* input, uint8_t* output,
                           const int8_t* weights,
                           const conv3x3_shape& shape);

bool conv3x3_stride1_valid_tiled_cpp(const uint8_t* input, uint8_t* output,
                                     const int8_t* weights,
                                     const conv3x3_shape& shape,
                                     uint32_t tile_oh, uint32_t tile_ow,
                                     uint32_t tile_oc);

ptrdiff_t first_mismatch(const uint8_t* lhs, const uint8_t* rhs, size_t bytes);

} // namespace yolo
} // namespace lml

#endif // LML_YOLO_CONV3X3_HPP
