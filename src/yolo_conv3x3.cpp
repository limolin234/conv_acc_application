#include "yolo_conv3x3.hpp"

#include <algorithm>

namespace lml {
namespace yolo {

uint8_t quantize_acc(int32_t acc) {
    const int64_t prod = static_cast<int64_t>(acc);
    const int64_t rounding = (prod >= 0) ? (int64_t{1} << 7) : -(int64_t{1} << 7);
    int32_t q = static_cast<int32_t>((prod + rounding) >> 8);
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    return static_cast<uint8_t>(static_cast<int8_t>(q));
}

size_t nchw_offset(uint32_t c, uint32_t h, uint32_t w,
                   uint32_t H, uint32_t W) {
    return (static_cast<size_t>(c) * H + h) * W + w;
}

void conv3x3_stride1_valid_region(const uint8_t* input, uint8_t* output,
                                  const int8_t* weights,
                                  const conv3x3_shape& shape,
                                  const conv3x3_tile& tile) {
    const uint32_t out_h = shape.in_h - 2;
    const uint32_t out_w = shape.in_w - 2;
    const uint32_t oc_end = std::min(shape.cout, tile.oc0 + tile.oc_count);
    const uint32_t oy_end = std::min(out_h, tile.oy0 + tile.oh);
    const uint32_t ox_end = std::min(out_w, tile.ox0 + tile.ow);

    for (uint32_t co = tile.oc0; co < oc_end; ++co) {
        for (uint32_t oy = tile.oy0; oy < oy_end; ++oy) {
            for (uint32_t ox = tile.ox0; ox < ox_end; ++ox) {
                int32_t acc = 0;
                for (uint32_t ci = 0; ci < shape.cin; ++ci) {
                    const int8_t* k =
                        weights + (static_cast<size_t>(co) * shape.cin + ci) * 9;
                    for (uint32_t ky = 0; ky < 3; ++ky) {
                        for (uint32_t kx = 0; kx < 3; ++kx) {
                            const int8_t in = static_cast<int8_t>(
                                input[nchw_offset(ci, oy + ky, ox + kx,
                                                  shape.in_h, shape.in_w)]);
                            acc += static_cast<int32_t>(in) *
                                   static_cast<int32_t>(k[ky * 3 + kx]);
                        }
                    }
                }
                output[nchw_offset(co, oy, ox, out_h, out_w)] =
                    quantize_acc(acc);
            }
        }
    }
}

void conv3x3_stride1_valid(const uint8_t* input, uint8_t* output,
                           const int8_t* weights,
                           const conv3x3_shape& shape) {
    conv3x3_tile all{0, shape.cout, 0, 0, shape.in_h - 2, shape.in_w - 2};
    conv3x3_stride1_valid_region(input, output, weights, shape, all);
}

bool conv3x3_stride1_valid_tiled_cpp(const uint8_t* input, uint8_t* output,
                                     const int8_t* weights,
                                     const conv3x3_shape& shape,
                                     uint32_t tile_oh, uint32_t tile_ow,
                                     uint32_t tile_oc) {
    if (!input || !output || !weights || shape.in_h < 3 || shape.in_w < 3 ||
        shape.cin == 0 || shape.cout == 0 || tile_oh == 0 || tile_ow == 0 ||
        tile_oc == 0) {
        return false;
    }

    const uint32_t out_h = shape.in_h - 2;
    const uint32_t out_w = shape.in_w - 2;
    for (uint32_t oc = 0; oc < shape.cout; oc += tile_oc) {
        for (uint32_t oy = 0; oy < out_h; oy += tile_oh) {
            for (uint32_t ox = 0; ox < out_w; ox += tile_ow) {
                conv3x3_tile tile{
                    oc,
                    std::min(tile_oc, shape.cout - oc),
                    oy,
                    ox,
                    std::min(tile_oh, out_h - oy),
                    std::min(tile_ow, out_w - ox),
                };
                conv3x3_stride1_valid_region(input, output, weights, shape, tile);
            }
        }
    }
    return true;
}

ptrdiff_t first_mismatch(const uint8_t* lhs, const uint8_t* rhs, size_t bytes) {
    for (size_t i = 0; i < bytes; ++i) {
        if (lhs[i] != rhs[i]) {
            return static_cast<ptrdiff_t>(i);
        }
    }
    return -1;
}

} // namespace yolo
} // namespace lml
