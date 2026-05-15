#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "yolo_conv3x3.hpp"

namespace {

constexpr uint32_t kTileH = 30;
constexpr uint32_t kTileW = 30;
constexpr uint32_t kTileC = 4;

struct layer_desc {
    const char* name;
    uint32_t in_h;
    uint32_t in_w;
    uint32_t cin;
    uint32_t cout;
};

static uint32_t xorshift32(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

static uint8_t rand_u8(uint32_t& s, uint8_t mod) {
    return static_cast<uint8_t>(xorshift32(s) % mod);
}

static int8_t rand_i8_small(uint32_t& s) {
    return static_cast<int8_t>(static_cast<int>(rand_u8(s, 5)) - 2);
}

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

static bool run_yolo_cpp_check(uint32_t seed) {
    const layer_desc layers[] = {
        {"p3", 160, 160, 4, 16},
        {"p4", 80, 80, 16, 32},
        {"p5", 40, 40, 16, 64},
    };

    std::vector<uint8_t> current(static_cast<size_t>(layers[0].cin) *
                                 layers[0].in_h * layers[0].in_w);
    fill_random_feature(current, seed);

    printf("yolo-like C++ tiled check:\n");
    printf("  layers=3, 3x3 stride1 valid, tile=%ux%u oc=%u\n",
           kTileH, kTileW, kTileC);

    for (uint32_t li = 0; li < sizeof(layers) / sizeof(layers[0]); ++li) {
        const layer_desc& l = layers[li];
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
                kTileH, kTileW, kTileC)) {
            fprintf(stderr, "invalid shape/tile for layer=%s\n", l.name);
            return false;
        }
        const ptrdiff_t mismatch = lml::yolo::first_mismatch(
            out_full.data(), out_tiled.data(), out_full.size());
        if (mismatch >= 0) {
            fprintf(stderr,
                    "mismatch layer=%s byte=%td full=0x%02x tiled=0x%02x\n",
                    l.name, mismatch, out_full[static_cast<size_t>(mismatch)],
                    out_tiled[static_cast<size_t>(mismatch)]);
            return false;
        }

        const uint64_t macs =
            static_cast<uint64_t>(l.cout) * l.cin * out_h * out_w * 9u;
        printf("  %-6s in=%ux%ux%u out=%ux%ux%u macs=%" PRIu64 " PASS\n",
               l.name, l.cin, l.in_h, l.in_w, l.cout, out_h, out_w, macs);
        current.swap(out_tiled);
    }

    printf("  PASS\n");
    return true;
}

} // namespace

int main(int argc, char** argv) {
    uint32_t seed = 0x12345678u;
    if (argc == 3 && strcmp(argv[1], "--seed") == 0) {
        seed = static_cast<uint32_t>(strtoul(argv[2], nullptr, 0));
    } else if (argc != 1) {
        fprintf(stderr, "Usage: %s [--seed N]\n", argv[0]);
        return 2;
    }
    return run_yolo_cpp_check(seed) ? 0 : 1;
}
