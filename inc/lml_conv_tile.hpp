#ifndef LML_CONV_TILE_HPP
#define LML_CONV_TILE_HPP

#include <stdint.h>

#include "lml_acc_conv.hpp"

namespace lml {
namespace conv {

struct tile_shape {
    uint8_t channels = 4;
    uint8_t in_h = kDefaultConvInputH;
    uint16_t in_w = kDefaultConvInputW;
    uint8_t out_h = kDefaultConvInputH - 2;
    uint16_t out_w = kDefaultConvInputW - 2;
};

struct tensor_layout {
    uint32_t channel_stride = 0;
    uint16_t row_stride = 0;
};

struct write_back_config {
    int32_t q_multiplier = 1;
    uint8_t q_shift = 8;
    int16_t q_zero_point = 0;
};

struct kernel_pack {
    const instruction* regs = nullptr;
    uint32_t count = 0;
};

struct tile_task {
    uint32_t input_phys = 0;
    uint32_t bias_phys = 0;
    uint32_t output_phys = 0;
    tensor_layout input_layout;
    tensor_layout bias_layout;
    tensor_layout output_layout;
    tile_shape shape;
    uint8_t ci_mask = 0x0f;
    uint8_t co_mask = 0x0f;
    write_back_config write_cfg;
    kernel_pack kernels;
    bool sync_after = true;
};

class tile_program_builder {
public:
    explicit tile_program_builder(instruction_buffer& ins)
        : ins_(ins) {}

    void clear() {
        ins_.clear();
    }

    bool add_tile(const tile_task& task) {
        return push(instr::config_regs(task.input_layout.channel_stride,
                                       task.input_layout.row_stride,
                                       task.shape.channels,
                                       task.shape.in_h,
                                       task.shape.in_w)) &&
               push_kernels(task.kernels) &&
               push(instr::bias(task.bias_phys,
                                task.bias_layout.channel_stride,
                                task.bias_layout.row_stride,
                                task.shape.channels,
                                task.shape.out_h,
                                task.shape.out_w * 4u)) &&
               push(instr::read_conv(true, task.input_phys, false,
                                     task.ci_mask, task.co_mask,
                                     task.shape.in_h, task.shape.in_w)) &&
               push(instr::read_conv(false, task.input_phys, true,
                                     task.ci_mask, task.co_mask,
                                     task.shape.in_h, task.shape.in_w)) &&
               push(instr::write_back_config(task.write_cfg.q_multiplier,
                                             task.write_cfg.q_shift,
                                             task.write_cfg.q_zero_point)) &&
               push(instr::write_axi(task.output_phys,
                                     task.output_layout.channel_stride,
                                     task.output_layout.row_stride,
                                     task.shape.channels,
                                     task.shape.out_h,
                                     task.shape.out_w)) &&
               push(instr::bias(task.bias_phys,
                                task.bias_layout.channel_stride,
                                task.bias_layout.row_stride,
                                task.shape.channels,
                                task.shape.out_h,
                                task.shape.out_w * 4u)) &&
               (!task.sync_after || push(instr::mode(0x1)));
    }

    bool finish() {
        return push(instr::mode(0x1)) && push(instr::mode(0x0));
    }

private:
    bool push(instruction ins) {
        return ins_.push(ins);
    }

    bool push_kernels(const kernel_pack& kernels) {
        for (uint32_t i = 0; i < kernels.count; ++i) {
            if (!push(kernels.regs[i])) {
                return false;
            }
        }
        return true;
    }

    instruction_buffer& ins_;
};

} // namespace conv
} // namespace lml

#endif // LML_CONV_TILE_HPP
