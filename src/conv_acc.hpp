#ifndef LML_CONV_ACC
#define LML_CONV_ACC

#include <stdint.h>

namespace lml{
    struct instruction{
        uint64_t l;
        uint64_t h;
        instruction(uint64_t h, uint64_t l) : h(h), l(l) {}
    };

    class accelerator{
    public:
        accelerator(uint32_t* regmap) : regmap(regmap) {}
        ~accelerator();
        void run(instruction* instructions, size_t num_instructions){
            regmap[1] = (uint32_t)instructions;
            regmap[2] = num_instructions;
            regmap[0] |= 0x00000002; // start
        };
    private:
        uint32_t* regmap;
    };
    
}
#endif
