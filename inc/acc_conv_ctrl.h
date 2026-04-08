#ifndef lml_acc_conv
#define lml_acc_conv

#include<stdint.h>
#include<cassert>

struct alignas(16) instruction{
    uint64_t l;
    uint64_t h;
    instruction(){
    
    }
}

#endif