#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include "tlsf.h"


#define MYCTRL_IOC_MAGIC 'M'
#define MYCTRL_IOC_GET_DMA_PHYS _IOR(MYCTRL_IOC_MAGIC, 0, uint64_t)
#define MYCTRL_IOC_GET_DMA_SIZE _IOR(MYCTRL_IOC_MAGIC, 1, uint32_t)
#define MYCTRL_IOC_GET_REG_PHYS _IOR(MYCTRL_IOC_MAGIC, 2, uint64_t)
#define MYCTRL_IOC_GET_REG_SIZE _IOR(MYCTRL_IOC_MAGIC, 3, uint32_t)

typedef struct{
    uint64_t l;
    uint64_t h;
}instruction;

int main(void)
{
    int fd;
    long page_size;
    uint64_t reg_phys = 0, dma_phys = 0;
    uint32_t reg_size = 0, dma_size = 0;
    void *reg_map, *dma_map;
    volatile uint32_t *reg;

    fd = open("/dev/myctrl0", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    ioctl(fd, MYCTRL_IOC_GET_REG_PHYS, &reg_phys);
    ioctl(fd, MYCTRL_IOC_GET_REG_SIZE, &reg_size);
    ioctl(fd, MYCTRL_IOC_GET_DMA_PHYS, &dma_phys);
    ioctl(fd, MYCTRL_IOC_GET_DMA_SIZE, &dma_size);

    printf("reg phys = 0x%llx size = 0x%x\n", (unsigned long long)reg_phys, reg_size);
    printf("dma phys = 0x%llx size = 0x%x\n", (unsigned long long)dma_phys, dma_size);

    page_size = sysconf(_SC_PAGESIZE);

    reg_map = mmap(NULL, reg_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0 * page_size);
    dma_map = mmap(NULL, dma_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 256 * page_size);

    if (reg_map == MAP_FAILED) {
        perror("mmap reg");
        close(fd);
        return 1;
    }

    if (dma_map == MAP_FAILED) {
        perror("mmap dma");
        munmap(reg_map, reg_size);
        close(fd);
        return 1;
    }
    reg = (uint32_t*)reg_map;

    tlsf_t pool = tlsf_create_with_pool(dma_map,dma_size);
    instruction* instructions = tlsf_memalign(pool,64,128/8*512);
    reg[1] = (uint32_t)instructions;
    reg[2] = 1;
    instructions[0].l = 0x4ULL;
    
    munmap(dma_map, dma_size);
    munmap(reg_map, reg_size);
    close(fd);
    return 0;
}