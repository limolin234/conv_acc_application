#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

#define EXCH_PHYS_ADDR  0x30000000ULL
#define EXCH_SIZE       0x10000000UL

int main(void)
{
    int fd;
    void *exch_map;

    fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem");
        return 1;
    }

    exch_map = mmap(NULL, EXCH_SIZE,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED,
                    fd,
                    EXCH_PHYS_ADDR);
    if (exch_map == MAP_FAILED) {
        perror("mmap exchange");
        close(fd);
        return 1;
    }

    printf("exchange map = %p\n", exch_map);

    tlsf_t pool = tlsf_create_with_pool(exch_map, EXCH_SIZE);

    munmap(exch_map, EXCH_SIZE);
    close(fd);
    return 0;
}
