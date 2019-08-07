#include <stdio.h>
#include<sys/time.h>
#include <malloc.h>
#include "ars_v0.9.1.h"

int main()
{
    //uint64_t rand_num = 524288L;
    //uint64_t rand_num = 4194304L;
    //uint64_t rand_num = 16777216L;
    //uint64_t rand_num = 134217728L;
    //uint64_t rand_num = 813694976L;
    uint64_t rand_num = 4043309056L;
    //uint64_t rand_num = 4096L*2;
    const uint8_t in[16] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    //const uint8_t in2[16] = {0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    const uint8_t key[16] = {0x22,0x21,0x22,0x23,0x24,0x25,0x26,0x27,0x28,0x29,0x2a,0x2b,0x2c,0x2d,0x00,0xff};
    uint8_t *out = (uint8_t *)calloc(1 ,(rand_num/8)*sizeof(uint8_t));
    
    struct timeval start;
    struct timeval end;
    gettimeofday(&start,NULL);
    ars(rand_num, 0.74, in, key, out);
    gettimeofday(&end,NULL);
    int cost = (end.tv_sec-start.tv_sec)*1000000+(end.tv_usec-start.tv_usec);
    printf("time cost: %d us\n",cost);
    
    printf("bitmask=");
    for(int i = 0; i != (rand_num / 8); ++i)
    {
        #if 0
        printf("%02X", out[i]);
        if( (i+1) % 16 == 0) printf("\nbitmask=");
        #endif
    }
    printf("\n");
    
    free(out);
    
    return 0;
}
