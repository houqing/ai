#include <stdio.h>
#include<sys/time.h>
#include <malloc.h>
#include <stdint.h> 

/*****************************************************************************
 函数名  : ars
 功能描述: pseudo random number, 128bit block, use fixed 2 rounds generation
 输入参数: rand_num     random numbers
           input        128bit input buffer pointer
           key          128bit asr key used to generate the psuedo random numbers
 输出参数: out          bitmask output buffer pointer
 返回值  : void
*****************************************************************************/
inline void ars(const uint64_t rand_num, const float ratio, const uint8_t *in, const uint8_t *key, uint8_t *out)
{   
    const uint16_t threshold = INT16_MAX * ratio;
    const uint8_t in_offset[16] = {0x01};
    const uint8_t inc_step[16] = {0x02};
    const uint8_t key_const[16] = {0xBB,0x67,0xAE,0x85,0x84,0xCA,0xA7,0x3B,0x9E,0x37,0x79,0xB9,0x7F,0x4A,0x7C,0x15};
    const uint8_t bit_mask[16] = {0x00,0x80,0x00,0x80,0x00,0x80,0x00,0x80,0x00,0x80,0x00,0x80,0x00,0x80,0x00,0x80};
    
    const uint64_t loop_time = rand_num / 16;
    __asm volatile(
        ".arch  armv8-a+crypto \n"
 
        "ldr x0, %[loop_time] \n"
        "mov x6, #0x8 \n"   // 128bit has 8 set of u16
 
        "ld1 {v2.16b}, %[key_const] \n" // save key const
 
        "ldr x1, %[in] \n"
        "ld1 {v0.16b}, [x1] \n" // tmp input
        "ldr x2, %[key] \n"
        "ld1 {v1.16b}, [x2] \n" // first round key
        "add v5.2d, v1.2d, v2.2d \n" // second round key

        // generate in2
        "ldp x10, x11, %[in_offset] \n"
        "ldp x12, x13, [x1] \n"
        "adds x14, x12, x10 \n"
        "adc x15, x13, x11 \n"
        "mov v10.d[0], x14 \n"
        "mov v10.d[1], x15 \n"
        
        "ldr x4, %[key] \n"
        "ld1 {v11.16b}, [x4] \n" // first round key
        "add v15.2d, v11.2d, v2.2d \n" // second round key
        
        // save input inc step
    #ifdef CONFIG_ENABLE_PERIOD_64BIT
        "ld1 {v4.16b}, %[inc_step] \n"
    #else        
        "ldp x10, x11, %[inc_step] \n"
    #endif
   
        // save input and key to register
        "ld1 {v3.16b}, [x1] \n"
        //"ld1 {v5.16b}, [x2] \n"
        
        // save input and key to register
        "mov v13.16b, v10.16b \n"
        //"ld1 {v15.16b}, [x4] \n"
        
        // bit mask
        "ldr w7, %[threshold] \n"
        "dup v20.8h, w7 \n"
        //"ld1 {v20.8h}, %[threshold] \n"
        "ld1 {v21.16b}, %[bit_mask] \n"
        //save out pointer addr to register
        "ldr x5, %[out] \n"  
    ".ARS: \n"
        // tmp = aese tmp, key
        "aese   v0.16b, v1.16b \n" 
        // tmp = aesmc tmp, tmp
        "aesmc  v0.16b, v0.16b \n" 
        // tmp = aese tmp, key
        //"add v1.2d, v1.2d, v2.2d \n" 
        // tmp = aese tmp, key
        "aese   v10.16b, v11.16b \n"
        // tmp = aesmc tmp, tmp
        "aesmc  v10.16b, v10.16b \n"
        // key = key + key_const
        //"add v11.2d, v11.2d, v2.2d \n"
         
        // last round we don't do aesmc
        // tmp = aese tmp, key
        "aese v0.16b, v5.16b \n"
        "aesmc  v0.16b,v0.16b \n"
        "aese v10.16b, v15.16b \n"
        "aesmc  v10.16b,v10.16b \n"
        //"eor v0.16b, v0.16b, v1.16b \n"
        //"eor v10.16b, v10.16b, v11.16b \n"
       
        //gen random num bit mask
        "ushr v0.8h, v0.8h, #0x01 \n"
        "sub v22.8h, v0.8h, v20.8h \n"
        "ushr v10.8h, v10.8h, #0x01 \n"
        "sub v23.8h, v10.8h, v20.8h \n"

        // out group-1, step-1
        "bit v29.16b, v22.16b, v21.16b \n"
        // out group-1, step-2
        "ushr v29.8h, v29.8h, #0x1 \n"

        // out group-2, step-1
        "bit v29.16b, v23.16b, v21.16b \n"

        "subs x6, x6, #1 \n"
        "bne .L0 \n"
        "mov x6, #0x8 \n"

        "st1 {v29.16b}, [x5] \n"
        "add x5, x5, 16 \n"

    ".L0: \n"
        // out group-2, step-2
        "ushr v29.8h, v29.8h, #0x1 \n"

    #ifdef CONFIG_ENABLE_PERIOD_64BIT
        "add v3.2d, v3.2d, v4.2d \n"
        "add v13.2d, v13.2d, v4.2d \n"
    #else
        "mov x12, v3.d[0] \n"
        "mov x13, v3.d[1] \n"
        "adds x14, x12, x10 \n"
        "adc x15, x13, x11 \n"
        "mov v3.d[0], x14 \n"
        "mov v3.d[1], x15 \n"

        "mov x12, v13.d[0] \n"
        "mov x13, v13.d[1] \n"
        "adds x14, x12, x10 \n"
        "adc x15, x13, x11 \n"
        "mov v13.d[0], x14 \n"
        "mov v13.d[1], x15 \n"
    #endif
        
        // restore input
        "mov v0.16b, v3.16b \n"
        "mov v10.16b, v13.16b \n"
        
        //restore key
        //"mov v1.16b, v5.16b \n"
        //"mov v11.16b, v15.16b \n"
        
        "subs	x0, x0, 1 \n"
        "bne	.ARS \n"
        :
        :[in] "m" (in), [out] "m" (out), [in_offset] "m" (in_offset), [key] "m" (key), [key_const] "m" (key_const), [inc_step] "m" (inc_step), [bit_mask] "m" (bit_mask), [loop_time] "m" (loop_time), [threshold] "m" (threshold)
        :"x0", "x1", "x2", "x3", "x4", "x5", "x6", "w7", "x7", "x10", "x11", "x12", "x13", "x14", "x15", "v0", "v1", "v2", "v3", "v4", "v5", "v10", "v11", "v12", "v13", "v14", "v15", "v20", "v21", "v22", "v23", "v29", "memory"
       );
}