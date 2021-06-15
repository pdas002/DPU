#include <stdio.h>
#include <stdint.h>
#include "alex.h"
#include <alloc.h>
#include <mram.h>
#include <perfcounter.h>


#define BUFFER_SIZE 3136
#define MAX_READ 2048
__mram_noinit uint8_t inputdata[BUFFER_SIZE];

void ebnn_compute(float *input, uint8_t *output){
   perfcounter_config(COUNT_CYCLES, true);

  l_conv_pool_bn_bst0(input, temp1);
   perfcounter_t run_time = perfcounter_get();
   printf("Time: %lu\n", run_time);  

  l_b_conv_pool_bn_bst1(temp1, temp2);
  l_b_conv_bn_bst2(temp2, temp1);
  l_b_conv_bn_bst3(temp1, temp2);
  l_b_conv_pool_bn_bst4(temp2, temp1);
  l_b_linear_bn_softmax5(temp1, output);
}

int main()
{

   uint8_t output[1];

   float* f = mem_alloc(BUFFER_SIZE);
   mram_read(inputdata, f, MAX_READ);
   mram_read(inputdata+MAX_READ, (void*)f+MAX_READ, BUFFER_SIZE-MAX_READ);
   perfcounter_config(COUNT_CYCLES, true);


   ebnn_compute(f, output);
   perfcounter_t run_time = perfcounter_get();
   printf("Read Time: %lu\n", run_time);  

   printf("predicted: %d\n", output[0]);

  return 0;
}
