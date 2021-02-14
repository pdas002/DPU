#include <stdio.h>
#include <stdint.h>
#include "binary_mnist.h"
#include <alloc.h>
#include <mram.h>
#include <perfcounter.h>

#define BUFFER_SIZE 392
__mram_noinit uint8_t inputdata[BUFFER_SIZE];

void ebnn_compute(uint8_t *input, uint8_t *output){
   perfcounter_config(COUNT_CYCLES, true);

  l_b_conv_pool_bn_bst0(input, temp1);
   perfcounter_t run_time = perfcounter_get();
   printf("Time: %lu\n", run_time);  

  l_b_linear_bn_softmax1(temp1, output);
}


int main()
{
 

  uint8_t output[1];

   uint8_t* input = mem_alloc(BUFFER_SIZE);
   mram_read(inputdata, input, BUFFER_SIZE);
 
 for(int j = 0; j < 4; j++) {
   ebnn_compute(&input[98*j], output);
   printf("predicted: %d\n", output[0]);

   }
  return 0;
}
