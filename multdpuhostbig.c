#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include "mnist.h"
#ifndef DPU_BINARY
#define DPU_BINARY "./multdpualexbig"
#endif

#define BUFFER_SIZE 489216
#define LBUFFER_SIZE 624

int main(void) {
    uint32_t count;
    int total = 0;
    load_mnist();

    float* input_data = malloc(9984*784*sizeof(float));

    for(int i=0; i<9984; i++)
    {
        for(int j=0;j<784;j++)

        {

                input_data[i*784+j] = (float)test_image[i][j];

        }

    }

  struct dpu_set_t set, dpu;

  DPU_ASSERT(dpu_alloc(1, NULL, &set));
  DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));


  printf("Moving input data to MRAM \n");
  int dpunum = 0;
  DPU_FOREACH(set, dpu) {
    // each dpu will have a different digit
    DPU_ASSERT(dpu_prepare_xfer(dpu, &input_data[dpunum*784*156]));
    dpunum++;
  }

  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "inputdata", 0, BUFFER_SIZE, DPU_XFER_DEFAULT));

 dpunum=0;
  DPU_FOREACH(set, dpu) {
    // each dpu will have a different digit
    DPU_ASSERT(dpu_prepare_xfer(dpu, &test_label[dpunum*156]));
    dpunum++;
  }

  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "labeldata", 0, LBUFFER_SIZE, DPU_XFER_DEFAULT));

  printf("Moved Data to MRAM \n");
  DPU_ASSERT(dpu_launch(set, DPU_ASYNCHRONOUS));
  DPU_ASSERT(dpu_sync(set));
  DPU_FOREACH(set, dpu) {
   DPU_ASSERT(dpu_log_read(dpu, stdout));
    DPU_ASSERT(dpu_copy_from(dpu, "total", 0, (uint8_t *)&count, sizeof(count)));
    total += count;
    printf("Total: %d\n", total);
  }

  DPU_ASSERT(dpu_free(set));
  free(input_data);
  return 0;
}
