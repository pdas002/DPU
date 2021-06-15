#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include "binarymnistlarge.h"

#ifndef DPU_BINARY
#define DPU_BINARY "./multdpubin"
#endif

#define BUFFER_SIZE 15288
#define LBUFFER_SIZE 624 

int main(void) {
  uint32_t count;
  int32_t total =0;
  struct dpu_set_t set, dpu;

  DPU_ASSERT(dpu_alloc(64, NULL, &set));
  DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));


  printf("Moving input data to MRAM \n");
  int dpunum = 0;
  DPU_FOREACH(set, dpu) {
    // each dpu will have a different digit
    DPU_ASSERT(dpu_prepare_xfer(dpu, &test_data[15288*dpunum]));
    dpunum++;
  }

  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "inputdata", 0, BUFFER_SIZE, DPU_XFER_DEFAULT));
 dpunum=0;
  DPU_FOREACH(set, dpu) {
    // each dpu will have a different digit
    DPU_ASSERT(dpu_prepare_xfer(dpu, &test_labels[dpunum*156]));
    dpunum++;
  }

  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "labeldata", 0, LBUFFER_SIZE, DPU_XFER_DEFAULT));
  printf("Moved Data to MRAM \n");
  DPU_ASSERT(dpu_launch(set, DPU_ASYNCHRONOUS));
  DPU_ASSERT(dpu_sync(set));
  DPU_FOREACH(set, dpu) {
//   DPU_ASSERT(dpu_log_read(dpu, stdout));
    DPU_ASSERT(dpu_copy_from(dpu, "total", 0, (uint8_t *)&count, sizeof(count)));
    total += count;
    printf("Total: %d\n", total);
  }

  DPU_ASSERT(dpu_free(set));

  return 0;
}
