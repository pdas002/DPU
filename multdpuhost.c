#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>
#include "mnist_data.h"

#ifndef DPU_BINARY
#define DPU_BINARY "./multdpualex"
#endif

#define BUFFER_SIZE 3136

int main(void) {
  struct dpu_set_t set, dpu;

  DPU_ASSERT(dpu_alloc(20, NULL, &set));
  DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));


  printf("Moving input data to MRAM \n");
  int dpunum = 0;
  DPU_FOREACH(set, dpu) {
    // each dpu will have a different digit
    DPU_ASSERT(dpu_prepare_xfer(dpu, &train_data[dpunum*784]));
    dpunum++;
  }

  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "inputdata", 0, BUFFER_SIZE, DPU_XFER_DEFAULT));

  printf("Moved Data to MRAM \n");
  DPU_ASSERT(dpu_launch(set, DPU_ASYNCHRONOUS));
  DPU_ASSERT(dpu_sync(set));
  DPU_FOREACH(set, dpu) {
    DPU_ASSERT(dpu_log_read(dpu, stdout));
  }

  DPU_ASSERT(dpu_free(set));

  return 0;
}
