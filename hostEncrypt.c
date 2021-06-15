#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <stdio.h>

#ifndef DPU_BINARY
#define DPU_BINARY "./dpuEncrypt"
#endif

#define KEY_SIZE 32
#define IV_SIZE 16
#define BUFFER_SIZE 4096

#define str(s) #s
#define INPUT str(data)
#define OUTPUT str(data0)

uint8_t key[KEY_SIZE] = { 0x60, 0x3d, 0xeb, 0x10, 0x15, 0xca, 0x71, 0xbe, 0x2b, 0x73, 0xae, 0xf0, 0x85, 0x7d, 0x77, 0x81,
                      0x1f, 0x35, 0x2c, 0x07, 0x3b, 0x61, 0x08, 0xd7, 0x2d, 0x98, 0x10, 0xa3, 0x09, 0x14, 0xdf, 0xf4 };
uint8_t iv[IV_SIZE]  = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f };



//Fills buffer with datat from file
int fillBuffer(uint8_t** buf){
        int read, fSize;
        FILE *fp = fopen(INPUT, "rb");
	if(fp == NULL){
		printf("Could not open file for reading\n");
		return -1;
	}

        //get Filesize 
        fseek(fp, 0, SEEK_END);
        fSize = ftell(fp);
        rewind(fp);

        //Allocate memory for buffer
        *buf = malloc(fSize);
	if(*buf == NULL){
		printf("OOM\n");
		return -1;
	}
        //Fill Buffer
        read = fread(*buf,1,fSize,fp);
	if(read != fSize){
		printf("Did not read entire file!\n");
		return -1;
	}
	fclose(fp);
        return 0;

}

//Writes buffer contents to file
int writeBuffer(uint8_t* buf, int len){
        int w;
        FILE *fp = fopen(OUTPUT, "wb+");
	if(fp == NULL){
		printf("Could not open file for writing!\n");
		return -1;
	}

        //write Buffer
        w = fwrite(buf,1,len,fp);
	if(w != len){
		printf("Could not write everything to file!\n");
		return -1;
	}
	fclose(fp);
        return 0;
}



int main(void) {
  ////////////////////////////////Data INIT/////////////////////////
  int len;
  uint8_t* buf = NULL;
  if(fillBuffer(&buf) != 0){
	free(buf);
	return -1;
  } 


  /////////////////////////////////DPU INIT///////////////////////

  struct dpu_set_t set, dpu;
  DPU_ASSERT(dpu_alloc(1, NULL, &set));
  DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));

  ///////////////////////////Data transfer////////////////////////////////////
  printf("Moving data to MRAM... \n");
  int dpunum = 0;
  DPU_FOREACH(set, dpu) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, buf));
    dpunum++;
  }

  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "buf", 0, BUFFER_SIZE, DPU_XFER_DEFAULT));
  dpunum=0;
  DPU_FOREACH(set, dpu) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, key));
    dpunum++;
  }

  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "key", 0, KEY_SIZE, DPU_XFER_DEFAULT));
  dpunum=0;
  DPU_FOREACH(set, dpu) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, iv));
    dpunum++;
  }

  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "iv", 0, IV_SIZE, DPU_XFER_DEFAULT));
  
  printf("Moved Data to MRAM...\n Starting DPU launch...\n");

  ///////////////////////////DPU launch///////////////////////////////////////
  DPU_ASSERT(dpu_launch(set, DPU_ASYNCHRONOUS));
  DPU_ASSERT(dpu_sync(set));
  DPU_FOREACH(set, dpu) {
//    DPU_ASSERT(dpu_log_read(dpu, stdout));
        DPU_ASSERT(dpu_copy_from(dpu, "result", 0,buf, BUFFER_SIZE));
   	printf("Got result...\n");
	if(writeBuffer(buf, BUFFER_SIZE) != 0){
		printf("Error writing!\n");
	}else{
 		printf("Wrote result to file...\n");
	}
  }

  DPU_ASSERT(dpu_free(set));
  free(buf);
  return 0;
}

