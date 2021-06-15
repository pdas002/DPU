#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <alloc.h>
#include <mram.h>
#include "aes.h"


//Size definitions in bytes

//The maximum a mram_read can do
#define MAX_READ 2048

//Size of cyphertext (multiple of 8 and 2048)
#define BUFFER_SIZE 4096

//Size of key and IV
#define KEY_SIZE 32
#define IV_SIZE 16

//Number of loops to read buffer
#define COUNT (BUFFER_SIZE/MAX_READ)

//MRAM arrays to hold buffer, key and IV
__mram_noinit uint8_t buf[BUFFER_SIZE];
__mram_noinit uint8_t key[KEY_SIZE];
__mram_noinit uint8_t iv[IV_SIZE];

//The output buffer
__mram_noinit uint8_t result[BUFFER_SIZE];


int main(){
        struct AES_ctx ctx;
	uint8_t wkey[KEY_SIZE];
        uint8_t wiv[IV_SIZE];
	uint8_t wbuf[MAX_READ];

	//Read in the MRAM data to the WRAM
	mram_read(key, wkey, KEY_SIZE);
	mram_read(iv, wiv, IV_SIZE);
	int loops = COUNT;

	//If the size of buffer is < 2048
	#if COUNT == 0
		//Read the MRAM buffer to local buff
		mram_read(buf, wbuf, BUFFER_SIZE);
		//Start the AES algo
                AES_init_ctx_iv(&ctx, wkey, wiv);
 		//Decrypt
                AES_CBC_decrypt_buffer(&ctx, wbuf, BUFFER_SIZE);
             	//Copy back to MRAM
		memcpy(result, wbuf, BUFFER_SIZE);
	#else
		//Start the context
		AES_init_ctx_iv(&ctx, wkey, wiv);
		for(int i = 0; i < loops; i++){
			//Copy part of buffer to local buffer
			mram_read(buf+i*MAX_READ, wbuf, MAX_READ);
       			//Decrypt the buffer
	 		AES_CBC_decrypt_buffer(&ctx, wbuf, MAX_READ);
			//Write result to MRAM
			mram_write(wbuf, result+i*MAX_READ, MAX_READ);
		}
	#endif
        return 0; 
}

