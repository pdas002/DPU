#include <stdio.h>
#include <stdint.h>
#include <perfcounter.h>


int main() {

	int num = 3;
	float num2 = 3.145; 
	int result1;
	float result2;

   	perfcounter_config(COUNT_CYCLES, true);

	result1 = num*num;

	perfcounter_t run_time = perfcounter_get();
   	printf("Int Time: %lu\n", run_time);


	perfcounter_config(COUNT_CYCLES, true);

        result2 = num2*num2;

        run_time = perfcounter_get();
        printf("Float Time: %lu\n", run_time);  

	return 0;


}
