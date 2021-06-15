#include <stdio.h>
#include <stdint.h>
#include <perfcounter.h>


int main() {

	int num = 100;
	int result1;
	int result2;

   	perfcounter_config(COUNT_CYCLES, true);

	result1 = num+num;

	perfcounter_t run_time = perfcounter_get();
   	printf("Add Time: %lu\n", run_time);


	perfcounter_config(COUNT_CYCLES, true);

        result2 = num*num;

        run_time = perfcounter_get();
        printf("Multiplication Time: %lu\n", run_time);  

	return 0;


}
