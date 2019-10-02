#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h> 

int main(int argc, char* argv[]) {
   long long n, i;
   int thread_count;
   double factor;
   double sum = 0.0;
   double start, finish;
   
   //obtener num threads y largo de la serie
   if (argc != 3) fprintf(stderr, "how to use: %s <thread_count> <n>\n", argv[0]);
   thread_count = strtol(argv[1], NULL, 10);
   n = strtoll(argv[2], NULL, 10);
   
   start = omp_get_wtime();//-----------------------START
   
#  pragma omp parallel for num_threads(thread_count) \ 
	  private(factor)
   for (i = 0; i < n; i++) {
	   factor = (i % 2 == 0) ? 1.0 : -1.0; 
# pragma omp critical
	   sum += factor/(2*i+1);
   }
   
   finish = omp_get_wtime();//-----------------------FINISH
   
   printf("Elapsed time = %e seconds\n", finish - start);
   
   return 0;
}  

