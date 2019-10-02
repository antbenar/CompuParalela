firstOE: omp_odd_even1.c
	gcc -g -Wall -fopenmp -o firstOE omp_odd_even1.c

secondOE: omp_odd_even2.c
	gcc -g -Wall -fopenmp -I. -o secondOE omp_odd_even2.c

criticalPi: omp_piCritical.c
	gcc -g -Wall -fopenmp -o criticalPi omp_piCritical.c -lm   

atomicPi: omp_piAtomic.c
	gcc -g -Wall -fopenmp -o atomicPi omp_piAtomic.c -lm   

tokenizer: omp_tokenize_r.c
	gcc -g -Wall -fopenmp -o tokenizer omp_tokenize_r.c