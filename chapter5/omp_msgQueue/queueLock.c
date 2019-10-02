/* File:     omp_msglk.c
 *
 * Purpose:  Simulate message-passing using OpenMP.  This version uses
 *           an atomic directive and OpenMP locks to protect critical
 *           sections.
 *
 * Compile:  gcc -g -Wall -fopenmp -o omp_msglk omp_msglk.c queue_lk.c
 *           needs queue_lk.h
 * Usage:    ./omp_msglk <number of threads> <number of messages each 
 *                  thread sends>
 *
 * Input:    None
 * Output:   Source, destination and contents of each message received.
 *
 *
 * Notes:
 * 1.  DEBUG flag for more verbose output
 * 2.  This version uses locks to control access to the message queues.
 *
 * IPP:      Section 5.8.9 (pp. 248 and ff.)
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

const int MAX_MSG = 10000;


//------------------------------QUEUE




struct queue_node_s {
	int src;
	int mesg;
	struct queue_node_s* next_p;
};

struct queue_s{
	omp_lock_t lock;
	int enqueued;
	int dequeued;
	struct queue_node_s* front_p;
	struct queue_node_s* tail_p;
};



struct queue_s* Allocate_queue() {
	struct queue_s* q_p = malloc(sizeof(struct queue_s));
	q_p->enqueued = q_p->dequeued = 0;
	q_p->front_p = NULL;
	q_p->tail_p = NULL;
	omp_init_lock(&q_p->lock);
	return q_p;
}  /* Allocate_queue */

/* Frees nodes in queue:  leaves queue struct allocated and lock
* initialized */
void Free_queue(struct queue_s* q_p) {
	struct queue_node_s* curr_p = q_p->front_p;
	struct queue_node_s* temp_p;
	
	while(curr_p != NULL) {
		temp_p = curr_p;
		curr_p = curr_p->next_p;
		free(temp_p);
	}
	q_p->enqueued = q_p->dequeued = 0;
	q_p->front_p = q_p->tail_p = NULL;
}   /* Free_queue */

void Print_queue(struct queue_s* q_p) {
	struct queue_node_s* curr_p = q_p->front_p;
	
	printf("queue = \n");
	while(curr_p != NULL) {
		printf("   src = %d, mesg = %d\n", curr_p->src, curr_p->mesg);
		curr_p = curr_p->next_p;
	}
	printf("enqueued = %d, dequeued = %d\n", q_p->enqueued, q_p->dequeued);
	printf("\n");
}  /*  Print_Queue */

void Enqueue(struct queue_s* q_p, int src, int mesg) {
	struct queue_node_s* n_p = malloc(sizeof(struct queue_node_s));
	n_p->src = src;
	n_p->mesg = mesg;
	n_p->next_p = NULL;
	if (q_p->tail_p == NULL) { /* Empty Queue */
		q_p->front_p = n_p;
		q_p->tail_p = n_p;
	} else {
		q_p->tail_p->next_p = n_p;
		q_p->tail_p = n_p;
	}
	q_p->enqueued++;
}  /* Enqueue */

int Dequeue(struct queue_s* q_p, int* src_p, int* mesg_p) {
	struct queue_node_s* temp_p;
	
	if (q_p->front_p == NULL) return 0;
	*src_p = q_p->front_p->src;
	*mesg_p = q_p->front_p->mesg;
	temp_p = q_p->front_p;
	if (q_p->front_p == q_p->tail_p)  /* One node in list */
		q_p->front_p = q_p->tail_p = NULL;
	else
		q_p->front_p = temp_p->next_p;
	free(temp_p);
	q_p->dequeued++;
	return 1;
}  /* Dequeue */

int Search(struct queue_s* q_p, int mesg, int* src_p) {
	struct queue_node_s* curr_p = q_p->front_p;
	
	while (curr_p != NULL)
		if (curr_p->mesg == mesg) {
			*src_p = curr_p->src;
			return 1;
	} else {
			curr_p = curr_p->next_p;
		}
		return 0;
		
}  /* Search */





//------------------------------ END QUEUE


void Usage(char* prog_name);
void Send_msg(struct queue_s* msg_queues[], int my_rank, 
      int thread_count, int msg_number);
void Try_receive(struct queue_s* q_p, int my_rank);
int Done(struct queue_s* q_p, int done_sending, int thread_count);

/*-------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
   int thread_count;
   int send_max;
   struct queue_s** msg_queues;
   int done_sending = 0;
   double start, finish;
   
   if (argc != 3) Usage(argv[0]);
   thread_count = strtol(argv[1], NULL, 10);
   send_max = strtol(argv[2], NULL, 10);
   if (thread_count <= 0 || send_max < 0) Usage(argv[0]);

   msg_queues = malloc(thread_count*sizeof(struct queue_node_s*));

   
   start = omp_get_wtime();//-----------------------START
   
#  pragma omp parallel num_threads(thread_count) \
      default(none) shared(thread_count, send_max, msg_queues, done_sending)
   {
      int my_rank = omp_get_thread_num();
      int msg_number;
      srandom(my_rank);
      msg_queues[my_rank] = Allocate_queue();

#     pragma omp barrier /* Don't let any threads send messages  */
                         /* until all queues are constructed     */

      for (msg_number = 0; msg_number < send_max; msg_number++) {
         Send_msg(msg_queues, my_rank, thread_count, msg_number);
         Try_receive(msg_queues[my_rank], my_rank);
      }
#     pragma omp atomic
      done_sending++;

      while (!Done(msg_queues[my_rank], done_sending, thread_count))
         Try_receive(msg_queues[my_rank], my_rank);

      /* My queue is empty, and everyone is done sending             */
      /* So my queue won't be accessed again, and it's OK to free it */
      Free_queue(msg_queues[my_rank]);
      free(msg_queues[my_rank]);
   }  /* omp parallel */

   finish = omp_get_wtime();//-----------------------FINISH
   printf("Elapsed time = %e seconds\n", finish - start);
   
   
   free(msg_queues);
   return 0;
}  /* main */

/*-------------------------------------------------------------------*/
void Usage(char *prog_name) {
   fprintf(stderr, "usage: %s <number of threads> <number of messages>\n",
         prog_name);
   fprintf(stderr, "   number of messages = number sent by each thread\n");
   exit(0);
}  /* Usage */

/*-------------------------------------------------------------------*/
void Send_msg(struct queue_s* msg_queues[], int my_rank, 
      int thread_count, int msg_number) {
// int mesg = random() % MAX_MSG;
   int mesg = -msg_number;
   int dest = random() % thread_count;
   struct queue_s* q_p = msg_queues[dest];
   omp_set_lock(&q_p->lock);
   Enqueue(q_p, my_rank, mesg);
   omp_unset_lock(&q_p->lock);
}  /* Send_msg */

/*-------------------------------------------------------------------*/
void Try_receive(struct queue_s* q_p, int my_rank) {
   int src, mesg;
   int queue_size = q_p->enqueued - q_p->dequeued;

   if (queue_size == 0) return;
   else if (queue_size == 1) {
      omp_set_lock(&q_p->lock);
      Dequeue(q_p, &src, &mesg);  
      omp_unset_lock(&q_p->lock);
   } else
      Dequeue(q_p, &src, &mesg);
   //printf("Thread %d > received %d from %d\n", my_rank, mesg, src);
}   /* Try_receive */

/*-------------------------------------------------------------------*/
int Done(struct queue_s* q_p, int done_sending, int thread_count) {
   int queue_size = q_p->enqueued - q_p->dequeued;
   if (queue_size == 0 && done_sending == thread_count)
      return 1;
   else 
      return 0;
}   /* Done */
