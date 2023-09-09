/**
 * @file pth_sort_2.c
 * @brief 
 * parallel sample sort with Pthread
 * simulate MPI alltoall broadcast by semaphore
 * 
 * cyclic assignment to simulate random bucket assignment for splitter selection
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <semaphore.h>
#include <math.h>
#include "../util/utils.h"
#include "../util/timer.h"

typedef struct comm_node {
  int* buf_addr;
  sem_t ready;
  sem_t wait;
} comm_node;

typedef struct comm_t {
  int N;
  comm_node* nodes;
  int* bucket_counts; 
  pthread_barrier_t b;
} comm_t;

void initComm(comm_t *c, int n);
void destComm(comm_t *c);
void alltoallBcast(comm_t* c, int my_rank, int* send_buf, int* recv_buf, int count);
int cmpfunc(const void *a, const void *b) {
  return (*(int*)a - *(int*)b);
}
#define MAX __INT_MAX__

int *a, *oracle, *output;
int n, nthds;
comm_t new_comm;

void* thread_func(void* arg){
  long my_rank = (long) arg;
  int i, length, count, d, r, par;
  length = n / nthds;
  int* subarray = malloc(sizeof(int) * length);
  /* cyclic assignment for splitter selection (pseudo random) */
  for (i = my_rank, count = 0; i < n; i+=nthds){
    if (count == length){
      length *= 2;
      subarray = realloc(subarray, sizeof(int) * length);
    }
    subarray[count++] = a[i];
  }
  qsort(subarray, count, sizeof(int), cmpfunc);

  int* local_splitters, *global_splitters, *final_splitters;
  local_splitters = malloc(sizeof(int) * (nthds-1));
  global_splitters = malloc(sizeof(int) * (nthds-1) * nthds);
  final_splitters = malloc(sizeof(int) * (nthds-1));
  /* local splitter selection */
  d = count / nthds;
  r = count % nthds;
  for (i = 1, par = 0; i < nthds; i++){
    par += (i < r ? d+1 : d); 
    local_splitters[i-1] = subarray[par];
  }
  free(subarray);

  /* broadcast local splitters */
  alltoallBcast(&new_comm, my_rank, local_splitters, global_splitters, nthds-1);

  /* final splitter selection */
  qsort(global_splitters, (nthds-1) * nthds, sizeof(int), cmpfunc);
  d = nthds/2;
  for (i = 0; i < nthds-1; i++){
    final_splitters[i] = global_splitters[d+i*nthds];
  }

  /* my bucket */
  int bucket_low, bucket_high;
  bucket_low = my_rank == 0? -1 : final_splitters[my_rank-1];
  bucket_high = my_rank == nthds-1 ? MAX : final_splitters[my_rank];
  int my_bucket_size = n / nthds;
  int* my_bucket = malloc(sizeof(int) * my_bucket_size);
  for (i = 0, count = 0; i< n; i++){
    if (a[i] > bucket_low && a[i] <= bucket_high){
      if (count == my_bucket_size){
        my_bucket_size *=2;
        my_bucket = realloc(my_bucket, sizeof(int)*my_bucket_size);
      }
      my_bucket[count++] = a[i];
    }
  }
  /* broadcast bucket count in communicator */
  new_comm.bucket_counts[my_rank] = count;
  pthread_barrier_wait(&new_comm.b);

  /* write result to correct position */
  int start_pos = 0;
  for (i = 0; i< my_rank; i++){
    start_pos += new_comm.bucket_counts[i];
  }
  qsort(my_bucket, count, sizeof(int), cmpfunc);
  for (i = 0; i < count; i++){
    output[start_pos+i] = my_bucket[i];
  }

  free(local_splitters);
  free(global_splitters);
  free(final_splitters);
  free(my_bucket);
  return NULL;
}


int main(int argc, char* argv[]){
  if (argc != 3){
    printf("./pth_sort_2 <num_threads> <input_filename>\n");
    exit(-1);
  }
  /* read from file */
  readSorting(argv[2], &n, &a, &oracle);

  nthds = atoi(argv[1]);
  if (n < (nthds*nthds)) {
    printf("sample sort: n > square(nthds)\n");
    free(a); free(oracle);
    exit(-1);
  }

  output = malloc(sizeof(int)*n);
  pthread_t* tid = malloc(sizeof(pthread_t) * nthds);
  initComm(&new_comm, nthds);

  /* performance measurement */
  double start, end;
  GET_TIME(start);

  /* create threads */
  for(long rank = 0; rank < nthds; rank++){
    pthread_create(&tid[rank], NULL, thread_func, (void *) rank);
  }

  /* join threads */
  for (int rank = 0; rank < nthds; rank++){
    pthread_join(tid[rank], NULL);
  }

  /* correctness checking and timing output*/
  GET_TIME(end);
  checkSorting(n, output, oracle);
  printf("computed in %.8fs\n", end-start);

  /* cleanup */
  destComm(&new_comm);
  free(a); free(oracle); free(output);
  free(tid);
  return 0;
}

/* ------------------------- */
void initComm(comm_t *c, int n){
  c->N = n;
  c->nodes = malloc(sizeof(comm_node) * n);
  c->bucket_counts = malloc(sizeof(int) * n);
  pthread_barrier_init(&c->b, NULL, n);
  for (int i = 0; i < n; i++){
    sem_init(&c->nodes[i].ready, 0, 0);
    sem_init(&c->nodes[i].wait, 0, 0);
  }
}

void destComm(comm_t *c){
  for (int i = 0; i < c->N; i++){
    sem_destroy(&c->nodes[i].ready);
    sem_destroy(&c->nodes[i].wait);
  }
  pthread_barrier_destroy(&c->b);
  free(c->nodes);
  free(c->bucket_counts);
}
/* ------------------------- */

/**
 * @brief 
 * alltoall broadcast using semaphores
 */
void alltoallBcast(comm_t* c, int my_rank, int* send_buf, int* recv_buf, int count){
  int recv_rank;
  recv_rank = (my_rank + c->N - 1) % c->N;
  int* inter_buf = send_buf;
  int round, i, inc;
  for (i = 0, inc = 0; i < count; i++, inc++){
    recv_buf[inc] = send_buf[i];
  }

  for (round = 0; round < c->N - 1; round++){
    // send to (rank+1)
    c->nodes[my_rank].buf_addr = inter_buf;
    sem_post(&c->nodes[my_rank].ready);
    // recv from (rank-1)
    sem_wait(&c->nodes[recv_rank].ready);
    inter_buf = c->nodes[recv_rank].buf_addr;
    for (i = 0; i < count; i++){
      recv_buf[inc++] = inter_buf[i];
    }
    // finish recv from (rank-1)
    sem_post(&c->nodes[recv_rank].wait);
    // wait for (rank+1) to finish recv
    sem_wait(&c->nodes[my_rank].wait);
  }
  pthread_barrier_wait(&c->b);
}