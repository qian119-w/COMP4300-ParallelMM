/**
 * @file pth_sort.c
 * @brief 
 * parallel sample sort with Pthread
 * simulate MPI alltoall broadcast by conditional variable
 * 
 * Use rand() to implement random bucket assignment for splitter selection
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <semaphore.h>
#include <math.h>
#include "../util/utils.h"
#include "../util/timer.h"

typedef struct {
  int* addr;
  int ready;
  pthread_mutex_t mutex;
  pthread_cond_t proceed; 
} comm_node;

typedef struct {
  int N;
  comm_node* nodes;
  int* bucket_counts; 
  pthread_barrier_t b;
} comm_t;

typedef struct {
  int rank;
  int* subarray;
  int length;
} thd_arg;

void initComm(comm_t *c, int n);
void destComm(comm_t *c);
void alltoallBcast(comm_t *c, int my_rank, int* send_buf, int* recv_buf, int count);
int cmpfunc(const void *a, const void *b) {
  return (*(int*)a - *(int*)b);
}
#define MAX __INT_MAX__

int n, nthds;
int* a, *oracle, *output;
comm_t new_comm;

void* thread_func(void* arg){
  thd_arg* my_arg = (thd_arg*) arg;
  int* local_splitters, *global_splitters, *final_splitters;
  int i, d, r, par;
  local_splitters = malloc(sizeof(int) * (nthds-1));
  global_splitters = malloc(sizeof(int) * (nthds-1) * nthds);
  final_splitters = malloc(sizeof(int) * (nthds-1));
  /* local splitter selection */
  qsort(my_arg->subarray, my_arg->length, sizeof(int), cmpfunc);
  d = my_arg->length / nthds;
  r = my_arg->length % nthds;
  for (i = 1, par = 0; i < nthds; i++){
    par += (i < r ? d+1 : d); 
    local_splitters[i-1] = my_arg->subarray[par];
  }
  free(my_arg->subarray);

  /* broadcast local splitters */
  alltoallBcast(&new_comm, my_arg->rank, local_splitters, global_splitters, nthds-1);

  /* final splitter selection */
  qsort(global_splitters, (nthds-1) * nthds, sizeof(int), cmpfunc);
  d = nthds/2;
  for (i = 0; i < nthds-1; i++){
    final_splitters[i] = global_splitters[d+i*nthds];
  }

  /* bucket sort */
  int bucket_low, bucket_high, count;
  bucket_low = my_arg->rank == 0? -1 : final_splitters[my_arg->rank-1];
  bucket_high = my_arg->rank == nthds-1 ? MAX : final_splitters[my_arg->rank];
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
  new_comm.bucket_counts[my_arg->rank] = count;
  pthread_barrier_wait(&new_comm.b);

  /* write result to correct position */
  int start_pos = 0;
  for (i = 0; i< my_arg->rank; i++){
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
    printf("./pth_sort <num_threads> <input_filename>\n");
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
  thd_arg* args = malloc(sizeof(thd_arg) * nthds);
  initComm(&new_comm, nthds);

  /* random bucket assignment for splitter selection */
  int rank, length, count, idx;
  int* subarray, *hasAssigned;
  hasAssigned = malloc(sizeof(int) * n);
  memset(hasAssigned, 0, sizeof(int) * n); 
  srand(time(NULL));
  int d, r;
  d = n / nthds; r = n % nthds;
  for (rank = 0; rank < nthds; rank++){
    count = 0;
    length = rank < r ? d + 1 : d;
    subarray = malloc(sizeof(int) * length);
    while (count < length){
      while (hasAssigned[idx = rand() % n]);
      subarray[count++] = a[idx];
      hasAssigned[idx] = 1;
    }
    args[rank] = (thd_arg) {.rank = rank, .subarray = subarray, .length = length};
  }
  free(hasAssigned);

  /* performance measurement */
  double start, end;
  GET_TIME(start);

  /* create threads */
  for (rank = 0; rank < nthds; rank++){
    pthread_create(&tid[rank], NULL, thread_func, (void *) &args[rank]);
  }

  /* join threads */
  for (rank = 0; rank < nthds; rank++){
    pthread_join(tid[rank], NULL);
  }

  /* correctness checking and timing output*/
  GET_TIME(end);
  checkSorting(n, output, oracle);
  printf("computed in %.8fs\n", end-start);

  /* cleanup */
  destComm(&new_comm);
  free(a); free(oracle); free(output);
  free(tid); free(args);
  return 0;
}

/* ------------------------- */
void initComm(comm_t *c, int n){
  c->N = n;
  c->nodes = malloc(sizeof(comm_node) * n);
  c->bucket_counts = malloc(sizeof(int) * n);
  pthread_barrier_init(&c->b, NULL, n);
  for (int i = 0; i < c->N; i++){
    c->nodes[i].ready = 0;
    pthread_mutex_init(&c->nodes[i].mutex, NULL);
    pthread_cond_init(&c->nodes[i].proceed, NULL);
  }
}

void destComm(comm_t *c){
  for (int i = 0; i < c->N; i++){
    pthread_mutex_destroy(&c->nodes[i].mutex);
    pthread_cond_destroy(&c->nodes[i].proceed);
  }
  pthread_barrier_destroy(&c->b);
  free(c->nodes);
  free(c->bucket_counts);
}
/* ------------------------- */
/**
 * @brief 
 * alltoall broadcast using condition variables
 */
void alltoallBcast(comm_t *c, int my_rank, int* send_buf, int* recv_buf, int count){
  pthread_mutex_lock(&c->nodes[my_rank].mutex);
  c->nodes[my_rank].addr = send_buf;
  c->nodes[my_rank].ready = c->N;
  pthread_cond_broadcast(&c->nodes[my_rank].proceed);
  pthread_mutex_unlock(&c->nodes[my_rank].mutex);

  int i, j, recv_rank, inc = 0;
  for (i = 0; i < c->N; i++){
    // cyclic receive from (my_rank + i)
    recv_rank = (my_rank + i) % c-> N;
    pthread_mutex_lock(&c->nodes[recv_rank].mutex);
    if (c->nodes[recv_rank].ready <= 0){
      while (pthread_cond_wait(&c->nodes[recv_rank].proceed, &c->nodes[recv_rank].mutex) != 0);
    }
    c->nodes[recv_rank].ready--;
    int* inter_recv = c->nodes[recv_rank].addr;
    for (j = 0; j < count; j++){
      recv_buf[inc++] = inter_recv[j];
    }
    pthread_mutex_unlock(&c->nodes[recv_rank].mutex);
  }
  pthread_barrier_wait(&c->b);
}