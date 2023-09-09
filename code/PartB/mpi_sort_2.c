/**
 * @file mpi_sort_2.c
 * @brief 
 * parallel sample sort with MPI
 * 
 * input array a is only available for P0
 * such that P0 fills buckets for all processes
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <string.h>
#include "../util/utils.h"

int cmpfunc(const void *a, const void *b) {
  return (*(int*)a - *(int*)b);
}
#define MAX __INT_MAX__

int main(int argc, char *argv[]){
  int my_rank, comm_sz;
  int* a = NULL, *oracle = NULL, *output = NULL;
  int n;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  if (argc != 2){
    printf("mpiexec -n <nprocs> ./mpi_sort_2 <input_filename>\n");
    exit(-1);
  }

  /* read from file */
  if (my_rank == 0){
    readSorting(argv[1], &n, &a, &oracle);
    if (n < (comm_sz * comm_sz)) {
      printf("sample sort: n > square(nprocs)\n");
      exit(-1);
    }
    output = malloc(sizeof(int)*n);
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  /* performance measurement */
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();
  
  /* block assignment by MPI_scatter */
  int avg = n / comm_sz;
  int length = my_rank == comm_sz - 1 ? n - avg * my_rank : avg;
  int *subarray = malloc(sizeof(int) * length);
  int *counts_send = NULL, *disps = NULL;
  if (my_rank == 0){
    counts_send = malloc(sizeof(int) * comm_sz);
    disps = malloc(sizeof(int) * comm_sz);
    for (int rank = 0; rank < comm_sz; rank++){
      counts_send[rank] = rank == comm_sz - 1? n - avg * rank : avg;
      disps[rank] = rank == 0? 0 : disps[rank-1] + counts_send[rank-1];
    }
  }
  MPI_Scatterv(a, counts_send, disps, MPI_INT, subarray, length, MPI_INT, 0, MPI_COMM_WORLD);

  /* local splitter selection */
  qsort(subarray, length, sizeof(int), cmpfunc);
  int i, j, d, r, par;
  int* local_splitters, *global_splitters = NULL, *final_splitters = NULL;
  local_splitters = malloc(sizeof(int) * (comm_sz-1));
  d = length / comm_sz;
  r = length % comm_sz;
  for (i = 1, par = 0; i < comm_sz; i++){
    par += (i < r ? d+1 : d); 
    local_splitters[i-1] = subarray[par];
  }
  free(subarray);

  /* P0 gathers local splitters */
  if (my_rank == 0){
    global_splitters = malloc(sizeof(int) * (comm_sz-1) * comm_sz);
    final_splitters = malloc(sizeof(int) * (comm_sz-1));
  }
  MPI_Gather(local_splitters, comm_sz-1, MPI_INT, global_splitters, comm_sz-1, MPI_INT, 0, MPI_COMM_WORLD);
  int* bucket_counts = malloc(sizeof(int) * comm_sz);
  int** buckets = NULL;
  
  /* final splitter selection and bucket assignment by P0 */
  if (my_rank == 0){
    qsort(global_splitters, (comm_sz-1) * comm_sz, sizeof(int), cmpfunc);
    par = comm_sz/2;
    for (i = 0; i < comm_sz-1; i++){
      final_splitters[i] = global_splitters[par+i*comm_sz];
    }
    int size = n/comm_sz;
    int* bucket_sizes = malloc(sizeof(int) * comm_sz);
    buckets = malloc (sizeof(int*) * comm_sz);
    bucket_sizes[0] = size;
    subarray = malloc(sizeof(int) * size);
    for (i = 1; i < comm_sz; i++){
      bucket_sizes[i] = size;
      buckets[i] = malloc(sizeof(int) *  size);
    }
    memset(bucket_counts, 0, sizeof(int) * comm_sz);
    // assign buckets
    int inLastBucket;
    for (i = 0; i < n; i++){
      inLastBucket = 1;
      for (j = 0; j < comm_sz-1; j++){
        if (a[i] < final_splitters[j]){
          if (j == 0){
            if (bucket_counts[j] == bucket_sizes[j]){
              bucket_sizes[j] *= 2;
              subarray = realloc(subarray, sizeof(int) * bucket_sizes[j]);
            }
            subarray[bucket_counts[j]++] = a[i];
          } else {
            if (bucket_counts[j] == bucket_sizes[j]){
              bucket_sizes[j] *= 2;
              buckets[j] = realloc(buckets[j], sizeof(int) * bucket_sizes[j]);
            }
            buckets[j][bucket_counts[j]++] = a[i];
          }
          inLastBucket = 0;
          break;
        } 
      }
      if (inLastBucket){
        if (bucket_counts[comm_sz-1] == bucket_sizes[comm_sz-1]){
          bucket_sizes[comm_sz-1] *= 2;
          buckets[comm_sz-1] = realloc(buckets[comm_sz-1], sizeof(int)*bucket_sizes[comm_sz-1]);
        }
        buckets[comm_sz-1][bucket_counts[comm_sz-1]++] = a[i];
      }
    }
    free(bucket_sizes);
  }
  /* broadcast bucket counts */
  MPI_Bcast(bucket_counts, comm_sz, MPI_INT, 0, MPI_COMM_WORLD);
  
  /* send buckets */
  if (my_rank == 0){
    MPI_Request requests[comm_sz];
    disps[0] = 0;
    for (i = 1; i < comm_sz; i++){
      disps[i] = disps[i-1] + bucket_counts[i-1];
      MPI_Isend(buckets[i], bucket_counts[i], MPI_INT, i, 0, MPI_COMM_WORLD, &requests[i]);
    }
    MPI_Waitall(comm_sz-1, &requests[1], MPI_STATUSES_IGNORE);
    for (i = 1; i < comm_sz; i++){
      free(buckets[i]);
    }
    free(buckets);
  } else {
    subarray = malloc(sizeof(int) * bucket_counts[my_rank]);
    MPI_Recv(subarray, bucket_counts[my_rank], MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  /* bucket sort */
  qsort(subarray, bucket_counts[my_rank], sizeof(int), cmpfunc);

  /* gather sorted buckets */
  MPI_Gatherv(subarray, bucket_counts[my_rank], MPI_INT, output, bucket_counts, disps, MPI_INT, 0, MPI_COMM_WORLD);

  /* correctness checking and timing output*/  
  double end = MPI_Wtime();
  if (my_rank == 0){
    checkSorting(n, output, oracle);
    printf("computed in %.8fs\n", end-start);
    free(counts_send);
    free(disps);
  }
  free(bucket_counts);
  MPI_Finalize();
  return 0;
}