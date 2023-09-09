/**
 * @file mpi_sort.c
 * @brief 
 * parallel sample sort with MPI
 * 
 * P0 broadcast input array a to each process
 * such that processes fill buckets and sort locally
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
  int i, n, d, r, par, count;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  if (argc != 2){
    printf("mpiexec -n <nprocs> ./mpi_sort <input_filename>\n");
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

  /* broadcast array a */
  if (my_rank != 0)
    a = malloc(sizeof(int) * n);
  MPI_Bcast(a, n, MPI_INT, 0, MPI_COMM_WORLD);
  
  /* local splitter selection */
  int avg, length, start_pos; 
  avg = n / comm_sz;
  length = my_rank == comm_sz - 1 ? n - avg * my_rank : avg;
  start_pos = my_rank * avg;
  int *subarray = malloc(sizeof(int) * length);
  for (i = 0; i < length; i++){
    subarray[i] = a[start_pos+i];
  }
  qsort(subarray, length, sizeof(int), cmpfunc);

  int* local_splitters, *global_splitters, *final_splitters;
  local_splitters = malloc(sizeof(int) * (comm_sz-1));
  global_splitters = malloc(sizeof(int) * (comm_sz-1) * comm_sz);
  final_splitters = malloc(sizeof(int) * (comm_sz-1));

  d = length / comm_sz;
  r = length % comm_sz;
  for (i = 1, par =0; i < comm_sz; i++){
    par += (i < r ? d+1 : d); 
    local_splitters[i-1] = subarray[par];
  }

  /* broadcast local splitters by Allgather */
  MPI_Allgather(local_splitters, comm_sz-1, MPI_INT, global_splitters, (comm_sz-1) * comm_sz, MPI_INT, MPI_COMM_WORLD);

  /* final splitter selection */
  qsort(global_splitters, (comm_sz-1) * comm_sz, sizeof(int), cmpfunc);
  par = comm_sz/2;
  for (i = 0; i < comm_sz-1; i++){
    final_splitters[i] = global_splitters[par+i*comm_sz];
  }

  /* bucket sort */
  int bucket_low, bucket_high;
  bucket_low = my_rank == 0? -1 : final_splitters[my_rank-1];
  bucket_high = my_rank == comm_sz-1 ? MAX : final_splitters[my_rank];
  int my_bucket_size = n / comm_sz;
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
  qsort(my_bucket, count, sizeof(int), cmpfunc);
  
  /* gather bucket count in communicator */
  int* counts = NULL, *disps = NULL;
  if (my_rank == 0){
    counts = malloc(sizeof(int) * comm_sz);
    disps = malloc(sizeof(int) * comm_sz);
  }
  MPI_Gather(&count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (my_rank == 0){
    disps[0] = 0;
    for (i = 0; i < comm_sz; i++){
      disps[i] = disps[i-1] + counts[i-1];
    }
  }

  /* gather sorted buckets */
  MPI_Gatherv(my_bucket, count, MPI_INT, output, counts, disps, MPI_INT, 0, MPI_COMM_WORLD);

  /* correctness checking and timing output*/  
  double end = MPI_Wtime();
  if (my_rank == 0){
    checkSorting(n, output, oracle);
    printf("computed in %.8fs\n", end-start);
  }
  MPI_Finalize();
  return 0;
}