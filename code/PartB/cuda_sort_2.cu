/**
 * @file cuda_sort.c
 * @brief 
 * parallel sample sort with CUDA
 * 
 * multiple thread blocks
 * use multiple kernels to achieve synchronisation between thread blocks
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../util/utils.h"
#include "../util/timer.h"
#define MAX __INT_MAX__
__host__ void readSorting(char *filename, int* n, int** a, int** oracle);
__host__ void checkSorting(int n, int* a, int* oracle);
__device__ int* realloc(int* old_a, int length, int new_length);
__host__ __device__ void swap(int* a, int* b);
__host__ __device__ int partition(int* a, int low, int high);
__host__ __device__ void qsort(int* arr, int l, int h);
__device__ int** buckets;

/* kernel : local splitter selection */
__global__ void splitter_selection(int* a, int* np, int* shared){
  int nthds = gridDim.x * blockDim.x;
  int my_rank = blockDim.x * blockIdx.x + threadIdx.x;
  int n = *np;
  int* global_splitters = &shared[0];
  if (my_rank == 0){
    buckets = (int**) malloc(sizeof(int*) * nthds);
  }
  int length = n / nthds;
  int* subarray = (int*) malloc(sizeof(int)*length);
  int i, count, d, r, par;
  /* cyclic assignment for splitter selection (pseudo random) */
  for (i = my_rank, count = 0; i < n; i+=nthds){
    if (count == length){
      subarray = realloc(subarray, length, length*2);
      length *= 2;
    }
    subarray[count++] = a[i];
  }
  qsort(subarray, 0, count-1);

  /* local splitter selection */
  int* local_splitters = (int*) malloc(sizeof(int) * (nthds-1));
  d = count / nthds;
  r = count % nthds;
  for (i = 1, par = 0; i < nthds; i++){
    par += (i < r ? d+1 : d); 
    local_splitters[i-1] = subarray[par];
  }
  free(subarray);

  /* broadcast local splitters in shared memory */
  int start_pos = (nthds-1)*my_rank;
  for (i = 0; i < nthds-1; i++){
    global_splitters[start_pos+i] = local_splitters[i];
  }
  free(local_splitters);
}

/* kernel : bucket sort */
__global__ void bucket_sort(int* a, int* np, int* shared){
  int nthds = gridDim.x * blockDim.x;
  int my_rank = blockDim.x * blockIdx.x + threadIdx.x;
  int n = *np;
  int* final_splitters = &shared[nthds*(nthds-1)];
  int* bucket_counts = &shared[nthds*nthds];

  int bucket_low, bucket_high, i, count;
  bucket_low = my_rank == 0? -1 : final_splitters[my_rank-1];
  bucket_high = my_rank == nthds-1 ? MAX : final_splitters[my_rank];
  int my_bucket_size = n / nthds;
  int* my_bucket = (int*) malloc(sizeof(int) * my_bucket_size);
  for (i = 0, count = 0; i< n; i++){
    if (a[i] > bucket_low && a[i] <= bucket_high){
      if (count == my_bucket_size){
        my_bucket = realloc(my_bucket, my_bucket_size, my_bucket_size*2);
        my_bucket_size *=2;
      }
      my_bucket[count++] = a[i];
    }
  }
  /* broadcast bucket count in shared memory */
  bucket_counts[my_rank] = count;
  qsort(my_bucket, 0, count-1);
  buckets[my_rank] = my_bucket;
}

/* kernel : write result to correct position */
__global__ void out(int* output, int* shared){
  int nthds = gridDim.x * blockDim.x;
  int my_rank = blockDim.x * blockIdx.x + threadIdx.x;
  int* bucket_counts = &shared[nthds*nthds];
  int* my_bucket = buckets[my_rank];
  int i, start_pos;
  for (i = 0, start_pos = 0; i< my_rank; i++){
    start_pos += bucket_counts[i];
  }
  for (i = 0; i < bucket_counts[my_rank]; i++){
    output[start_pos+i] = my_bucket[i];
  }
  free(my_bucket);
}


int main(int argc, char* argv[]){
  int n, num_blk, thd_per_blk, nthds;
  int *a, *oracle, *output, *shared;
  if (argc != 4){
    printf("./cuda_sort_2  <num_block> <num_threads_per_block> <input_filename>\n");
    exit(-1);
  }
  /* read from file */
  readSorting(argv[3], &n, &a, &oracle);

  num_blk = atoi(argv[1]);
  thd_per_blk = atoi(argv[2]);
  nthds = num_blk * thd_per_blk;
  if (n < (nthds*nthds)) {
    printf("sample sort: n > square(nthds)\n");
    exit(-1);
  }
  output = (int*) malloc(sizeof(int)*n);
  shared = (int*) malloc(sizeof(int)*nthds*(nthds+1));

  /* host and device memory allocation */
  int* a_device, *out_device, *n_device, *shared_device;
  cudaMalloc(&a_device, sizeof(int)*n);
  cudaMalloc(&out_device, sizeof(int)*n);
  cudaMalloc(&n_device, sizeof(int));
  cudaMalloc(&shared_device, sizeof(int)*nthds*(nthds+1));

  cudaMemcpy(a_device, a, sizeof(int)*n, cudaMemcpyHostToDevice);
  cudaMemcpy(n_device, &n, sizeof(int), cudaMemcpyHostToDevice);

  cudaDeviceSetLimit(cudaLimitMallocHeapSize, sizeof(int)*n*10);

  /* performance measurement */
  double start, end;
  GET_TIME(start);

  /* kernel : local splitter selection */
  splitter_selection<<<num_blk, thd_per_blk>>>(a_device, n_device, shared_device);
  cudaMemcpy(shared, shared_device, sizeof(int)*nthds*(nthds-1), cudaMemcpyDeviceToHost);

  /* global splitter sort and final splitter selection in host */
  qsort(shared, 0, (nthds-1)*nthds-1);
  int* final_splitters = &shared[nthds*(nthds-1)];
  int d = nthds / 2;
  for (int i = 0; i < nthds-1; i++){
    final_splitters[i] = shared[d+i*nthds];
  }
  cudaMemcpy(&shared_device[nthds*(nthds-1)], &shared[nthds*(nthds-1)], sizeof(int)*(nthds-1), cudaMemcpyHostToDevice);

  /* kernel : bucket sort */
  bucket_sort<<<num_blk, thd_per_blk>>>(a_device, n_device, shared_device);

  /* kernel : write result */
  out<<<num_blk, thd_per_blk>>>(out_device, shared_device);
  cudaDeviceSynchronize();

  /* correctness checking & timing output*/
  GET_TIME(end);
  cudaMemcpy(output, out_device, sizeof(int)*n, cudaMemcpyDeviceToHost);
  checkSorting(n, output, oracle);
  printf("computed in %.8fs\n", end-start);

  cudaFree(a_device); cudaFree(out_device); cudaFree(shared_device);
  free(a); free(oracle); free(output);
  return 0;
}

/* ************* */
__host__ void readSorting(char *filename, int* n, int** a, int** oracle){
  int i;
  FILE* fr = fopen(filename, "r");
  fscanf(fr, "%d", n);
  *a = (int*) malloc(sizeof(int)* *n);
  *oracle = (int*) malloc(sizeof(int)* *n);
  for (i =0; i< *n; i++){
    fscanf(fr, "%d", &(*a)[i]);
  }
  fclose(fr);

  char output_buf[50];
  sprintf(output_buf, "../testcases/output_sorting_%d", *n);
  fr = fopen(output_buf, "r");
  for (i = 0; i< *n; i++){
    fscanf(fr, "%d", &(*oracle)[i]);
  }
  fclose(fr);
}

__host__ void checkSorting(int n, int* a, int* oracle){
  int i;
  int incorrect = 0;
  for (i = 0; i < n; i++){
    if (a[i] != oracle[i]){
      // printf("a[%d]=%d vs %d\n", i, a[i], oracle[i]);
      incorrect = 1;
    }
  }
  if (incorrect) printf("Sorting incorrect\n");
  else printf("Sorting correct\n");
}

__device__ int* realloc(int* old_a, int length, int new_length){
  int* new_a = (int*) malloc (sizeof(int)*new_length);

  for (int i=0; i<length; i++){
    new_a[i] = old_a[i];
  }
  free(old_a);
  return new_a;
}

/* ************* */
/* cuda qsort */
__host__ __device__ void swap(int* a, int* b){
  int temp = *a;
  *a = *b;
  *b = temp;
}

__host__ __device__ int partition(int* a, int low, int high){
  int pi = a[high];
  int i = low-1;
  for (int j = low; j < high; j++){
    if (a[j] < pi){
      i++;
      swap(&a[i], &a[j]);
    }
  }
  swap(&a[i+1], &a[high]);
  return i+1;
}

__host__ __device__  void qsort(int* arr, int l, int h){
  int* stack = (int*) malloc(sizeof(int) * (h - l + 1));
  int top = -1;

  stack[++top] = l;
  stack[++top] = h;

  while (top >= 0) {
    h = stack[top--];
    l = stack[top--];

    int p = partition(arr, l, h);

    if (p - 1 > l) {
      stack[++top] = l;
      stack[++top] = p - 1;
    }

    if (p + 1 < h) {
      stack[++top] = p + 1;
      stack[++top] = h;
    }
  }
  free(stack);
}
/* ************* */
