/**
 * @file pth_summa_2.c
 * @brief 
 * Square processor layout -- "round-up" approach with cyclic mapping
 * Each process mapped to at most 2 blocks
 * 
 * Use non-blocking ibcast to prevent deadlock in columns
 * 
 * Memory allocation of full size A, B, C
 * Might cause system memory overflow with large problem size
 * because of the nature of shared memory system (each thread does not have dedicated core)
 * hence it is better not to simultaneously use high <nthds> with a large problem size
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <semaphore.h>
#include <math.h>
#include "../util/utils.h"
#include "../util/timer.h"

/* define functions */
#define bz_idx(i, grid_sz) (i == grid_sz - 1 ? 1 : 0)
#define max(a, b) (a > b? a : b)

int cmpfunc (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}

/* communicator and data type definitions */
typedef struct comm_node {
  int *comm_addr;
  sem_t ready;
  sem_t wait;
  int ibast_count;
} comm_node;

// simulate MPI "subarray" type
typedef struct sub_array {
  int block_i, block_j;
  int array_i, array_j;
} sub_array;
sub_array INT_TYPE = {1,1,1,1};

typedef struct comm_t {
  int N; // number of threads in this comm
  comm_node* nodes;
  int* counts;
  sub_array **types;
  pthread_mutex_t mutex;
  pthread_barrier_t b;
} comm_t;

/* functions */
void initComm(comm_t *c, int n);
void destComm(comm_t *c);
void broadcast(comm_t *c, int sender, int my_rank, int msg_sz, int* comm_addr, sub_array* type);
void ibroadcast(comm_t *c, int block, int loop,  int sender, int my_rank, int global_rank, int msg_sz, int* comm_addr, sub_array* type);
void wait(comm_t *c, int my_rank);
void buildSubarrayTypes(int* MKN, int proc_grid_sz, int** MKN_sz, sub_array** ABC_dt);
void alltoallw_scatter(comm_t * c, int my_rank, int sender, int* counts_send, int* disp_send, sub_array** types_send, int* buf_send, int* buf_recv, int* counts_recv, int* disp_recv, sub_array** types_recv);
void alltoallw_gather(comm_t * c, int my_rank, int receiver, int* counts_send, int* disp_send, sub_array** types_send, int* buf_send, int* buf_recv, int* counts_recv, int* disp_recv, sub_array** types_recv);
void distributeBlocksP0(int my_rank, int comm_sz, int grid_sz, int* my_rows, int* my_cols, int* MKN, int** MKN_sz, sub_array** ABC_dt, int** A, int** B, int** local_A, int** local_B, int* flags);
void gatherResultsP0(int my_rank, int comm_sz, int grid_sz, int* my_rows, int* my_cols, int* MKN, int** MKN_sz, sub_array* C_dt, int** C, int** local_C, int* flags);

/* shared communicator -- simulate MPI runtime */
comm_t new_comm;
comm_t *row_comms, *col_comms;
int** row_group_mem, **col_group_mem;
int** row_ranks, **col_ranks;
/* shared program input */
int comm_sz, grid_sz, logical_p;
char* input_filename;

/* simulate MPI program */
void *thread_func(void *arg){
  int my_rank = (long) arg;
  int **A, **B, **C, *oracle;
  int** local_A, **local_B, **local_C;
  int MKN[3];
  /* read from file */
  if (my_rank == 0){
    mpi_readInputC(input_filename, MKN, &A, &B, &C, &oracle);
    if (grid_sz > MKN[0] || grid_sz > MKN[1] || grid_sz > MKN[2]){
      printf("matrix dimensions >= sqrt(nprocs)\n");
      exit(-1);
    }
  } else {
    mallocCont(&A, 1, 1);
    mallocCont(&B, 1, 1);
    mallocCont(&C, 1, 1);
    oracle = malloc(sizeof(int));
  }
  broadcast(&new_comm, 0, my_rank, 3, MKN, &INT_TYPE);
  /* memory allocation of !! full size A, B, C */
  mallocCont(&local_A, MKN[0], MKN[1]);
  mallocCont(&local_B, MKN[1], MKN[2]);
  mallocCont(&local_C, MKN[0], MKN[2]);
  memset(&local_A[0][0], 0, sizeof(int)* MKN[0]* MKN[1]);
  memset(&local_C[0][0], 0, sizeof(int)* MKN[0]* MKN[2]);

  /* cyclic mapping */
  int flags[2] = {0, 0};
  int my_rows[2] = {-1, -1}, my_cols[2] = {-1, -1};
  int i, k, r, c;
  for (i = 0; i < 2; i++){
    if (my_rank + i*comm_sz < logical_p){
      flags[i] = 1;
      my_rows[i] = (my_rank + i*comm_sz) / grid_sz;
      my_cols[i] = (my_rank + i*comm_sz) % grid_sz;
    }
  }
  /* determine block sizes and construct subarray types */
  int M_sz[2], K_sz[2], N_sz[2];
  int* MKN_sz[3] = {M_sz, K_sz, N_sz};
  sub_array A_dt[4], B_dt[4], C_dt[4];
  sub_array* ABC_dt[3] = {A_dt, B_dt, C_dt}; 
  buildSubarrayTypes(MKN, grid_sz, MKN_sz, ABC_dt);

  /* performance measurement */
  double start, end;
  GET_TIME(start);

  distributeBlocksP0(my_rank, comm_sz, grid_sz, my_rows, my_cols, MKN, MKN_sz, ABC_dt, A, B, local_A, local_B, flags);

  /* SUMMA algorithm */
  int sender_rank, rank_in_comm;
  int start_i, start_j, start_k;
  int kk, ii, jj;
  for (k = 0; k < grid_sz; k++){
    /* broadcast A in row comm */
    for (i = 0 ; i < 2; i++){
      if (flags[i]){
        sender_rank = row_group_mem[my_rows[i]][k];
        rank_in_comm = row_ranks[my_rows[i]][sender_rank];
        int my_rank_in_comm = row_ranks[my_rows[i]][my_rank];
        start_i = my_rows[i] * M_sz[0];
        start_j = k * K_sz[0];
        r = bz_idx(my_rows[i], grid_sz);
        c = bz_idx(k, grid_sz);
        broadcast(&row_comms[my_rows[i]], rank_in_comm, my_rank_in_comm, 1, &local_A[start_i][start_j], &A_dt[r*2+c]);
      }
    }
    // switch order to prevent deadlock with cyclic mapping
    if (k == grid_sz -1){
      for (i = 1; i >=0; i--){
        if (flags[i]){
          if (i == 0 && my_cols[0] == my_cols[1]) break;
          sender_rank = col_group_mem[my_cols[i]][k];
          rank_in_comm = col_ranks[my_cols[i]][sender_rank];
          int my_rank_in_comm = col_ranks[my_cols[i]][my_rank];
          start_i = k * K_sz[0];
          start_j = my_cols[i] * N_sz[0];
          r = bz_idx(k, grid_sz);
          c = bz_idx(my_cols[i], grid_sz);
          ibroadcast(&col_comms[my_cols[i]], i, k, rank_in_comm, my_rank_in_comm, my_rank, 1, &local_B[start_i][start_j], &B_dt[r*2+c]);
        }
      }
    } else {
      for (i = 0 ; i < 2; i++){
      
      // if (k == grid_sz -1) i = 1-i;
        if (flags[i]){
          if (i == 1 && my_cols[0] == my_cols[1]) break;
          sender_rank = col_group_mem[my_cols[i]][k];
          rank_in_comm = col_ranks[my_cols[i]][sender_rank];
          int my_rank_in_comm = col_ranks[my_cols[i]][my_rank];
          start_i = k * K_sz[0];
          start_j = my_cols[i] * N_sz[0];
          r = bz_idx(k, grid_sz);
          c = bz_idx(my_cols[i], grid_sz);
          /* use non blocking ibcast to prevent deadlock in column broadcast */
          ibroadcast(&col_comms[my_cols[i]], i, k, rank_in_comm, my_rank_in_comm, my_rank, 1, &local_B[start_i][start_j], &B_dt[r*2+c]);
        }
        // if (k == grid_sz -1) i = 1-i;
      }
    }
    /* wait for broadcast to finish */
    for (i = 0 ; i < 2; i++){
      if (flags[i]){
        if (i == 1 && my_cols[0] == my_cols[1]) break;
        int my_rank_in_comm = col_ranks[my_cols[i]][my_rank];
        wait(&col_comms[my_cols[i]], my_rank_in_comm);
      }
    }

    for (i = 0; i < 2; i++){
      if (flags[i]){
        start_i = my_rows[i] * M_sz[0];
        start_j = my_cols[i] * N_sz[0];
        start_k = k * K_sz[0];
        int m = bz_idx(k, grid_sz);
        r = bz_idx(my_rows[i], grid_sz);
        c = bz_idx(my_cols[i], grid_sz);
        for (kk = 0; kk < K_sz[m]; kk++){
          for (ii = 0; ii < M_sz[r]; ii++){
            for (jj = 0; jj < N_sz[c]; jj++){
              local_C[ii+start_i][jj+start_j] += local_A[ii+start_i][kk+start_k] * local_B[kk+start_k][jj+start_j];
            }
          }
        }
      }
    }
  }
  gatherResultsP0(my_rank, comm_sz, grid_sz, my_rows, my_cols, MKN, MKN_sz, C_dt, C, local_C, flags);

  /* correctness checking and timing output*/
  GET_TIME(end);
  if (my_rank == 0){
    checkCorrectness(MKN[0], MKN[2], C, oracle);
    printf("computed in %.8fs\n", end-start);
  }

  free(A); free(B); free(C); free(oracle);
  free(local_A); free(local_B); free(local_C);
  return NULL;
}


int main(int argc, char* argv[]){
  if (argc != 3){
    printf("./pth_summa_2 <num_threads> <input_filename>\n");
    exit(-1);
  }
  comm_sz = atoi(argv[1]);
  input_filename = argv[2];
  grid_sz = (int) ceil(sqrt((double) comm_sz));
  logical_p = grid_sz * grid_sz;
  pthread_t tid[comm_sz];
  int i, j;

  /* -------------------------------------*/
  /* build world, row and column communicators */
  initComm(&new_comm, comm_sz);
  row_comms = malloc(sizeof(comm_t) * grid_sz);
  col_comms = malloc(sizeof(comm_t) * grid_sz);
  row_group_mem = malloc(sizeof(int*) * grid_sz);
  col_group_mem = malloc(sizeof(int*) * grid_sz);
  row_ranks = malloc(sizeof(int*) * grid_sz);
  col_ranks = malloc(sizeof(int*) * grid_sz);
  for (i = 0; i < grid_sz; i++){
    row_group_mem[i] = malloc(sizeof(int) * grid_sz);
    col_group_mem[i] = malloc(sizeof(int) * grid_sz);
    row_ranks[i] = malloc(sizeof(int) * comm_sz);
    col_ranks[i] = malloc(sizeof(int) * comm_sz);
    memset(row_ranks[i], -1, sizeof(int) * comm_sz);
    memset(col_ranks[i], -1, sizeof(int) * comm_sz);
  }

  int row_sort[grid_sz], col_sort[grid_sz];
  int row_group_uniq_mem[grid_sz][grid_sz];
  int col_group_uniq_mem[grid_sz][grid_sz];
  int row_uniq_n[grid_sz], col_uniq_n[grid_sz];

  for (i=0; i < grid_sz; i++){
    for (j=0; j < grid_sz; j++){
      row_group_mem[i][j] = row_sort[j] = (i * grid_sz + j) % comm_sz;
      col_group_mem[i][j] = col_sort[j] = (j * grid_sz + i) % comm_sz;
    }
    // uniq group members
    qsort(row_sort, grid_sz, sizeof(int), cmpfunc);
    qsort(col_sort, grid_sz, sizeof(int), cmpfunc);
    row_group_uniq_mem[i][0] = row_sort[0];
    col_group_uniq_mem[i][0] = col_sort[0];
    row_uniq_n[i] = col_uniq_n[i] = 1;
    for (j=1; j < grid_sz; j++){
      if (row_sort[j] != row_sort[j-1]){
        row_group_uniq_mem[i][row_uniq_n[i]++] = row_sort[j];
      }
      if (col_sort[j] != col_sort[j-1]){
        col_group_uniq_mem[i][col_uniq_n[i]++] = col_sort[j];
      }
    }
    for (j = 0; j < row_uniq_n[i]; j++){
      row_ranks[i][row_group_uniq_mem[i][j]] = j;
    }
    for (j = 0; j < col_uniq_n[i]; j++){
      col_ranks[i][col_group_uniq_mem[i][j]] = j;
    }
  
    initComm(&row_comms[i], row_uniq_n[i]);
    initComm(&col_comms[i], col_uniq_n[i]);
  }

  /* -------------------------------------*/

  /* create threads */
  for (long rank = 0; rank < comm_sz; rank++) {
    pthread_create(&tid[rank], NULL, thread_func, (void *)rank);
  }

  /* join threads */
  for (int rank = 0; rank < comm_sz; rank++) {
    pthread_join(tid[rank], NULL);
  }

  /* cleanup */
  for (i = 0; i < grid_sz; i++){
    destComm(&row_comms[i]);
    destComm(&col_comms[i]);
  }
  destComm(&new_comm);
  free(row_comms);
  free(col_comms);
  return 0;
}

/* -------------------------------------*/
void initComm(comm_t *c, int n){
  c->N = n;
  c->nodes = malloc(sizeof(comm_node) * n);
  pthread_barrier_init(&c->b, NULL, n);
  pthread_mutex_init(&c->mutex, NULL);

  for (int i = 0; i < n; i++){
    c->nodes[i].ibast_count = 0;
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
  pthread_mutex_destroy(&c->mutex);
  free(c->nodes);
}

/* -------------------------------------*/
/**
 * @brief 
 * build MPI-equivalent subarray types for for each submatrix size
 */
void buildSubarrayTypes(int* MKN, int proc_grid_sz, int** MKN_sz, sub_array** ABC_dt){
  int i, j, k;
  for (i = 0; i < 3; i++){
    MKN_sz[i][0] = floor((double) MKN[i] / (double) proc_grid_sz);
    MKN_sz[i][1] = MKN[i] - MKN_sz[i][0] * (proc_grid_sz-1);
  }
  struct dim {
    int r, c; //row, col
  };
  struct dim ABC_dim[3] = {{0, 1}, {1, 2}, {0, 2}}; // M x K, K x N, M x N

  for (k = 0; k < 3; k++){ // loop over ABC
    for (i = 0; i < 2; i++){
      for (j = 0; j < 2; j++){
        ABC_dt[k][i*2+j].block_i = MKN_sz[ABC_dim[k].r][i];
        ABC_dt[k][i*2+j].block_j = MKN_sz[ABC_dim[k].c][j];
        ABC_dt[k][i*2+j].array_i = MKN[ABC_dim[k].r];
        ABC_dt[k][i*2+j].array_j = MKN[ABC_dim[k].c];
      }
    }
  }
}


/* -------------------------------------*/
/**
 * @brief 
 * associated wait operation for non-blocking broadcast
 */
void wait(comm_t *c, int my_rank){
  sem_wait(&c->nodes[my_rank].wait);
  c->nodes[my_rank].ibast_count = 0;
}

/**
 * @brief 
 * non-blocking broadcast
 * use shared buffer to store data to be sent
 */
void ibroadcast(comm_t *c, int block, int loop, int sender, int my_rank, int global_rank, int msg_sz, int* comm_addr, sub_array* type){
  /* sender : buffer data */
  if (my_rank == sender){
    for (int i = 0; i < c->N; i++){
      c->nodes[i].comm_addr = comm_addr;
      sem_post(&c->nodes[i].ready);
    }
  }
  /* receive */
  sem_wait(&c->nodes[my_rank].ready);
  if (my_rank != sender){
    int *inter_recv = c->nodes[my_rank].comm_addr;
    for (int j = 0; j < msg_sz; j++){
      for (int k = 0; k < type->block_i; k++){
        for (int m = 0; m < type->block_j; m++){
          comm_addr[j+ k*type->array_j+ m] = inter_recv[j+ k*type->array_j+ m];
        }
      }
    }
  }
  
  /* use mutex and semaphore for the associated wait operation */
  pthread_mutex_lock(&c->mutex);
  c->nodes[my_rank].ibast_count = 1;
  for (int i = 0; i < c->N; i++){
    if (c->nodes[i].ibast_count != 1) goto end;
  }
  for (int i = 0; i < c->N; i++){
    sem_post(&c->nodes[i].wait);
  }
  end:
  pthread_mutex_unlock(&c->mutex);
}

/* -------------------------------------*/
/**
 * @brief 
 *  O(logp) tree-structured blocking broadcast implemented with semaphores and barrier
 */
void broadcast(comm_t *c, int sender, int my_rank, int msg_sz, int* comm_addr, sub_array* type){
  /* use rank 0 as the sender */
  if (my_rank == 0 && my_rank != sender){ 
    my_rank = sender;
  } else if (my_rank == sender && my_rank != 0){
    my_rank = 0;
  }
  int partner;
  
  c->nodes[my_rank].comm_addr = comm_addr; 
  if (my_rank != 0){
    sem_post(&c->nodes[my_rank].ready);
  }
  /* tree-structured broadcast */
  for (int i = pow(2, floor(log2((double) c->N))); i >= 1; i >>= 1){
    partner = my_rank ^ i;
    if (my_rank % i == 0 && partner < my_rank) {
      sem_wait(&c->nodes[my_rank].wait);
    } else if (my_rank % i == 0 && partner < c->N){
      sem_wait(&c->nodes[partner].ready);
      int *dest = c->nodes[partner].comm_addr;
      int *src = c->nodes[my_rank].comm_addr;
      for (int j = 0; j < msg_sz; j++){
        for (int k = 0; k < type->block_i; k++){
          for (int m = 0; m < type->block_j; m++){
            dest[j+k*type->array_j+ m] = src[j+ k*type->array_j+ m];
          }
        }
      }
      sem_post(&c->nodes[partner].wait);
    }
  }
  pthread_barrier_wait(&c->b);
}

/**
 * @brief 
 * use Pthread (simplified) version of alltoallw to distribute submatrices of A and B
 * cyclic mapping, hence twice its usage in the worst case
 */
void distributeBlocksP0(int my_rank, int comm_sz, int grid_sz, int* my_rows, int* my_cols, int* MKN, int** MKN_sz, sub_array** ABC_dt, int** A, int** B, int** local_A, int** local_B, int* flags){
  int i, j, k, rr, cc, n, m, r, c;
  int counts_send[2][comm_sz], counts_recv[2][comm_sz];
  int dis_send[2][comm_sz], dis_recv[2][comm_sz];
  sub_array* types_send[2][comm_sz], *types_recv[2][comm_sz];
  
  for (i = 0; i < 2; i++){
    for (j = 0; j < comm_sz; j++){
      for (k = 0; k < 2; k++){
        types_recv[k][j] = types_send[k][j] = &INT_TYPE;
        counts_recv[k][j] = counts_send[k][j] = 0;
        dis_recv[k][j] = dis_send[k][j] = 0;
      }
    }
    if (flags[i]){
      r = bz_idx(my_rows[i], grid_sz);
      c = bz_idx(my_cols[i], grid_sz);
      counts_recv[0][0] = 1; // A
      dis_recv[0][0] = my_rows[i] * MKN_sz[0][0] * MKN[1] + my_cols[i] * MKN_sz[1][0];
      types_recv[0][0] = &ABC_dt[0][r*2+c];
      counts_recv[1][0] = 1; // B
      dis_recv[1][0] = my_rows[i] * MKN_sz[1][0] * MKN[2] + my_cols[i] * MKN_sz[2][0];
      types_recv[1][0] = &ABC_dt[1][r*2+c];
    }
    if (my_rank == 0){
      int logical_p = grid_sz * grid_sz;
      for (int rank = 0; rank < comm_sz; rank++){
        if (rank + i*comm_sz < logical_p){
          rr = (rank + i*comm_sz) / grid_sz;
          cc = (rank + i*comm_sz) % grid_sz;
          m = bz_idx(rr, grid_sz);
          n = bz_idx(cc, grid_sz);
          counts_send[0][rank] = counts_send[1][rank] = 1;
          types_send[0][rank] = &ABC_dt[0][m*2+n];
          types_send[1][rank] = &ABC_dt[1][m*2+n];
          dis_send[0][rank] = rr * MKN_sz[0][0] * MKN[1] + cc * MKN_sz[1][0];
          dis_send[1][rank] = rr * MKN_sz[1][0] * MKN[2] + cc * MKN_sz[2][0];
        }
      }
    }
    alltoallw_scatter(&new_comm, my_rank, 0, counts_send[0], dis_send[0], types_send[0], &A[0][0], &local_A[0][0], counts_recv[0], dis_recv[0], types_recv[0]);
    alltoallw_scatter(&new_comm, my_rank, 0, counts_send[1], dis_send[1], types_send[1], &B[0][0], &local_B[0][0], counts_recv[1], dis_recv[1], types_recv[1]);
  }
}

/**
 * @brief 
 * use Pthread (simplified) version of alltoallw to gather submatrices of C
 * cyclic mapping, hence twice its usage in the worst case
 */
void gatherResultsP0(int my_rank, int comm_sz, int grid_sz, int* my_rows, int* my_cols, int* MKN, int** MKN_sz, sub_array* C_dt, int** C, int** local_C, int* flags){
  int i, j, r, c, rr, cc, m, n;
  int counts_send[comm_sz], counts_recv[comm_sz];
  int dis_send[comm_sz], dis_recv[comm_sz];
  sub_array* types_send[comm_sz], *types_recv[comm_sz];
  
  for (i = 0; i < 2; i++){
    for (j = 0; j < comm_sz; j++){
      types_recv[j] = types_send[j] = &INT_TYPE;
      counts_recv[j] = counts_send[j] = 0;
      dis_recv[j] = dis_send[j] = 0;
    }
    if (flags[i]){
      r = bz_idx(my_rows[i], grid_sz);
      c = bz_idx(my_cols[i], grid_sz);
      counts_send[0] = 1; 
      dis_send[0] = my_rows[i] * MKN_sz[0][0] * MKN[2] + my_cols[i] * MKN_sz[2][0];
      types_send[0] = &C_dt[r*2+c];
    }
    if (my_rank == 0){
      int logical_p = grid_sz * grid_sz;
      for (int rank = 0; rank < comm_sz; rank++){
        if (rank + i*comm_sz < logical_p){
          rr = (rank + i*comm_sz) / grid_sz;
          cc = (rank + i*comm_sz) % grid_sz;
          m = bz_idx(rr, grid_sz);
          n = bz_idx(cc, grid_sz);
          counts_recv[rank] = 1;
          types_recv[rank] = &C_dt[m*2+n];
          dis_recv[rank] = rr * MKN_sz[0][0] * MKN[2] + cc * MKN_sz[2][0];
        }
      }
    }
    alltoallw_gather(&new_comm, my_rank, 0, counts_send, dis_send, types_send, &local_C[0][0], &C[0][0], counts_recv, dis_recv, types_recv);
  }
}

/**
 * @brief 
 * O(logp) tree-structured scatter operation
 * restrictive version of alltoallw 
 * -- only allows scatter by P0 but supports operation with multiple matrix sizes
 */
void alltoallw_scatter(comm_t * c, int my_rank, int sender, int* counts_send, int* disp_send, sub_array** types_send, int* buf_send, int* buf_recv, int* counts_recv, int* disp_recv, sub_array** types_recv){
  if (my_rank == sender){
    c->counts = counts_send;
    c->types = types_send;
  }
  int partner, count, i, j, k, n, m, disp_j;

  /* tree-structured scatter */
  for (i = pow(2, floor(log2((double) c->N))); i >= 1; i >>= 1){
    partner = my_rank ^ i;
    if (my_rank % i == 0 && partner < my_rank) {
      /* receiver */
      sem_wait(&c->nodes[my_rank].ready);
    } else if (my_rank % i == 0 && partner < c->N){
      /* send upper half of data owned to partner */
      count = 0;
      for (j = partner; j < partner + i; j++){
        if (j < c->N){
          count += c->counts[j] * (c->types[j]->block_i) * (c->types[j]->block_j);
        }
      }
      int* inter_buf = malloc(sizeof(int) * count);
      if (my_rank == 0){
        /* sender copies from source memory */
        count = 0;
        for (j = partner; j < partner + i; j++){
          if (j < c->N){
            disp_j = disp_send[j];
            for (k = 0; k < counts_send[j]; k++){
              for (n = 0; n < types_send[j]->block_i; n++){
                for (m = 0; m < types_send[j]->block_j; m++){
                  inter_buf[count++] = buf_send[disp_j+ k+ n*(types_send[j]->array_j) + m];
                }
              }
            }
          } 
        }

      } else {
        /* intermediate nodes : copy from intermediate buffers */
        int* my_buf_send = c->nodes[my_rank].comm_addr;
        disp_j = 0;
        for (j = my_rank; j < partner; j++){
          disp_j += c->counts[j] * (c->types[j]->block_i) * (c->types[j]->block_j);
        }
        for (j = 0; j < count; j++){
          inter_buf[j] = my_buf_send[disp_j+j];
        }
      }
      c->nodes[partner].comm_addr = inter_buf;
      sem_post(&c->nodes[partner].ready);
    }
  }
  /* copy data into destination memory */
  if (my_rank != 0){
    /* receive from buffers */
    int* my_buf_recv = c->nodes[my_rank].comm_addr;
    count = 0;
    for (i = 0; i < counts_recv[0]; i++){
      for (j = 0; j < types_recv[0]->block_i; j++){
        for (k = 0; k < types_recv[0]->block_j; k++){
          buf_recv[disp_recv[0] + i+ j*types_recv[0]->array_j + k] = my_buf_recv[count++];
        }
      }
    }
  } else {
    /* sender : receive from source memory */
    for (i = 0; i < counts_recv[0]; i++){
      for (j = 0; j < types_recv[0]->block_i; j++){
        for (k = 0; k < types_recv[0]->block_j; k++){
          buf_recv[disp_recv[0] + i+ j*types_recv[0]->array_j + k] = buf_send[disp_send[0] + j*types_recv[0]->array_j + k];
        }
      }
    }
  }
  if (my_rank != 0){
    free(c->nodes[my_rank].comm_addr);
  }
  pthread_barrier_wait(&c->b);
}

/**
 * @brief 
 * O(logp) tree-structured gather operation
 * restrictive version of alltoallw 
 * -- only allows gather by P0 but supports operation with multiple matrix sizes
 */
void alltoallw_gather(comm_t * c, int my_rank, int receiver, int* counts_send, int* disp_send, sub_array** types_send, int* buf_send, int* buf_recv, int* counts_recv, int* disp_recv, sub_array** types_recv){
  if (my_rank == receiver){
    c->counts = counts_recv;
    c->types = types_recv;
  }
  int partner, count, i, j, k, n, m, disp_j;
  pthread_barrier_wait(&c->b);
  int* inter_buf = NULL;
  /* t = 0 */
  if (my_rank != 0){
    /* preprocess: copy non-contiguous data into contiguous memory */
    count = counts_send[0] * types_send[0]->block_i * types_send[0]->block_j;
    inter_buf = malloc(sizeof(int) * count);
    for (i = 0, count =0; i < counts_send[0]; i++){
      for (j = 0; j < types_send[0]->block_i; j++){
        for (k = 0; k < types_send[0]->block_j; k++){
          inter_buf[count++] = buf_send[disp_send[0] + i + j*(types_send[0]->array_j) +k];
        }
      }
    }
  } else {
  /* receiver: copy its own data into destination memory */
    for (i = 0; i < counts_recv[0]; i++){
      for (j = 0; j < types_recv[0]->block_i; j++){
        for (k = 0; k < types_recv[0]->block_j; k++){
          buf_recv[disp_recv[0] + i + j*(types_recv[0]->array_j) + k] = buf_send[disp_send[0] + i + j*(types_send[0]->array_j) + k];
        }
      }
    }
  }
  /* tree-structured gather */
  for (i = 1; i < c->N; i<<=1){
    partner = my_rank ^ i;
    if (my_rank % i == 0 && partner < my_rank) {
      /* sender */
      c->nodes[my_rank].comm_addr = inter_buf;
      sem_post(&c->nodes[my_rank].ready);
      sem_wait(&c->nodes[my_rank].wait);
      break;
    }else if (my_rank % i == 0 && partner < c->N){
      /* receiver */
      if (my_rank != 0){
        /* intermediate nodes : combine received data in buffer */
        count = disp_j = 0;
        for (j = my_rank; j < partner + i; j++){
          if (j == partner){
            disp_j = count;
          }
          if (j < c->N){
            count += c->counts[j] * c->types[j]->block_i * c->types[j]->block_j;
          }
        }
        inter_buf = realloc(inter_buf, sizeof(int) * count);
        sem_wait(&c->nodes[partner].ready);
        int* inter_recv = c->nodes[partner].comm_addr;
        for (j = disp_j; j < count; j++){
          inter_buf[j] = inter_recv[j-disp_j];
        }
        sem_post(&c->nodes[partner].wait);
      } else {
        /* destination receiver : copy to destination memory */
        sem_wait(&c->nodes[partner].ready);
        int* inter_recv = c->nodes[partner].comm_addr;
        count = 0;
        for (j = partner; j < partner + i; j++){
          if (j < c->N){
            disp_j = disp_recv[j];
            for (k = 0; k < counts_recv[j]; k++){
              for (n = 0; n < types_recv[j]->block_i; n++){
                for (m = 0; m < types_recv[j]->block_j; m++){
                  buf_recv[disp_j+ k + n*(types_recv[j]->array_j) + m] = inter_recv[count++];
                }
              }
            }
          } 
        }
        sem_post(&c->nodes[partner].wait);
      }
    }
  }

  if (my_rank != 0){
    free(c->nodes[my_rank].comm_addr);
  }
  pthread_barrier_wait(&c->b);
}