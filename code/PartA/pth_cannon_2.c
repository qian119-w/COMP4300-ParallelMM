/**
 * @file pth_cannon_2.c
 * @brief 
 * Square processor layout -- "round-up" approach with cyclic mapping
 * Each process mapped to at most 2 blocks
 * 
 * Use non-blocking isend to prevent deadlock in columns
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

/* communicator and data type definitions */
typedef struct comm_node {
  // collective
  int *comm_addr;
  sem_t ready;
  sem_t wait;
  // point to point
  int ** ptp_addrs; 
  sem_t* ptp_readys;
  sem_t* ptp_waits;
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
  pthread_cond_t cond;
  pthread_barrier_t b;
} comm_t;

/* functions */
void initComm(comm_t *c, int n);
void destComm(comm_t *c);
void broadcast(comm_t *c, int sender, int my_rank, int msg_sz, int* comm_addr, sub_array* type);
void buildSubarrayTypes(int* MKN, int proc_grid_sz, int** MKN_sz, sub_array** ABC_dt);
void alltoallw_scatter(comm_t * c, int my_rank, int sender, int* counts_send, int* disp_send, sub_array** types_send, int* buf_send, int* buf_recv, int* counts_recv, int* disp_recv, sub_array** types_recv);
void alltoallw_gather(comm_t * c, int my_rank, int receiver, int* counts_send, int* disp_send, sub_array** types_send, int* buf_send, int* buf_recv, int* counts_recv, int* disp_recv, sub_array** types_recv);
void distributeBlocksP0(int my_rank, int comm_sz, int grid_sz, int* my_rows, int* my_cols, int* MKN, int** MKN_sz, sub_array** ABC_dt, int** A, int** B, int** local_A, int** local_B, int* flags);
void gatherResultsP0(int my_rank, int comm_sz, int grid_sz, int* my_rows, int* my_cols, int* MKN, int** MKN_sz, sub_array* C_dt, int** C, int** local_C, int* flags);
void sendrecv_replace(comm_t *c, int my_rank, int global_rank, int dest, int src, int* buf, int count);
void isend(comm_t* c, int my_rank, int* buf, int count, int dest, sem_t** request);
void wait(sem_t* request);
void recv(comm_t* c, int my_rank, int* buf, int count, int src);

/* shared communicator -- simulate MPI runtime */
comm_t new_comm;
/* shared program input */
int comm_sz, grid_sz, logical_p;
char* input_filename;

/* simulate MPI program */
void *thread_func(void *arg){
  int my_rank = (long) arg;
  int **A, **B, **C, *oracle;
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

  /* determine block sizes and construct subarray types */
  int M_sz[2], K_sz[2], N_sz[2];
  int* MKN_sz[3] = {M_sz, K_sz, N_sz};
  sub_array A_dt[4], B_dt[4], C_dt[4];
  sub_array* ABC_dt[3] = {A_dt, B_dt, C_dt}; 
  buildSubarrayTypes(MKN, grid_sz, MKN_sz, ABC_dt);

  /* cyclic mapping */
  int flags[2] = {0, 0};
  int my_rows[2] = {-1, -1}, my_cols[2] = {-1, -1};
  int dest_A[2], dest_B[2];
  int src_A[2], src_B[2];
  int* local_A[2], *local_B[2], *temp_B[2], *local_C[2];
  int i, k, r, c;
  for (i = 0; i < 2; i++){
    if (my_rank + i*comm_sz < logical_p){
      flags[i] = 1;
      my_rows[i] = (my_rank + i*comm_sz) / grid_sz;
      my_cols[i] = (my_rank + i*comm_sz) % grid_sz;
      dest_A[i] = ((my_cols[i] + grid_sz - 1) % grid_sz + my_rows[i] * grid_sz) % comm_sz;
      src_A[i] = ((my_cols[i] + 1) % grid_sz + my_rows[i] * grid_sz) % comm_sz;
      dest_B[i] = (((my_rows[i] + grid_sz - 1) % grid_sz) * grid_sz + my_cols[i]) % comm_sz;
      src_B[i] = (((my_rows[i] + 1) % grid_sz) * grid_sz + my_cols[i]) % comm_sz;
      r = bz_idx(my_rows[i], grid_sz);
      c = bz_idx(my_cols[i], grid_sz);
      local_A[i] = malloc(sizeof(int) * M_sz[r] * max(K_sz[0], K_sz[1]));
      local_B[i] = malloc(sizeof(int) * max(K_sz[0], K_sz[1]) * N_sz[c]);
      temp_B[i] = malloc(sizeof(int) * max(K_sz[0], K_sz[1]) * N_sz[c]);
      local_C[i] = malloc(sizeof(int) * M_sz[r] * N_sz[c]);
      memset(local_A[i], 0, sizeof(int) * M_sz[r] * max(K_sz[0], K_sz[1]));
      memset(local_C[i], 0, sizeof(int) * M_sz[r] * N_sz[c]);
    }
  }
  /* performance measurement */
  double start, end;
  GET_TIME(start);

  distributeBlocksP0(my_rank, comm_sz, grid_sz, my_rows, my_cols, MKN, MKN_sz, ABC_dt, A, B, local_A, local_B, flags);
  int shifted_coor, shifted_k;


  /* Cannon algorithm */
  int kk, ii, jj;
  sem_t* request[2];
  for (k = 0; k < grid_sz; k++){
    for (i = 0; i < 2; i++){
      if (flags[i]){
        shifted_coor = (my_rows[i] + my_cols[i] + k) % grid_sz;
        shifted_k = bz_idx(shifted_coor, grid_sz);
        r = bz_idx(my_rows[i], grid_sz);
        c = bz_idx(my_cols[i], grid_sz);

        for (kk = 0; kk < K_sz[shifted_k]; kk++){
          for (ii = 0; ii < M_sz[r]; ii++){
            for (jj = 0; jj < N_sz[c]; jj++){
              local_C[i][ii*N_sz[c] + jj] += local_A[i][ii*K_sz[shifted_k]+kk] * local_B[i][kk*N_sz[c]+jj];
            }
          }
        }
      }
    }
    /* shift A */
    for (i = 0; i < 2; i++){
      if (flags[i]){
        r = bz_idx(my_rows[i], grid_sz);
        sendrecv_replace(&new_comm, my_rank, my_rank, dest_A[i], src_A[i], local_A[i], M_sz[r] * max(K_sz[0], K_sz[1]));
      }
    }

    /* shift B */
    for (i = 0; i < 2; i++){
      if (flags[i]){
        c = bz_idx(my_cols[i], grid_sz);
        /* use non-blocking isend to avoid deadlock by blocking sendrecv */
        isend(&new_comm, my_rank, local_B[i], max(K_sz[0], K_sz[1]) * N_sz[c], dest_B[i], &request[i]);
        if (comm_sz == 2){
          recv(&new_comm, my_rank, temp_B[1-i], max(K_sz[0], K_sz[1]) * N_sz[c], src_B[i]);
        } else {
          recv(&new_comm, my_rank, temp_B[i], max(K_sz[0], K_sz[1]) * N_sz[c], src_B[i]);
        }
      }
    }
    /* wait for send recv to finish */
    for (i = 0; i < 2; i++){
      if (flags[i]){
        wait(request[i]);
        c = bz_idx(my_cols[i], grid_sz);
        for (int mm = 0; mm < max(K_sz[0], K_sz[1]); mm++){
          for (int nn = 0; nn < N_sz[c]; nn++){
            local_B[i][mm*N_sz[c] + nn] = temp_B[i][mm*N_sz[c] + nn];
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
  for (i = 0; i < 2; i++){
    if (flags[i]){
      free(local_A[i]); 
      free(local_B[i]); 
      free(temp_B[i]);
      free(local_C[i]);
    }
  }
  return NULL;
}

int main(int argc, char* argv[]){
  if (argc != 3){
    printf("./pth_cannon_2 <num_threads> <input_filename>\n");
    exit(-1);
  }
  comm_sz = atoi(argv[1]);
  input_filename = argv[2];
  grid_sz = (int) ceil(sqrt((double) comm_sz));
  logical_p = grid_sz * grid_sz;
  pthread_t tid[comm_sz];

  /* build communicator */
  initComm(&new_comm, comm_sz);

  /* create threads */
  for (long rank = 0; rank < comm_sz; rank++) {
    pthread_create(&tid[rank], NULL, thread_func, (void *)rank);
  }

  /* join threads */
  for (int rank = 0; rank < comm_sz; rank++) {
    pthread_join(tid[rank], NULL);
  }

  /* cleanup */
  destComm(&new_comm);
  return 0;
}


/* -------------------------------------*/
void initComm(comm_t *c, int n){
  c->N = n;
  c->nodes = malloc(sizeof(comm_node) * n);
  pthread_barrier_init(&c->b, NULL, n);
  pthread_mutex_init(&c->mutex, NULL);
  for (int i = 0; i < n; i++){
    c->nodes[i].ptp_addrs = malloc(sizeof (int*) * n);
    c->nodes[i].ptp_readys = malloc(sizeof(sem_t) * n);
    c->nodes[i].ptp_waits = malloc(sizeof(sem_t) * n);
    for (int j = 0; j < n; j++){
      sem_init(&c->nodes[i].ptp_readys[j], 0, 0);
      sem_init(&c->nodes[i].ptp_waits[j], 0, 0);
    }
    sem_init(&c->nodes[i].ready, 0, 0);
    sem_init(&c->nodes[i].wait, 0, 0);
  }
}

void destComm(comm_t *c){
  for (int i = 0; i < c->N; i++){
    sem_destroy(&c->nodes[i].ready);
    sem_destroy(&c->nodes[i].wait);
    for (int j = 0; j < c->N; j++){
      sem_destroy(&c->nodes[i].ptp_readys[j]);
      sem_destroy(&c->nodes[i].ptp_waits[j]);
    }
    free(c->nodes[i].ptp_addrs);
    free(c->nodes[i].ptp_readys);
    free(c->nodes[i].ptp_waits);
  }
  pthread_barrier_destroy(&c->b);
  pthread_mutex_destroy(&c->mutex);
  free(c->nodes);
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

/**
 * @brief 
 * use Pthread (simplified) version of alltoallw to distribute submatrices of A and B
 * cyclic mapping, hence twice its usage in the worst case
 */
void distributeBlocksP0(int my_rank, int comm_sz, int grid_sz, int* my_rows, int* my_cols, int* MKN, int** MKN_sz, sub_array** ABC_dt, int** A, int** B, int** local_A, int** local_B, int* flags){
  int i, j, k, rr, cc, n, m, r, c, shifted_coor, shifted_k;
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
      shifted_coor = (my_rows[i]+ my_cols[i]) % grid_sz;
      shifted_k = bz_idx(shifted_coor, grid_sz);
      r = bz_idx(my_rows[i], grid_sz);
      c = bz_idx(my_cols[i], grid_sz);
      counts_recv[0][0] = MKN_sz[0][r] * MKN_sz[1][shifted_k]; // A
      counts_recv[1][0] = MKN_sz[1][shifted_k] * MKN_sz[2][c]; // B
    }
    if (my_rank == 0){
      int logical_p = grid_sz * grid_sz;
      for (int rank = 0; rank < comm_sz; rank++){
        if (rank + i*comm_sz < logical_p){
          rr = (rank + i*comm_sz) / grid_sz;
          cc = (rank + i*comm_sz) % grid_sz;
          shifted_coor = (rr + cc) % grid_sz;
          shifted_k = bz_idx(shifted_coor, grid_sz);
          m = bz_idx(rr, grid_sz);
          n = bz_idx(cc, grid_sz);
          counts_send[0][rank] = counts_send[1][rank] = 1;
          types_send[0][rank] = &ABC_dt[0][m*2+shifted_k];
          types_send[1][rank] = &ABC_dt[1][shifted_k*2+n];
          dis_send[0][rank] = rr * MKN_sz[0][0] * MKN[1] + shifted_coor * MKN_sz[1][0];
          dis_send[1][rank] = shifted_coor * MKN_sz[1][0] * MKN[2] + cc * MKN_sz[2][0];
        }
      }
    }
    alltoallw_scatter(&new_comm, my_rank, 0, counts_send[0], dis_send[0], types_send[0], &A[0][0], local_A[i], counts_recv[0], dis_recv[0], types_recv[0]);
    alltoallw_scatter(&new_comm, my_rank, 0, counts_send[1], dis_send[1], types_send[1], &B[0][0], local_B[i], counts_recv[1], dis_recv[1], types_recv[1]);
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
      counts_send[0] = MKN_sz[0][r] * MKN_sz[2][c];
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
    alltoallw_gather(&new_comm, my_rank, 0, counts_send, dis_send, types_send, local_C[i], &C[0][0], counts_recv, dis_recv, types_recv);
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
          buf_recv[disp_recv[0] + i + j*types_recv[0]->array_j + k] = my_buf_recv[count++];
        }
      }
    }
  } else {
    /* sender : receive from source memory */
    for (i = 0; i < counts_recv[0]; i++){ // recv int
      for (j = 0; j < types_recv[0]->block_i; j++){
        for (k = 0; k < types_recv[0]->block_j; k++){
          buf_recv[disp_recv[0] + i + j*types_recv[0]->array_j + k] = buf_send[disp_send[0] + 
          (i/types_send[0]->block_j) *types_send[0]->array_j + (i % types_send[0]->block_j)];
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
    for (i = 0; i < counts_send[0]; i++){
      for (j = 0; j < types_send[0]->block_i; j++){
        for (k = 0; k < types_send[0]->block_j; k++){
          buf_recv[disp_recv[0] + (i/types_recv[0]->block_j) *(types_recv[0]->array_j) + i%types_recv[0]->block_j] = buf_send[disp_send[0] + i + j*(types_send[0]->array_j) + k];
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

/**
 * @brief 
 * MPI-equivalent sendrecv_replace
 * use buffer to store data to be sent
 */
void sendrecv_replace(comm_t *c, int my_rank, int global_rank, int dest, int src, int* buf, int count){
  /* sender : buffer data */
  int * inter_buf = malloc(sizeof(int) * count);
  for (int i = 0; i < count; i++){
    inter_buf[i] = buf[i];
  }
  c->nodes[dest].ptp_addrs[my_rank] = inter_buf;
  sem_post(&c->nodes[dest].ptp_readys[my_rank]);
  /* receive */
  sem_wait(&c->nodes[my_rank].ptp_readys[src]);
  int* inter_recv = c->nodes[my_rank].ptp_addrs[src];
  for (int i = 0; i < count; i++){
    buf[i] = inter_recv[i];
  }
  sem_post(&c->nodes[my_rank].ptp_waits[src]);
  sem_wait(&c->nodes[dest].ptp_waits[my_rank]);
  free(inter_buf);
}

/* -------------------------------------*/
/**
 * @brief 
 * non-blocking send
 */
void isend(comm_t* c, int my_rank, int* buf, int count, int dest, sem_t** request){
  c->nodes[dest].ptp_addrs[my_rank] = buf;
  sem_post(&c->nodes[dest].ptp_readys[my_rank]);
  *request = &c->nodes[dest].ptp_waits[my_rank];
}

/**
 * @brief 
 * associated wait operation for non-blocking send
 */
void wait(sem_t* request){
  sem_wait(request);
}
/* -------------------------------------*/
/**
 * @brief 
 * MPI-equivalent blocking receive
 */
void recv(comm_t* c, int my_rank, int* buf, int count, int src){
  sem_wait(&c->nodes[my_rank].ptp_readys[src]);
  int* inter_recv = c->nodes[my_rank].ptp_addrs[src];
  for (int i = 0; i < count; i++){
    buf[i] = inter_recv[i];
  }
  sem_post(&c->nodes[my_rank].ptp_waits[src]);
}