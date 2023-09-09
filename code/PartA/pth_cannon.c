/**
 * @file pth_cannon.c
 * @brief 
 * square thread layout -- "round-down" approach
 * 
 * O(logp) scatter-gather, broadcast operations implemented with sempahores and barriers
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <semaphore.h>
#include <math.h>
#include "../util/utils.h"
#include "../util/timer.h"

#define bz_idx(i, grid_sz) (i == grid_sz - 1 ? 1 : 0)
#define max(a, b) (a > b? a : b)

/* type definition */
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

// simulate MPI "vector" type
typedef struct vec {
  int block_count;
  int block_length;
  int stride;
} vec;

typedef struct comm_t {
  int N;
  comm_node* nodes;
  int* counts;
  vec *send_type;
  vec *recv_type;
  pthread_barrier_t b;
} comm_t;
vec INT_TYPE = {1,1,1};

/* functions */
void buildVectorTypes(int my_rank, int* MKN, int grid_sz, int* my_coor, int** MKN_sz, vec** ABC_vec);
void initComm(comm_t *c, int n);
void destComm(comm_t *c);
void gatherv(comm_t *c, int my_rank, int global_rank, int receiver, int counts_recv[], int disp[], int* buf_send, int* buf_recv, int count_send, vec *send_type, vec* recv_type);
void scatterv(comm_t *c, int my_rank, int row, int sender, int counts_send[], int disp[], int* buf_send, int* buf_recv, int count_recv, vec *send_type, vec *recv_type);
void broadcast(comm_t *c, int sender, int my_rank, int msg_sz, int* comm_addr);
void sendrecv_replace(comm_t *c, int my_rank, int global_rank, int dest, int src, int* buf, int count);

/* shared communicator -- simulate MPI runtime */
comm_t new_comm;
comm_t *row_comms, *col_comms;
/* shared program input */
int comm_sz, grid_sz, logical_p;
char* input_filename;

/* simulate MPI program */
void *thread_func(void *arg){
  int my_rank = (long) arg;
  int my_coor[2] = {my_rank / grid_sz, my_rank % grid_sz};
  comm_t *row_comm = &row_comms[my_coor[0]];
  comm_t *col_comm = &col_comms[my_coor[1]];
  int **A, **B, **C, *oracle;
  int MKN[3];
  int i, j, k, r, c;
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
  broadcast(&new_comm, 0, my_rank, 3, MKN);
  /* determine block sizes and construct vector types */
  int M_sz[2], K_sz[2], N_sz[2];
  int* MKN_sz[3] = {M_sz, K_sz, N_sz};
  vec A_vec[2], B_vec[2], C_vec[2]; // 0: full row vec, 1: block vec
  vec* ABC_vec[3] = {A_vec, B_vec, C_vec};
  buildVectorTypes(my_rank, MKN, grid_sz, my_coor, MKN_sz, ABC_vec);

  /* performance measurement */
  double start, end;
  GET_TIME(start);

  /* ----------------------------------- */
  /* 2 phase scatter of A and B */
  int counts_send[2][grid_sz], disp[2][grid_sz];
  int* full_rows[2] = {NULL, NULL};// 0: A; 1: B
  r = bz_idx(my_coor[0], grid_sz);
  c = bz_idx(my_coor[1], grid_sz);
  // phase 1 : distribute full rows in P0 column communicator
  if (my_coor[1] == 0){
    full_rows[0] = malloc(sizeof(int) * M_sz[r] * MKN[1]);
    full_rows[1] = malloc(sizeof(int) * K_sz[r] * MKN[2]);
    for (i = 0; i < grid_sz; i++){
      if (i != grid_sz-1){
        counts_send[0][i] = M_sz[0]*MKN[1];
        counts_send[1][i] = K_sz[0]*MKN[2];
      }else{
        counts_send[0][i] = M_sz[1]*MKN[1];
        counts_send[1][i] = K_sz[1]*MKN[2];
      }
      disp[0][i] = i*M_sz[0]*MKN[1];
      disp[1][i] = i*K_sz[0]*MKN[2];
    }
    scatterv(col_comm, my_coor[0], 0, 0, counts_send[0], disp[0], &A[0][0], full_rows[0], M_sz[r]* MKN[1], &INT_TYPE, &INT_TYPE);
    scatterv(col_comm, my_coor[0], 0, 0, counts_send[1], disp[1], &B[0][0], full_rows[1], K_sz[r]* MKN[2], &INT_TYPE, &INT_TYPE);
  }

  int* local_A, *local_B, *local_C;
  local_A = malloc(sizeof(int) * M_sz[r] * max(K_sz[0], K_sz[1]));
  local_B = malloc(sizeof(int) * max(K_sz[0], K_sz[1]) * N_sz[c]);
  local_C = malloc(sizeof(int) * M_sz[r] * N_sz[c]);
  memset(local_C, 0, sizeof(int) * M_sz[r] * N_sz[c]);

  // phase 2 : partition full rows into submatrices and distribute in row communicators
  // integrated initial shift of A
  int shifted_coor = (my_coor[1]+ my_coor[0]) % grid_sz;
  int shifted_k = bz_idx(shifted_coor, grid_sz);
  for (j = 0 ; j < grid_sz; j++){
    int shifted = (j + my_coor[0]) % grid_sz;
    if (shifted != grid_sz - 1){
      counts_send[0][j] = K_sz[0];
    }else{
      counts_send[0][j] = K_sz[1];
    }
    disp[0][j] = shifted * K_sz[0];
    if (j != grid_sz - 1){
      counts_send[1][j] = N_sz[0];
    }else {
      counts_send[1][j] = N_sz[1];
    }
    disp[1][j] = j * N_sz[0];
  }

  scatterv(row_comm, my_coor[1], 1, 0, counts_send[0], disp[0], full_rows[0], local_A, K_sz[shifted_k], &A_vec[0], &A_vec[1]);
  scatterv(row_comm, my_coor[1], 1, 0, counts_send[1], disp[1], full_rows[1], local_B, N_sz[c], &B_vec[0], &B_vec[1]);

  // separate initial shift of B 
  int dest_r, src_r;
  src_r = shifted_coor;
  dest_r = (my_coor[0] + grid_sz - my_coor[1]) % grid_sz;
  sendrecv_replace(col_comm, my_coor[0], my_rank, dest_r, src_r, local_B, max(K_sz[0], K_sz[1]) * N_sz[c]);

  /* ----------------------------------- */
  /* Cannon algorithm */
  int mm, nn, kk;
  int left, right, up, down;
  left = (my_coor[1] + grid_sz - 1) % grid_sz;
  right = (my_coor[1] + 1) % grid_sz;
  up = (my_coor[0] + grid_sz - 1) % grid_sz;
  down = (my_coor[0] + 1) % grid_sz;
  mm = M_sz[r]; nn = N_sz[c];
  for (k = 0; k < grid_sz; k++){
    kk = K_sz[shifted_k];
    for (int kkk = 0; kkk < kk; kkk++){
      for (i = 0; i < mm; i++){
        for (j = 0; j < nn; j++){
          local_C[i*nn + j] += local_A[i*kk + kkk] * local_B[kkk*nn + j];
        }
      }
    }
    /* shift A and B */
    sendrecv_replace(row_comm, my_coor[1], my_rank, left, right, local_A, M_sz[r] * max(K_sz[0], K_sz[1]));
    sendrecv_replace(col_comm, my_coor[0], my_rank, up, down, local_B, max(K_sz[0], K_sz[1]) * N_sz[c]);
    shifted_coor = (shifted_coor + 1) % grid_sz;
    shifted_k = bz_idx(shifted_coor, grid_sz);
  }

  /* ----------------------------------- */
  /* collect results of C with 2-phase gather */
  int* full_rows_C = NULL;
  int counts_recv[grid_sz], disp_recv[grid_sz];
  if (my_coor[1] == 0) {
    for (j = 0 ; j < grid_sz; j++){
      if (j != grid_sz - 1){
        counts_recv[j] = N_sz[0];
      }else {
        counts_recv[j] = N_sz[1];
      }
      disp_recv[j] = j * N_sz[0];
    }
    full_rows_C = malloc(sizeof(int) * M_sz[r] * MKN[2]);
  }
  
  gatherv(row_comm, my_coor[1], my_rank, 0, counts_recv, disp_recv, local_C, full_rows_C, N_sz[c], &C_vec[1], &C_vec[0]);
  
  if (my_coor[1] == 0) {
    for (j = 0 ; j < grid_sz; j++){
      if (j != grid_sz - 1){
        counts_recv[j] = M_sz[0] * MKN[2];
      }else {
        counts_recv[j] = M_sz[1] * MKN[2];
      }
      disp_recv[j] = j * M_sz[0] * MKN[2];
    }
    gatherv(col_comm, my_coor[0], my_rank, 0, counts_recv, disp_recv, full_rows_C, &C[0][0], M_sz[r] * MKN[2], &INT_TYPE, &INT_TYPE);
    free(full_rows_C);
  }
  /* correctness checking and timing output */
  GET_TIME(end);
  if (my_rank ==0){
    checkCorrectness(MKN[0], MKN[2], C, oracle);
    printf("computed in %.8fs\n", end-start);
  }

  free(local_A); free(local_B); free(local_C);
  free(A); free(B); free(C); free(oracle);
  return NULL;
}

int main(int argc, char* argv[]){
  if (argc != 3){
    printf("./pth_cannon <num_threads> <input_filename>\n");
    exit(-1);
  }
  comm_sz = atoi(argv[1]);
  input_filename = argv[2];
  grid_sz = (int) floor(sqrt((double) comm_sz));
  logical_p = grid_sz * grid_sz;
  pthread_t tid[logical_p];

  /* build communicators */
  initComm(&new_comm, logical_p);
  row_comms = malloc(sizeof(comm_t) * grid_sz);
  col_comms = malloc(sizeof(comm_t) * grid_sz);
  for (int i = 0; i < grid_sz; i++){
    initComm(&row_comms[i], grid_sz);
    initComm(&col_comms[i], grid_sz);
  }

  /* create threads */
  for (long rank = 0; rank < logical_p; rank++) {
    pthread_create(&tid[rank], NULL, thread_func, (void *)rank);
  }

  /* join threads */
  for (int rank = 0; rank < logical_p; rank++) {
    pthread_join(tid[rank], NULL);
  }

  /* cleanup */
  for (int i = 0; i < grid_sz; i++){
    destComm(&row_comms[i]);
    destComm(&col_comms[i]);
  }
  free(row_comms);
  free(col_comms);
  return 0;
}

/* -------------------------------------*/
void initComm(comm_t *c, int n){
  c->N = n;
  c->nodes = malloc(sizeof(comm_node) * n);
  pthread_barrier_init(&c->b, NULL, n);
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
  free(c->nodes);
}
/* -------------------------------------*/
/**
 * @brief 
 * build MPI-equivalent vector types for 2 phase scatter-gather
 */
void buildVectorTypes(int my_rank, int* MKN, int grid_sz, int* my_coor, int** MKN_sz, vec** ABC_vec){
  int i, j, k;
  for (i = 0; i < 3; i++){
    MKN_sz[i][0] = floor((double) MKN[i] / (double) grid_sz);
    MKN_sz[i][1] = MKN[i] - MKN_sz[i][0] * (grid_sz-1);
  }
  struct dim {
    int r, c;
  };
  struct dim ABC_dim[3] = {{0, 1}, {1, 2}, {0, 2}}; // M x K, K x N, M x N
  i = bz_idx(my_coor[0], grid_sz);
  j = bz_idx(my_coor[1], grid_sz);
  int shifted_coor = (my_coor[1]+ my_coor[0]) % grid_sz;
  int shifted_j = bz_idx(shifted_coor, grid_sz);
  int indices[3] = {shifted_j, j, j};
  for (k = 0 ; k < 3; k++){
    // full row vec type
    ABC_vec[k][0] = (vec) {.block_count= MKN_sz[ABC_dim[k].r][i],.block_length= 1,.stride= MKN[ABC_dim[k].c]};
    // submatrix vec type
    ABC_vec[k][1] = (vec) {.block_count= MKN_sz[ABC_dim[k].r][i],.block_length= 1,.stride= MKN_sz[ABC_dim[k].c][indices[k]]};
  }
}

/**
 * @brief 
 * O(logp) tree-structured blocking gather operation implemented with sempahores and barrier
 * processes at intermediate levels send combined result to partner
 */
void gatherv(comm_t *c, int my_rank, int global_rank, int receiver, int counts_recv[], int disp[], int* buf_send, int* buf_recv, int count_send, vec *send_type, vec* recv_type){
  if (my_rank == receiver){
    c->counts = counts_recv;
    c->recv_type = recv_type;
  }
  /* use rank 0 as the receiver */
  if (my_rank == 0 && my_rank != receiver){
    my_rank = receiver;
  } else if (my_rank == receiver && my_rank != 0){
    int temp = c->counts[receiver];
    c->counts[receiver] = c->counts[0];
    c->counts[0] = temp;
    temp = disp[receiver];
    disp[receiver] = disp[0];
    disp[0] = temp;
    my_rank = 0;
  }
  pthread_barrier_wait(&c->b);
  int partner, i, j, k, n, m, count, disp_j;
  int* recv_buf = NULL;
  /* t = 0 */
  if (my_rank != 0){
    /* preprocess: copy non-contiguous data into contiguous memory */
    count = count_send * c->recv_type->block_count * c->recv_type->block_length;
    recv_buf = malloc(sizeof(int) * count);
    for (i = 0, count =0; i < count_send; i++){
      for (j = 0; j < send_type->block_count; j++){
        for (k = 0; k < send_type->block_length; k++){
          recv_buf[count++] = buf_send[i*(send_type->block_length)+ j*(send_type->stride) +k];
        }
      }
    }
  } else {
    /* receiver: copy its own data into destination memory */
    for (i = 0; i < counts_recv[0]; i++){
      for (j = 0; j < recv_type->block_count; j++){
        for (k = 0; k < recv_type->block_length; k++){
          buf_recv[disp[0] + i*(recv_type->block_length)+ j*(recv_type->stride) + k] = buf_send[i*(send_type->block_length) + j*(send_type->stride) + k];
        }
      }
    }
  }
  /* tree-structured gather */
  for (i = 1; i < c->N; i<<=1){
    partner = my_rank ^ i;
    if (my_rank % i == 0 && partner < my_rank) {
      /* sender */
      c->nodes[my_rank].comm_addr = recv_buf;
      sem_post(&c->nodes[my_rank].ready);
      sem_wait(&c->nodes[my_rank].wait);
      break;
    }else if (my_rank % i == 0 && partner < c->N){
      /* receiver */
      if (my_rank != 0){
        /* intermediate nodes : combine received data */
        count = disp_j = 0;
        for (j = my_rank; j < partner + i; j++){
          if (j == partner){
            disp_j = count;
          }
          if (j < c->N){
            count += c->counts[j];
          }
        }
        count *= (c->recv_type->block_count) * (c->recv_type->block_length);
        disp_j *= (c->recv_type->block_count) * (c->recv_type->block_length);
        recv_buf = realloc(recv_buf, sizeof(int) * count);
        sem_wait(&c->nodes[partner].ready);
        int* inter_recv = c->nodes[partner].comm_addr;
        for (j = disp_j; j < count; j++){
          recv_buf[j] = inter_recv[j-disp_j];
        }
        sem_post(&c->nodes[partner].wait);
      } else {
        /* destination receiver : receive in correct destination memory */
        sem_wait(&c->nodes[partner].ready);
        int* inter_recv = c->nodes[partner].comm_addr;
        count = 0;
        for (j = partner; j < partner + i; j++){
          if (j < c->N){
            disp_j = disp[j];
            for (k = 0; k < counts_recv[j]; k++){
              for (n = 0; n < recv_type->block_count; n++){
                for (m = 0; m < recv_type->block_length; m++){
                  buf_recv[disp_j+ k*(recv_type->block_length) + n*(recv_type->stride) + m] = inter_recv[count++];
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
 * O(logp) tree-structured blocking scatter operation implemented with sempahores and barrier
 * processes at intermediate levels send combined result to partner
 */
void scatterv(comm_t *c, int my_rank, int row, int sender, int counts_send[], int disp[], int* buf_send, int* buf_recv, int count_recv, vec *send_type, vec *recv_type){
  if (my_rank == sender){
    c->counts = counts_send;
    c->send_type = send_type;
  }
  /* use P0 as the sender */
  if (my_rank == 0 && my_rank != sender){ 
    my_rank = sender;
  } else if (my_rank == sender && my_rank != 0){
    int temp = c->counts[sender];
    c->counts[sender] = c->counts[0];
    c->counts[0] = temp;
    temp = disp[sender];
    disp[sender] = disp[0];
    disp[0] = temp;
    my_rank = 0;
  }
  pthread_barrier_wait(&c->b);
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
          count += c->counts[j];
        }
      }
      count *= (c->send_type->block_count) * (c->send_type->block_length);
      int* inter_buf = malloc(sizeof(int) * count);

      if (my_rank == 0){
        /* sender copies from source memory */
        count = 0;
        for (j = partner; j < partner + i; j++){
          if (j < c->N){
            disp_j = disp[j];
            for (k = 0; k < c->counts[j]; k++){
              for (n = 0; n < send_type->block_count; n++){
                for (m = 0; m < send_type->block_length; m++){
                  inter_buf[count++] = buf_send[disp_j+ k*(send_type->block_length) + n*(send_type->stride) + m];
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
          disp_j += c->counts[j];
        }
        disp_j *= (c->send_type->block_count)*(c->send_type->block_length);
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
    for (i = 0; i < count_recv; i++){
      for (j = 0; j < recv_type->block_count; j++){
        for (k = 0; k < recv_type->block_length; k++){
          buf_recv[i*(recv_type->block_length)+ j*(recv_type->stride) +k] = my_buf_recv[count++];
        }
      }
    }
  } else {
    /* sender : receive from source memory */
    for (i = 0; i < count_recv; i++){
      for (j = 0; j < recv_type->block_count; j++){
        for (k = 0; k < recv_type->block_length; k++){
          buf_recv[i*(recv_type->block_length)+ j*(recv_type->stride) +k] = buf_send[disp[0] + i*(send_type->block_length) + j*(send_type->stride) + k];
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
 * O(logp) tree-structured blocking broadcast implemented with semaphores and barrier
 */
void broadcast(comm_t *c, int sender, int my_rank, int msg_sz, int* comm_addr){
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
        dest[j] = src[j];
      }
      sem_post(&c->nodes[partner].wait);
    }
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