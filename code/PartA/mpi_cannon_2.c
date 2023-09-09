/**
 * @file mpi_cannon_2.c
 * @brief 
 * Square processor layout -- "round-up" approach with cyclic mapping
 * Each process mapped to at most 2 blocks
 * 
 * Use non blocking MPI_Isend to prevent deadlock from data "shift"s within columns
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <string.h>
#include "../util/utils.h"

#define bz_idx(i, grid_sz) (i == grid_sz - 1 ? 1 : 0)
#define max(a, b) (a > b? a : b)
void buildSubarrayTypes(int* MKN, int proc_grid_sz, int** MKN_sz, MPI_Datatype** ABC_dt);
void distributeBlocksP0(int my_rank, int comm_sz, int grid_sz, int* my_rows, int* my_cols, int* MKN, int** MKN_sz, MPI_Datatype** ABC_dt, int** A, int** B, int** local_A, int** local_B, int* flags);
void gatherResultsP0(int my_rank, int comm_sz, int grid_sz, int* my_rows, int* my_cols, int* MKN, int** MKN_sz, MPI_Datatype* C_dt, int** C, int** local_C, int* flags);

int main(int argc, char *argv[]){
  int** A, **B, **C, *oracle;
  int MKN[3];
  int i, k;
  int my_rank, comm_sz, grid_sz, logical_p;
  int r, c, mm, nn;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  if (argc != 2){
      printf("./mpi_cannon_2 <input_filename>\n");
      exit(-1);
  }
  /* round-up process grid */
  grid_sz = (int) ceil(sqrt((double) comm_sz));
  logical_p = grid_sz * grid_sz;

  /* Read inputs from file */
  if (my_rank == 0){
    mpi_readInputC(argv[1], MKN, &A, &B, &C, &oracle);
    if (grid_sz > MKN[0] || grid_sz > MKN[1] || grid_sz > MKN[2]){
      printf("matrix dimensions >= process grid size\n");
      exit(-1);
    }
  } else {
    mallocCont(&A, 1, 1);
    mallocCont(&B, 1, 1);
    mallocCont(&C, 1, 1);
    oracle = malloc(sizeof(int));
  }
  MPI_Bcast(MKN, 3, MPI_INT, 0, MPI_COMM_WORLD);

  /* determine block sizes and construct subarray types */
  int M_sz[2], K_sz[2], N_sz[2];
  int* MKN_sz[3] = {M_sz, K_sz, N_sz};
  MPI_Datatype A_dt[4], B_dt[4], C_dt[4];
  MPI_Datatype* ABC_dt[3] = {A_dt, B_dt, C_dt}; 
  buildSubarrayTypes(MKN, grid_sz, MKN_sz, ABC_dt);

  /* allocate local memory and identify neighbours */
  /* cyclic mapping : each processor will be assigned at most two blocks*/
  int flags[2] = {0, 0};
  int my_rows[2] = {-1, -1}, my_cols[2] = {-1, -1};
  int dest_A[2], dest_B[2];
  int src_A[2], src_B[2];
  int* local_A[2], *local_B[2], *temp_B[2], *local_C[2];
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
      memset(local_C[i], 0, sizeof(int) * M_sz[r] * N_sz[c]);
    }
  }

  /* performance measurement */
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();

  distributeBlocksP0(my_rank, comm_sz, grid_sz, my_rows, my_cols, MKN, MKN_sz, ABC_dt, A, B, local_A, local_B, flags);
  int shifted_coor, shifted_k;

  /* Cannon algorithm */
  int kk, ii, jj;
  MPI_Request request[2];
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
        MPI_Sendrecv_replace(local_A[i], M_sz[r] * max(K_sz[0], K_sz[1]), MPI_INT, dest_A[i], 0, src_A[i], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }

    /* shift B */
    for (i = 0; i < 2; i++){
      if (flags[i]){
        c = bz_idx(my_cols[i], grid_sz);
        /* use non-blocking isend to avoid deadlock by blocking sendrecv */
        MPI_Isend(local_B[i], max(K_sz[0], K_sz[1]) * N_sz[c], MPI_INT, dest_B[i], 1, MPI_COMM_WORLD, &request[i]);
        if (comm_sz == 2){
          MPI_Recv(temp_B[1-i], max(K_sz[0], K_sz[1]) * N_sz[c], MPI_INT, src_B[i], 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
          MPI_Recv(temp_B[i], max(K_sz[0], K_sz[1]) * N_sz[c], MPI_INT, src_B[i], 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
      }
    }

    /* wait for send recv to finish */
    for (i = 0; i < 2; i++){
      if (flags[i]){
        MPI_Wait(&request[i], MPI_STATUS_IGNORE);
        c = bz_idx(my_cols[i], grid_sz);
        for (mm = 0; mm < max(K_sz[0], K_sz[1]); mm++){
          for (nn = 0; nn < N_sz[c]; nn++){
            local_B[i][mm*N_sz[c] + nn] = temp_B[i][mm*N_sz[c] + nn];
          }
        }
      }
    }
  }

  gatherResultsP0(my_rank, comm_sz, grid_sz, my_rows, my_cols, MKN, MKN_sz, C_dt, C, local_C, flags);
 
  /* correctness checking and timing output */
  double end = MPI_Wtime();
  if (my_rank == 0){
    checkCorrectness(MKN[0], MKN[2], C, oracle);
    printf("Computed in %0.8fs\n", end-start);
  }

  MPI_Finalize();
  return 0;

}

/**
 * @brief 
 * build MPI "subarray" data types for each submatrix size
 */
void buildSubarrayTypes(int* MKN, int proc_grid_sz, int** MKN_sz, MPI_Datatype** ABC_dt){
  int i, j, k;
  /* partition matrices into sizes as balanced as possible */
  for (i = 0; i < 3; i++){
    MKN_sz[i][0] = floor((double) MKN[i] / (double) proc_grid_sz);
    MKN_sz[i][1] = MKN[i] - MKN_sz[i][0] * (proc_grid_sz-1);
  }
  int array_sz[2], array_subsize[2], array_start[2] = {0, 0};
  struct dim {
    int r, c;
  };
  struct dim ABC_dim[3] = {{0, 1}, {1, 2}, {0, 2}}; // M x K, K x N, M x N

  /* build 4 submatrix data types for each of A,B,C */
  for (k = 0; k < 3; k++){ 
    array_sz[0] = MKN[ABC_dim[k].r];
    array_sz[1] = MKN[ABC_dim[k].c];
    for (i = 0; i < 2; i++){
      array_subsize[0] = MKN_sz[ABC_dim[k].r][i];
      for (j = 0; j < 2; j++){
        array_subsize[1] = MKN_sz[ABC_dim[k].c][j];
        MPI_Type_create_subarray(2, array_sz, array_subsize, array_start, MPI_ORDER_C, MPI_INT, &ABC_dt[k][i*2+j]);
        MPI_Type_commit(&ABC_dt[k][i*2+j]);
      }
    }
  }
}

/**
 * @brief 
 * use MPI_Alltoallw to distribute submatrices of A and B
 * cyclic mapping, hence twice its usage in the worst case
 */
void distributeBlocksP0(int my_rank, int comm_sz, int grid_sz, int* my_rows, int* my_cols, int* MKN, int** MKN_sz, MPI_Datatype** ABC_dt, int** A, int** B, int** local_A, int** local_B, int* flags){
  int i, j, k, rr, cc, n, m, r, c;
  int shifted_coor, shifted_k;
  int counts_send[2][comm_sz], counts_recv[2][comm_sz];
  int dis_send[2][comm_sz], dis_recv[2][comm_sz];
  MPI_Datatype types_send[2][comm_sz], types_recv[2][comm_sz];
  
  for (i = 0; i < 2; i++){
    /* all processes (incl P0 from itself) receive */
    for (j = 0; j < comm_sz; j++){
      for (k = 0; k < 2; k++){
        types_recv[k][j] = types_send[k][j] = MPI_INT;
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
    /* P0 send */
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
          types_send[0][rank] = ABC_dt[0][m*2+shifted_k];
          types_send[1][rank] = ABC_dt[1][shifted_k*2+n];
          dis_send[0][rank] = sizeof(int) * (rr * MKN_sz[0][0] * MKN[1] + shifted_coor * MKN_sz[1][0]);
          dis_send[1][rank] = sizeof(int) * (shifted_coor * MKN_sz[1][0] * MKN[2] + cc * MKN_sz[2][0]);
        }
      }
    }

    MPI_Alltoallw(&A[0][0], counts_send[0], dis_send[0], types_send[0], local_A[i], counts_recv[0], dis_recv[0], types_recv[0], MPI_COMM_WORLD);
    MPI_Alltoallw(&B[0][0], counts_send[1], dis_send[1], types_send[1], local_B[i], counts_recv[1], dis_recv[1], types_recv[1], MPI_COMM_WORLD);
  }
}

/**
 * @brief 
 * use MPI_Alltoallw to collect submatrices of C
 * cyclic mapping, hence twice its usage in the worst case
 */
void gatherResultsP0(int my_rank, int comm_sz, int grid_sz, int* my_rows, int* my_cols, int* MKN, int** MKN_sz, MPI_Datatype* C_dt, int** C, int** local_C, int* flags){
  int i, j, r, c, rr, cc, m, n;
  int counts_send[comm_sz], counts_recv[comm_sz];
  int dis_send[comm_sz], dis_recv[comm_sz];
  MPI_Datatype types_send[comm_sz], types_recv[comm_sz];
  
  for (i = 0; i < 2; i++){
    /* all processes (incl P0 to itself) send */
    for (j = 0; j < comm_sz; j++){
      types_recv[j] = types_send[j] = MPI_INT;
      counts_recv[j] = counts_send[j] = 0;
      dis_recv[j] = dis_send[j] = 0;
    }
    if (flags[i]){
      r = bz_idx(my_rows[i], grid_sz);
      c = bz_idx(my_cols[i], grid_sz);
      counts_send[0] = MKN_sz[0][r] * MKN_sz[2][c];
    }
    /* P0 receive */
    if (my_rank == 0){
      int logical_p = grid_sz * grid_sz;
      for (int rank = 0; rank < comm_sz; rank++){
        if (rank + i*comm_sz < logical_p){
          rr = (rank + i*comm_sz) / grid_sz;
          cc = (rank + i*comm_sz) % grid_sz;
          m = bz_idx(rr, grid_sz);
          n = bz_idx(cc, grid_sz);
          counts_recv[rank] = 1;
          types_recv[rank] = C_dt[m*2+n];
          dis_recv[rank] = sizeof(int) * (rr * MKN_sz[0][0] * MKN[2] + cc * MKN_sz[2][0]);
        }
      }
    }
    MPI_Alltoallw(local_C[i], counts_send, dis_send, types_send, &C[0][0], counts_recv, dis_recv, types_recv, MPI_COMM_WORLD);
  }
}