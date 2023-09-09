/**
 * @file mpi_summa_2.c
 * @brief 
 * Square processor layout -- "round-up" approach with cyclic mapping
 * Each process mapped to at most 2 blocks
 * 
 * Build row and col communicators manually (native MPI_cart_create not applicable because grid_sz * grid_sz > comm_sz)
 * 
 * Use non-blocking MPI_Ibcast to prevent deadlock by in columns
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <string.h>
#include "../util/utils.h"


#define bz_idx(i, grid_sz) (i == grid_sz - 1 ? 1 : 0)
#define row(rank, grid_sz) (rank / grid_sz)
#define col(rank, grid_sz) (rank % grid_sz)

int cmpfunc (const void * a, const void * b) {
  return ( *(int*)a - *(int*)b );
}

void buildSubarrayTypes(int* MKN, int proc_grid_sz, int** MKN_sz, MPI_Datatype** ABC_dt);
void distributeBlocksP0(int my_rank, int comm_sz, int grid_sz, int* my_rows, int* my_cols, int* MKN, int** MKN_sz, MPI_Datatype** ABC_dt, int** A, int** B, int** local_A, int** local_B, int* flags);
void gatherResultsP0(int my_rank, int comm_sz, int grid_sz, int* my_rows, int* my_cols, int* MKN, int** MKN_sz, MPI_Datatype* C_dt, int** C, int** local_C, int* flags);
void buildComm(int comm_sz, int grid_sz, int* my_rows, int* my_cols, int (*row_group_mem)[grid_sz], int (*col_group_mem)[grid_sz], MPI_Comm* row_comms, MPI_Comm* col_comms, int (*row_ranks)[comm_sz], int (*col_ranks)[comm_sz], int* flags);

int main(int argc, char *argv[]){
  int** A, **B, **C, *oracle;
  int** local_A, **local_B, **local_C;
  int MKN[3];
  int i, k, r, c, m;
  int my_rank, comm_sz, grid_sz, logical_p;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  if (argc != 2){
    printf("./mpi_summa_2 <input_filename>\n");
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
  /* allocate local memory */
  mallocCont(&local_A, MKN[0], MKN[1]);
  mallocCont(&local_B, MKN[1], MKN[2]);
  mallocCont(&local_C, MKN[0], MKN[2]);
  memset(&local_C[0][0], 0, sizeof(int)* MKN[0]* MKN[2]);

  /* determine block sizes and construct subarray types */
  int M_sz[2], K_sz[2], N_sz[2];
  int* MKN_sz[3] = {M_sz, K_sz, N_sz};
  MPI_Datatype A_dt[4], B_dt[4], C_dt[4];
  MPI_Datatype* ABC_dt[3] = {A_dt, B_dt, C_dt}; 
  buildSubarrayTypes(MKN, grid_sz, MKN_sz, ABC_dt);

  /* cyclic mapping : each processor will be assigned at most two blocks*/
  int flags[2] = {0, 0};
  int my_rows[2] = {-1, -1}, my_cols[2] = {-1, -1};
  for (i = 0; i < 2; i++){
    if (my_rank + i*comm_sz < logical_p){
      flags[i] = 1;
      my_rows[i] = (my_rank + i*comm_sz) / grid_sz;
      my_cols[i] = (my_rank + i*comm_sz) % grid_sz;
    }
  }

  /* build row & col communicators */
  int row_group_mem[grid_sz][grid_sz], col_group_mem[grid_sz][grid_sz];
  MPI_Comm row_comms[grid_sz], col_comms[grid_sz];
  int row_ranks[2][comm_sz], col_ranks[2][comm_sz];
  buildComm(comm_sz, grid_sz, my_rows, my_cols, row_group_mem, col_group_mem, row_comms, col_comms, row_ranks, col_ranks, flags);

  /* performance measurement */
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();

  distributeBlocksP0(my_rank, comm_sz, grid_sz, my_rows, my_cols, MKN, MKN_sz, ABC_dt, A, B, local_A, local_B, flags);

  /* SUMMA algorithm */
  int sender_rank, rank_in_comm;
  int start_i, start_j, start_k, end_i, end_j, end_k;
  int kk, ii, jj;
  MPI_Request request[2];
  for (k = 0; k < grid_sz; k++){
    for (i = 0 ; i < 2; i++){
      if (flags[i]){
        // A row comm
        sender_rank = row_group_mem[my_rows[i]][k];
        rank_in_comm = row_ranks[i][sender_rank];
        start_i = my_rows[i] * M_sz[0];
        start_j = k * K_sz[0];
        r = bz_idx(my_rows[i], grid_sz);
        c = bz_idx(k, grid_sz);
        MPI_Bcast(&local_A[start_i][start_j], 1, A_dt[r*2+c], rank_in_comm, row_comms[my_rows[i]]);
      }
    }
    for (i = 0 ; i < 2; i++){
      if (flags[i]){
        // B col comm
        if (i == 1 && my_cols[0] == my_cols[1]) break;
        sender_rank = col_group_mem[my_cols[i]][k];
        rank_in_comm = col_ranks[i][sender_rank];
        start_i = k * K_sz[0];
        start_j = my_cols[i] * N_sz[0];
        r = bz_idx(k, grid_sz);
        c = bz_idx(my_cols[i], grid_sz);
        /* use non blocking Ibcast to prevent deadlock in column broadcast */
        MPI_Ibcast(&local_B[start_i][start_j], 1, B_dt[r*2+c], rank_in_comm, col_comms[my_cols[i]], &request[i]);
      }
    }
    /* wait for broadcast to finish */
    for (i = 0 ; i < 2; i++){
      if (flags[i]){
        if (i == 1 && my_cols[0] == my_cols[1]) break;
        MPI_Wait(&request[i], MPI_STATUS_IGNORE);
      }
    }

    for (i = 0; i < 2; i++){
      if (flags[i]){
        start_i = my_rows[i] * M_sz[0];
        start_j = my_cols[i] * N_sz[0];
        start_k = k * K_sz[0];
        m = bz_idx(k, grid_sz);
        r = bz_idx(my_rows[i], grid_sz);
        c = bz_idx(my_cols[i], grid_sz);
        end_i = M_sz[r]+start_i;
        end_j = N_sz[c]+start_j;
        end_k = K_sz[m]+start_k;
        for (kk = start_k; kk < end_k; kk++){
          for (ii = start_i; ii < end_i; ii++){
            for (jj = start_j; jj < end_j; jj++){
              local_C[ii][jj] += local_A[ii][kk] * local_B[kk][jj];
            }
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
  for (i = 0; i < 4; i++){
    MPI_Type_free(&A_dt[i]);
    MPI_Type_free(&B_dt[i]);
    MPI_Type_free(&C_dt[i]);
  }
  MPI_Finalize();
  free(local_A); free(local_B); free(local_C);
  free(A); free(B); free(C); free(oracle);
  return 0;
}

/**
 * @brief 
 * build MPI "subarray" data types for each submatrix size
 */
void buildSubarrayTypes(int* MKN, int proc_grid_sz, int** MKN_sz, MPI_Datatype** ABC_dt){
  int i, j, k;
  for (i = 0; i < 3; i++){
    MKN_sz[i][0] = floor((double) MKN[i] / (double) proc_grid_sz);
    MKN_sz[i][1] = MKN[i] - MKN_sz[i][0] * (proc_grid_sz-1);
  }
  int array_sz[2], array_subsize[2], array_start[2] = {0, 0};
  struct dim {
    int r, c; //row, col
  };
  struct dim ABC_dim[3] = {{0, 1}, {1, 2}, {0, 2}}; // M x K, K x N, M x N

  for (k = 0; k < 3; k++){ // loop over ABC
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
 * build row and column communicators without native MPI comm functions
 * 
 * Not counted towards performance measurement
 */
void buildComm(int comm_sz, int grid_sz, int* my_rows, int* my_cols, int (*row_group_mem)[grid_sz], int (*col_group_mem)[grid_sz], MPI_Comm* row_comms, MPI_Comm* col_comms, int (*row_ranks)[comm_sz], int (*col_ranks)[comm_sz], int* flags){
  int i, j;
  int row_sort[grid_sz], col_sort[grid_sz];
  int row_group_uniq_mem[grid_sz][grid_sz];
  int col_group_uniq_mem[grid_sz][grid_sz];
  int row_uniq_n[grid_sz], col_uniq_n[grid_sz];
  int world_ranks[comm_sz];
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  MPI_Group row_groups[grid_sz], col_groups[grid_sz];
  
  for (i =0 ; i<comm_sz; i++) world_ranks[i] = i;

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
    MPI_Group_incl(world_group, row_uniq_n[i], row_group_uniq_mem[i], &row_groups[i]);
    MPI_Group_incl(world_group, col_uniq_n[i], col_group_uniq_mem[i], &col_groups[i]);
    MPI_Comm_create(MPI_COMM_WORLD, row_groups[i], &row_comms[i]);
    MPI_Comm_create(MPI_COMM_WORLD, col_groups[i], &col_comms[i]);
  }
  for (i = 0; i <2; i++){
    if (flags[i]){
      MPI_Group_translate_ranks(world_group, comm_sz, world_ranks, row_groups[my_rows[i]], row_ranks[i]);
      MPI_Group_translate_ranks(world_group, comm_sz, world_ranks, col_groups[my_cols[i]], col_ranks[i]);
    }
  }
  MPI_Group_free(&world_group);
  for (i = 0; i < grid_sz; i++){
    MPI_Group_free(&row_groups[i]);
    MPI_Group_free(&col_groups[i]);
  }
}

/**
 * @brief 
 * use MPI_Alltoallw to distribute submatrices of A and B
 * cyclic mapping, hence twice its usage in the worst case
 */
void distributeBlocksP0(int my_rank, int comm_sz, int grid_sz, int* my_rows, int* my_cols, int* MKN, int** MKN_sz, MPI_Datatype** ABC_dt, int** A, int** B, int** local_A, int** local_B, int* flags){
  int i, j, k, rr, cc, n, m, r, c;
  int counts_send[2][comm_sz], counts_recv[2][comm_sz];
  int dis_send[2][comm_sz], dis_recv[2][comm_sz];
  MPI_Datatype types_send[2][comm_sz], types_recv[2][comm_sz];
  
  for (i = 0; i < 2; i++){
    for (j = 0; j < comm_sz; j++){
      for (k = 0; k < 2; k++){
        types_recv[k][j] = types_send[k][j] = MPI_INT;
        counts_recv[k][j] = counts_send[k][j] = 0;
        dis_recv[k][j] = dis_send[k][j] = 0;
      }
    }
    if (flags[i]){
      r = bz_idx(my_rows[i], grid_sz);
      c = bz_idx(my_cols[i], grid_sz);
      counts_recv[0][0] = 1; // A
      dis_recv[0][0] = sizeof(int) * (my_rows[i] * MKN_sz[0][0] * MKN[1] + my_cols[i] * MKN_sz[1][0]);
      types_recv[0][0] = ABC_dt[0][r*2+c];
      counts_recv[1][0] = 1; // B
      dis_recv[1][0] = sizeof(int) * (my_rows[i] * MKN_sz[1][0] * MKN[2] + my_cols[i] * MKN_sz[2][0]);
      types_recv[1][0] = ABC_dt[1][r*2+c];
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
          types_send[0][rank] = ABC_dt[0][m*2+n];
          types_send[1][rank] = ABC_dt[1][m*2+n];
          dis_send[0][rank] = sizeof(int) * (rr * MKN_sz[0][0] * MKN[1] + cc * MKN_sz[1][0]);
          dis_send[1][rank] = sizeof(int) * (rr * MKN_sz[1][0] * MKN[2] + cc * MKN_sz[2][0]);
        }
      }
    }

    MPI_Alltoallw(&A[0][0], counts_send[0], dis_send[0], types_send[0], &local_A[0][0], counts_recv[0], dis_recv[0], types_recv[0], MPI_COMM_WORLD);
    MPI_Alltoallw(&B[0][0], counts_send[1], dis_send[1], types_send[1], &local_B[0][0], counts_recv[1], dis_recv[1], types_recv[1], MPI_COMM_WORLD);
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
    for (j = 0; j < comm_sz; j++){
      types_recv[j] = types_send[j] = MPI_INT;
      counts_recv[j] = counts_send[j] = 0;
      dis_recv[j] = dis_send[j] = 0;
    }
    if (flags[i]){
      r = bz_idx(my_rows[i], grid_sz);
      c = bz_idx(my_cols[i], grid_sz);
      counts_send[0] = 1; 
      dis_send[0] = sizeof(int) * (my_rows[i] * MKN_sz[0][0] * MKN[2] + my_cols[i] * MKN_sz[2][0]);
      types_send[0] = C_dt[r*2+c];
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
          types_recv[rank] = C_dt[m*2+n];
          dis_recv[rank] = sizeof(int) * (rr * MKN_sz[0][0] * MKN[2] + cc * MKN_sz[2][0]);
        }
      }
    }
    MPI_Alltoallw(&local_C[0][0], counts_send, dis_send, types_send, &C[0][0], counts_recv, dis_recv, types_recv, MPI_COMM_WORLD);
  }
}