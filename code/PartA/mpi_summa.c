/**
 * @file mpi_summa.c
 * @brief 
 * Square processor layout -- "round-down" approach
 * (comm_sz - grid_sz * grid_sz) number of processes is not utilized. (idle)
 * 
 * MPI_Bcast for SUMMA broadcasts
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <string.h>
#include "../util/utils.h"


#define bz_idx(i, grid_sz) (i == grid_sz - 1 ? 1 : 0)
#define max(a, b) (a > b? a : b)

void buildSubarrayTypes(int my_rank, int* MKN, int grid_sz, int** MKN_sz, MPI_Datatype** ABC_dt);
void distributeBlocksP0(int my_rank, int grid_sz, int* my_coor, int* MKN, int** MKN_sz, int*** local_A, int*** local_B, int*** local_C, MPI_Datatype** ABC_dt, int** A, int** B, MPI_Comm new_world_comm);
void gatherResultsP0(int my_rank, int grid_sz, int* my_coor, int* MKN, int** MKN_sz, int** C, MPI_Datatype* C_dt, int** local_C, MPI_Comm new_world_comm);

int main(int argc, char *argv[]){
  int** A, **B, **C, *oracle;
  int MKN[3];
  int i, j, k;
  int my_rank, comm_sz, grid_sz, logical_p;
  int r, c, mm, kk, nn;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  if (argc != 2){
    printf("./mpi_summa <input_filename>\n");
    exit(-1);
  }
  /* round-down process grid */
  grid_sz = (int) floor(sqrt((double) comm_sz));
  logical_p = grid_sz* grid_sz;

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

  /* establish a new communicator */
  int periods[2] = {1, 1}, dims[2] = {grid_sz, grid_sz};
  MPI_Comm new_world_comm; 
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &new_world_comm);

  /* performance measurement */
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();

  if (my_rank < logical_p){
     /* determine block sizes and construct subarray types */
    int M_sz[2], K_sz[2], N_sz[2];
    int* MKN_sz[3] = {M_sz, K_sz, N_sz};
    MPI_Datatype A_dt[4], B_dt[4], C_dt[4];
    MPI_Datatype* ABC_dt[3] = {A_dt, B_dt, C_dt}; 
    buildSubarrayTypes(my_rank, MKN, grid_sz, MKN_sz, ABC_dt);
    /* build row and col communicators */
    int my_coor[2] = {my_rank / grid_sz, my_rank % grid_sz};
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(new_world_comm, my_coor[0], 0, &row_comm);
    MPI_Comm_split(new_world_comm, my_coor[1], 0, &col_comm);
    /* distribute A and B */
    int **local_A, **local_B, **local_C;
    distributeBlocksP0(my_rank, grid_sz, my_coor, MKN, MKN_sz, &local_A, &local_B, &local_C, ABC_dt, A, B, new_world_comm);
    // needs to keep local_A and local_B
    int* temp_A, *temp_B;
    r = bz_idx(my_coor[0], grid_sz);
    c = bz_idx(my_coor[1], grid_sz);
    mm = M_sz[r]; nn = N_sz[c];
    memset(&local_C[0][0], 0, sizeof(int)* mm * nn);
    temp_A = malloc(sizeof(int) * mm * max(K_sz[0], K_sz[1]));
    temp_B = malloc(sizeof(int) * max(K_sz[0], K_sz[1]) * nn);

    /* SUMMA algorithm */
    for (k = 0; k < grid_sz; k++){
      c = bz_idx(k, grid_sz);
      kk = K_sz[c];
      if (my_coor[1] == k){ // sender of A
        for (i = 0; i < mm; i++){
          for (j = 0; j < kk; j++){
            temp_A[i*kk+j] = local_A[i][j];
          }
        }
      }
      MPI_Bcast(temp_A, mm * kk, MPI_INT, k, row_comm);
      if (my_coor[0] == k){ // sender of B
        for (i = 0; i < kk; i++){
          for (j = 0; j < nn; j++){
            temp_B[i*nn+j] = local_B[i][j];
          }
        }
      }
      MPI_Bcast(temp_B, kk * nn, MPI_INT, k, col_comm);
      
      for (int kkk = 0; kkk < kk; kkk++){
        for (i = 0; i < mm; i++){
          for (j = 0; j < nn; j++){
            local_C[i][j] += temp_A[i*kk + kkk] * temp_B[kkk*nn + j];
          }
        }
      }
    }
    gatherResultsP0(my_rank, grid_sz, my_coor, MKN, MKN_sz, C, C_dt, local_C, new_world_comm);

    free(temp_A); free(temp_B);
    free(local_A); free(local_B); free(local_C);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&new_world_comm);
    for (i = 0; i < 4; i++){
      MPI_Type_free(&A_dt[i]);
      MPI_Type_free(&B_dt[i]);
      MPI_Type_free(&C_dt[i]);
    }
  }
  
  /* correctness checking and timing output*/
  double end = MPI_Wtime();
  if (my_rank == 0){
    checkCorrectness(MKN[0], MKN[2], C, oracle);
    printf("Computed in %0.8fs\n", end-start);
  }

  free(A); free(B); free(C); free(oracle);
  MPI_Finalize();
  return 0;
}

/**
 * @brief 
 * build MPI "subarray" data types for each submatrix size
 */
void buildSubarrayTypes(int my_rank, int* MKN, int grid_sz, int** MKN_sz, MPI_Datatype** ABC_dt){
  int i, j, k;
  for (i = 0; i < 3; i++){
    MKN_sz[i][0] = round((double) MKN[i] / (double) grid_sz);
    MKN_sz[i][1] = MKN[i] - MKN_sz[i][0] * (grid_sz-1);
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
 * use MPI_Alltoallw to distribute submatrices of A and B
 */
void distributeBlocksP0(int my_rank, int grid_sz, int* my_coor, int* MKN, int** MKN_sz, int*** local_A, int*** local_B, int*** local_C, MPI_Datatype** ABC_dt, int** A, int** B, MPI_Comm new_world_comm){
  int j, k, rr, cc, n, m, r, c;
  int logical_p = grid_sz * grid_sz;
  int counts_send[2][logical_p], counts_recv[2][logical_p];
  int dis_send[2][logical_p], dis_recv[2][logical_p];
  MPI_Datatype types_send[2][logical_p], types_recv[2][logical_p];
  
  for (j = 0; j < logical_p; j++){
    for (k = 0; k < 2; k++){
      types_recv[k][j] = types_send[k][j] = MPI_INT;
      counts_recv[k][j] = counts_send[k][j] = 0;
      dis_recv[k][j] = dis_send[k][j] = 0;
    }
  }

  r = bz_idx(my_coor[0], grid_sz);
  c = bz_idx(my_coor[1], grid_sz);
  mallocCont(local_A, MKN_sz[0][r], MKN_sz[1][c]);
  mallocCont(local_B, MKN_sz[1][r], MKN_sz[2][c]);
  mallocCont(local_C, MKN_sz[0][r], MKN_sz[2][c]);
  counts_recv[0][0] = MKN_sz[0][r] *  MKN_sz[1][c]; // A
  counts_recv[1][0] = MKN_sz[1][r] *  MKN_sz[2][c]; // B

  if (my_rank == 0){
    for (int rank = 0; rank < logical_p; rank++){
      if (rank < logical_p){
        rr = rank / grid_sz;
        cc = rank % grid_sz;
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

  MPI_Alltoallw(&A[0][0], counts_send[0], dis_send[0], types_send[0], &(*local_A)[0][0], counts_recv[0], dis_recv[0], types_recv[0], new_world_comm);
  MPI_Alltoallw(&B[0][0], counts_send[1], dis_send[1], types_send[1], &(*local_B)[0][0], counts_recv[1], dis_recv[1], types_recv[1], new_world_comm);
}

/**
 * @brief 
 * use MPI_Alltoallw to collect submatrices of C
 */
void gatherResultsP0(int my_rank, int grid_sz, int* my_coor, int* MKN, int** MKN_sz, int** C, MPI_Datatype* C_dt, int** local_C, MPI_Comm new_world_comm){
  int i, r, c;
  int logical_p = grid_sz * grid_sz;
  int counts_send[logical_p], counts_recv[logical_p];
  int dis_send[logical_p], dis_recv[logical_p];
  MPI_Datatype types_send[logical_p], types_recv[logical_p];

  for (i = 0; i < logical_p; i++){
    types_recv[i] = types_send[i] = MPI_INT;
    counts_recv[i] = counts_send[i] = 0;
    dis_recv[i] = dis_send[i] = 0;
  }
  r = bz_idx(my_coor[0], grid_sz);
  c = bz_idx(my_coor[1], grid_sz);
  counts_send[0] = MKN_sz[0][r] * MKN_sz[2][c];
  int rr, cc, m, n;
  if (my_rank == 0){
    for (int rank = 0; rank < logical_p; rank++){
      if (rank < logical_p){
        rr = rank / grid_sz;
        cc = rank % grid_sz;
        m = bz_idx(rr, grid_sz);
        n = bz_idx(cc, grid_sz);
        counts_recv[rank] = 1;
        types_recv[rank] = C_dt[m*2+n];
        dis_recv[rank] = sizeof(int) * (rr * MKN_sz[0][0] * MKN[2] + cc * MKN_sz[2][0]);
      }
    }
  }
  MPI_Alltoallw(&local_C[0][0], counts_send, dis_send, types_send, &C[0][0], counts_recv, dis_recv, types_recv, new_world_comm);
}