/**
 * @file mpi_cannon.c
 * @brief 
 * Square processor layout -- "round-down" approach
 * (comm_sz - grid_sz * grid_sz) number of processes is not utilized. (idle)
 * 
 * MPI_Sendrecv_replace for Cannon data "shift"s in rows and columns
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <string.h>
#include "../util/utils.h"

#define bz_idx(i, grid_sz) (i == grid_sz - 1 ? 1 : 0)
#define max(a, b) (a > b? a : b)
void Build_Vector_Types(int my_rank, int* MKN, int grid_sz, int* my_coor, int** MKN_sz, MPI_Datatype** ABC_vec);

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
      printf("./mpi_cannon <input_filename>\n");
      exit(-1);
  }
  /* round-down process grid */
  grid_sz = (int) floor(sqrt((double) comm_sz));
  logical_p = grid_sz * grid_sz;

  /* Read inputs from file */
  if (my_rank == 0){
    mpi_readInputC(argv[1], MKN, &A, &B, &C, &oracle);
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
  MPI_Bcast(MKN, 3, MPI_INT, 0, MPI_COMM_WORLD);

  /* establish a new communicator */
  int periods[2] = {1, 1}, dims[2] = {grid_sz, grid_sz};
  MPI_Comm new_world_comm; 
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &new_world_comm);

  /* performance measurement */
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();

  if (my_rank < logical_p){
    /* set up communicators and identify neighbours */
    int my_coor[2] = {my_rank / grid_sz, my_rank % grid_sz};
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(new_world_comm, my_coor[0], 0, &row_comm);
    MPI_Comm_split(new_world_comm, my_coor[1], 0, &col_comm);
    int left, right, up, down;
    MPI_Cart_shift(new_world_comm, 0, 1, &up, &down);
    MPI_Cart_shift(new_world_comm, 1, 1, &left, &right);
    int col_tag = 1, row_tag = 0;

    /* determine block sizes and construct vector types */
    int M_sz[2], K_sz[2], N_sz[2];
    int* MKN_sz[3] = {M_sz, K_sz, N_sz};
    MPI_Datatype A_vec[2], B_vec[2], C_vec[2]; // 0: full row vec, 1: block vec
    MPI_Datatype* ABC_vec[3] = {A_vec, B_vec, C_vec};
    Build_Vector_Types(my_rank, MKN, grid_sz, my_coor, MKN_sz, ABC_vec);
    int* local_A, *local_B, *local_C;

    /* ----------------------------------- */
    /* distribute A & B with 2-phase scatter */
    int* full_rows[2] = {NULL, NULL};// 0: A; 1: B
    int counts_send[2][grid_sz], disp[2][grid_sz];
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
      MPI_Scatterv(&A[0][0], counts_send[0], disp[0], MPI_INT, full_rows[0], M_sz[r]* MKN[1], MPI_INT, 0, col_comm);
      MPI_Scatterv(&B[0][0], counts_send[1], disp[1], MPI_INT, full_rows[1], K_sz[r]* MKN[2], MPI_INT, 0, col_comm);
    } 
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
    MPI_Scatterv(full_rows[0], counts_send[0], disp[0], A_vec[0], local_A, K_sz[shifted_k], A_vec[1], 0, row_comm);
    MPI_Scatterv(full_rows[1], counts_send[1], disp[1], B_vec[0], local_B, N_sz[c], B_vec[1], 0, row_comm);
    if (my_coor[1] == 0){
      free(full_rows[0]);
      free(full_rows[1]);
    }
    // separate initial shift of B 
    int initshift_up, initshift_down;
    MPI_Cart_shift(new_world_comm, 0, my_coor[1], &initshift_up, &initshift_down);
    MPI_Sendrecv_replace(local_B, max(K_sz[0], K_sz[1]) * N_sz[c], MPI_INT, initshift_up, col_tag, initshift_down, col_tag, new_world_comm, MPI_STATUS_IGNORE);

    /* ----------------------------------- */
    /* Cannon algorithm */
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
      /* shift A */
      MPI_Sendrecv_replace(local_A, M_sz[r] * max(K_sz[0], K_sz[1]), MPI_INT, left, row_tag, right, row_tag, new_world_comm, MPI_STATUS_IGNORE);
      /* shift B */
      MPI_Sendrecv_replace(local_B, max(K_sz[0], K_sz[1]) * N_sz[c], MPI_INT, up, col_tag, down, col_tag, new_world_comm, MPI_STATUS_IGNORE);
      shifted_coor = (shifted_coor+1) % grid_sz;
      shifted_k = bz_idx(shifted_coor, grid_sz);
    }

    /* ----------------------------------- */
    /* collect results of C with 2-phase gather */
    int* full_rows_C = NULL;
    int counts_recv[grid_sz], disp_recv[grid_sz];
    // phase 1 : collect submatrices into full rows in row communicators
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

    MPI_Gatherv(local_C, N_sz[c], C_vec[1], full_rows_C, counts_recv, disp_recv, C_vec[0], 0, row_comm);

    // phase 2: collect full rows into C in P0 column communicator
    if (my_coor[1] == 0){
      for (j = 0 ; j < grid_sz; j++){
        if (j != grid_sz - 1){
          counts_recv[j] = M_sz[0] * MKN[2];
        }else {
          counts_recv[j] = M_sz[1] * MKN[2];
        }
        disp_recv[j] = j * M_sz[0] * MKN[2];
      }

      MPI_Gatherv(full_rows_C, M_sz[r] * MKN[2], MPI_INT, &C[0][0], counts_recv, disp_recv, MPI_INT, 0, col_comm);
    }

    free(local_A); free(local_B); free(local_C);
    for (i = 0; i < 2; i++){
      MPI_Type_free(&A_vec[i]);
      MPI_Type_free(&B_vec[i]);
      MPI_Type_free(&C_vec[i]);
    }
    MPI_Comm_free(&new_world_comm);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
  }

  /* correctness checking and timing output */
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
 * build MPI vector types for 2-phase scatter and gather
 */
void Build_Vector_Types(int my_rank, int* MKN, int grid_sz, int* my_coor, int** MKN_sz, MPI_Datatype** ABC_vec){
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
    MPI_Type_vector(MKN_sz[ABC_dim[k].r][i], 1, MKN[ABC_dim[k].c], MPI_INT, &ABC_vec[k][0]);
    MPI_Type_create_resized(ABC_vec[k][0], 0, sizeof(int), &ABC_vec[k][0]);
    MPI_Type_commit(&ABC_vec[k][0]);
    // block vec type
    MPI_Type_vector(MKN_sz[ABC_dim[k].r][i], 1, MKN_sz[ABC_dim[k].c][indices[k]], MPI_INT, &ABC_vec[k][1]);
    MPI_Type_create_resized(ABC_vec[k][1], 0, sizeof(int), &ABC_vec[k][1]);
    MPI_Type_commit(&ABC_vec[k][1]);
  }
}