#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_blas.h>
#include <sys/time.h>

#define GET_TIME(now){ \
  struct timeval t; \
  gettimeofday(&t, NULL); \
  now = t.tv_sec + t.tv_usec/10000000.0; \
}

const int RMAX = 1000000;
int m, n, k, sol;
gsl_matrix *A, *B, *C;

void Generate_matrix(gsl_matrix *mat, int m, int n);
void Get_args(int argc, char* argv[]);
void Usage(char* prog_name);

int main (int argc, char* argv[])
{
  printf("starttttt");
  Get_args(argc, argv);
  printf("log\n");

  A = gsl_matrix_alloc (m, n);
  B = gsl_matrix_alloc (n, k);
  C = gsl_matrix_alloc (m, k);
  printf("log\n");
  Generate_matrix(A, m, n);
  printf("log\n");

  Generate_matrix(B, n, k);
  printf("log\n");

  double start, finish;
  GET_TIME(start);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, B, 0.0, C);
  GET_TIME(finish);
  printf("Elapsed time = %.6f seconds\n", finish-start);

 
  gsl_matrix_free(A);
  gsl_matrix_free(B);
  gsl_matrix_free(C);

  return 0;
}

void Get_args(int argc, char* argv[]){
  if(argc != 4)
    Usage(argv[0]);
  m = strtol(argv[1], NULL, 10);
  n = strtol(argv[2], NULL, 10);
  k = strtol(argv[3], NULL, 10);
  if ( m <= 0 || n <= 0 || k <= 0)
    Usage(argv[0]);
}
void Usage(char* prog_name){
    fprintf(stderr, "Usage: %s <m> <n> <k> <sol>\n", prog_name);
    exit(0);
}



void Generate_matrix(gsl_matrix *mat, int m, int n)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
          gsl_matrix_set(mat, i, j, (rand() % RMAX) / (RMAX / 10.0));
}


