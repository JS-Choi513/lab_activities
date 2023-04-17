#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#define N 4096
const int RMAX = 1000000;


#define GET_TIME(now){ \
  struct timeval t; \
  gettimeofday(&t, NULL); \
  now = t.tv_sec + t.tv_usec/10000000.0; \
}

void fill_matrices(double first[][N], double second[][N], double result[][N])
{
    srand(time(NULL)); // randomize seed
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            first[i][j] = rand() % 10;
            second[i][j] = rand() % 10;
            result[i][j] = 0;
        }
    }
}

int main()
{
    static double first[N][N], second[N][N], result[N][N];
    fill_matrices(first, second, result);
    double start, finish;
    GET_TIME(start);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N,N,N, 1.0, first, N, second, N, 0.0, result, N);
    GET_TIME(finish);
    printf("Elapsed time = %.6f seconds\n", finish-start);






    return 0;
}
