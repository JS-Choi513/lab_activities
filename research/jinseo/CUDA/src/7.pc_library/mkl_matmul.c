#include <stdio.h>
#include <stdlib.h>
#include <mkl/mkl.h>
#include <sys/time.h>



#define GET_TIME(now) { \
    struct timeval t; \
    gettimeofday(&t, NULL); \
    now = t.tv_sec + t.tv_usec/1000000.0; \
}

const int RMAX = 1000000;
#ifdef DEBUG
const int NCOUNT = 1; // number of multiplication
#else
const int NCOUNT = 1; // number of multiplication
#endif

void Get_args(int argc, char* argv[], int* m, int* n, int* k);
void Usage(char* prog_name);
void Generate_matrix(double mat[], int m, int n);
void Print_matrix(double mat[], int m, int n, char* title);


int main(int argc, char* argv[])
{
    int m, n, k;
    Get_args(argc, argv, &m, &n, &k);
 
    double *A, *B, *C;
    A = (double*)mkl_malloc(m * n * sizeof(double), 64);
    B = (double*)mkl_malloc(n * k * sizeof(double), 64);
    C = (double*)mkl_malloc(m * k * sizeof(double), 64);
 
    Generate_matrix(A, m, n);
    Generate_matrix(B, n, k);
 
    double start, finish, avg_elapsed = 0.0;
    GET_TIME(start);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, k, n, 1.0, A, n, B, k, 0, C, k);
    GET_TIME(finish);
    printf("[%3d] Elapsed time = %.6f seconds\n", count+1, finish-start);
    
#ifdef DEBUG
    Print_matrix(C, m, k, "The product is");
#endif
 
    printf("Average elapsed time = %.6f seconds\n", avg_elapsed);
 
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
 
    return 0;
}

void Get_args(int argc, char* argv[], int* m, int* n, int* k)
{
    if (argc != 4)
        Usage(argv[0]);
    
    *m = strtol(argv[1], NULL, 10);
    *n = strtol(argv[2], NULL, 10);
    *k = strtol(argv[3], NULL, 10);
    if (*m <= 0 || *n <= 0 || *k <= 0)
        Usage(argv[0]);
}




void Usage(char* prog_name)
{
    fprintf(stderr, "Usage: %s <m> <n> <k>\n", prog_name);
    exit(0);
}

void Generate_matrix(double mat[], int m, int n)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            mat[i*n + j] = (rand() % RMAX) / (RMAX / 10.0);
}


void Print_matrix(double mat[], int m, int n, char* title)
{
    printf("%s\n", title);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            printf("%f ", mat[i*n + j]);
        printf("\n");
    }
}

