#include <iostream>
#include <eigen3/Eigen/Dense>
#include <sys/time.h>
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE
using namespace Eigen;
using namespace std;

#define GET_TIME(now){\
  struct timeval t;\
  gettimeofday(&t, NULL); \
  now = t.tv_sec + t.tv_usec/10000000.0; \
}

const int RMAX = 1000000;
int m, n, k;

void Get_args(int argc, const char **argv);
void Generate_matrix(MatrixXd mat, int m, int n);
int main(int argc, char **argv)
{ 
  Get_args(argc, const_cast<const char**>(argv));

  MatrixXd A(m, n);
  MatrixXd B(n, k);
  MatrixXd C(n, k);
//  Matrix *B = new Matrix<double, n, k>;
//  Matrix *C = new Matrix<double, m, k>;
  Generate_matrix(A, m, n);
  Generate_matrix(B, n, k);
  
  double start, finish;
  GET_TIME(start);
  C = A*B.transpose();
  GET_TIME(finish);
  printf("Elapsed time = %.6f seconds\n", finish-start);

 
 
  
   
  return 0;
}

void Get_args(int argc, const char** argv){
  m = strtol(argv[1], NULL, 10);
  n = strtol(argv[2], NULL, 10);
  k = strtol(argv[3], NULL, 10);
}
void Generate_matrix(MatrixXd mat, int m, int n)
{ 
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
          mat(i,j) = (rand() % RMAX) / (RMAX / 10.0);

}









