#include <iostream>
#include <Eigen/Dense>
#include <sys/time.h>

#define GET_TIME(now){\
  struct timeval t;
  gettimeofday(&t, NULL); \
  now = t.tv_sec + t.tv_usec/10000000.0; \
}

const int RMAX = 1000000;
int m, n, k;

void Get_args(int argc, char **argv[]);

int main(int argc, char **argv)
{ 
  Get_args(argv, argv);

  Eigen::Matrix2d mat;
  mat << 1, 2,
         3, 4;
  Eigen::Vector2d u(-1,1), v(2,0);
  std::cout << "Here is mat*mat:\n" << mat*mat << std::endl;
  std::cout << "Here is mat*u:\n" << mat*u << std::endl;
  std::cout << "Here is u^T*mat:\n" << u.transpose()*mat << std::endl;
  std::cout << "Here is u^T*v:\n" << u.transpose()*v << std::endl;
  std::cout << "Here is u*v^T:\n" << u*v.transpose() << std::endl;
  std::cout << "Let's multiply mat by itself" << std::endl;
  mat = mat*mat;
  std::cout << "Now mat is mat:\n" << mat << std::endl;

  return 0;
}

void Get_args(int argc, char** argv[]){
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


