#pragma once
#include <Eigen/Dense>

template<class T, class S>
class spread {
public:
  spread(std::vector<T>& input) {
    dim = input[0].size();
    this->input = input;
    //cov_matrix.resize(dim, dim);
  }

  void doEvalEvec() {
    calcCovMatrix();
    getEigen();
  }

  S getCov() const {
    return covMatrix;
  }

  T getCenter() const {
    return center;
  }

  int getDim() const {
    return dim;
  }

  void printTest() {
    /*cov_matrix(0, 0) = 1.07;
    cov_matrix(0, 1) = 0.63;
    cov_matrix(1, 0) = 0.63;
    cov_matrix(1, 1) = 0.64;
    get_eigen();
    std::cout << "evals = " << eval << std::endl;
    std::cout << "evecs = " << evec << std::endl;*/
  }

  float getHighestEval() {
    float max = 0;
    for (int i = 0; i < dim; ++i)
      //max = std::max(max, eval(i));
      max += eval(i);

    return max;
  }

  T getEigenvec() {
    float min = 1000;
    T ret;
    for (int i = 0; i < dim; ++i)
      if (eval(i) < min)
      {
        min = eval(i);
        for (int j = 0; j < dim; ++j)
          ret(j) = evec(j, i);
      }
    return ret;
  }

private:
  void calcCovMatrix() {
    int size = input.size();
    calcCenter();
    covMatrix = S::Zero();

    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j)
        for (int k = 0; k < size; ++k)
          covMatrix(i, j) += (input[k](i) - center(i)) * (input[k](j) - center(j)) / size;
  }

  void calcCenter() {
    center = T::Zero();
    for (auto m : input)
    {
      center += m;
    }
    center /= input.size();
  }

  void getEigen() {
    //matrix is symmetric
    Eigen::SelfAdjointEigenSolver <S> solver(covMatrix);
    eval = solver.eigenvalues();
    evec = solver.eigenvectors();
  }

  std::vector<T> input;
  T center;
  T eval;
  S evec;
  S covMatrix;
  int dim = 0;

};