#pragma once

#include "Matrix.hpp"

class ActFunc{
public:
  static void tanh(MatD& x);
  static void tanh(VecD& x);
  static MatD tanhPrime(const MatD& x);
  static VecD tanhPrime(const VecD& x);

  static Real logistic(const Real x);
  static void logistic(MatD& x);
  static void logistic(VecD& x);
  static MatD logisticPrime(const MatD& x);
  static VecD logisticPrime(const VecD& x);

  static void relu(VecD& x);
  static VecD reluPrime(const VecD& x);
};

//f(x) = tanh(x)
#ifdef USE_EIGEN_TANH
inline void ActFunc::tanh(MatD& x){
  x = x.array().tanh();
}
inline void ActFunc::tanh(VecD& x){
  x = x.array().tanh();
}
#else
inline void ActFunc::tanh(MatD& x){
  x = x.unaryExpr(std::ptr_fun(::tanh));
}
inline void ActFunc::tanh(VecD& x){
  x = x.unaryExpr(std::ptr_fun(::tanh));
}
#endif

//f'(x) = 1-(f(x))^2
inline MatD ActFunc::tanhPrime(const MatD& x){
  return 1.0-x.array().square();
}
inline VecD ActFunc::tanhPrime(const VecD& x){
  return 1.0-x.array().square();
}

//f(x) = sigm(x)
inline Real ActFunc::logistic(const Real x){
  return 1.0/(1.0+::exp(-x));
}
inline void ActFunc::logistic(MatD& x){
  x = x.unaryExpr(std::ptr_fun((Real (*)(const Real))ActFunc::logistic));
}
inline void ActFunc::logistic(VecD& x){
  x = x.unaryExpr(std::ptr_fun((Real (*)(const Real))ActFunc::logistic));
}

//f'(x) = f(x)(1-f(x))
inline MatD ActFunc::logisticPrime(const MatD& x){
  return x.array()*(1.0-x.array());
}
inline VecD ActFunc::logisticPrime(const VecD& x){
  return x.array()*(1.0-x.array());
}

//ReLu
inline void ActFunc::relu(VecD& x){
  for (unsigned int i = 0; i < x.rows(); ++i){
    if (x.coeff(i, 0) <= 0.0){
      x.coeffRef(i, 0) = 0.0;
    }
  }
}

inline VecD ActFunc::reluPrime(const VecD& x){
  VecD res = x;
  for (unsigned int i = 0; i < res.rows(); ++i){
    if (res.coeff(i, 0) <= 0.0){
      res.coeffRef(i, 0) = 0.0;
    }
    else {
      res.coeffRef(i, 0) = 1.0;
    }
  }
  return res;
}
