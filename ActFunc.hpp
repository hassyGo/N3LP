#pragma once

#include "Matrix.hpp"

class ActFunc{
public:
  static void tanh(MatD& x);
  static MatD tanhPrime(const MatD& x);

  static double logistic(const double x);
  static void logistic(MatD& x);
  static MatD logisticPrime(const MatD& x);
};

//f(x) = tanh(x)
inline void ActFunc::tanh(MatD& x){
  x = x.unaryExpr(std::ptr_fun(::tanh));
}

//f'(x) = 1-(f(x))^2
inline MatD ActFunc::tanhPrime(const MatD& x){
  return 1.0-x.array().square();
}

//f(x) = sigm(x)
inline double ActFunc::logistic(const double x){
  return 1.0/(1.0+::exp(-x));
}
inline void ActFunc::logistic(MatD& x){
  x = x.unaryExpr(std::ptr_fun((double (*)(const double))ActFunc::logistic));
}

//f'(x) = f(x)(1-f(x))
inline MatD ActFunc::logisticPrime(const MatD& x){
  return x.array()*(1.0-x.array());
}
