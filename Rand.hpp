#pragma once

#include "Matrix.hpp"
#include <vector>

class Rand{
public:
  static Rand r_;

  Rand(unsigned long w_ = 88675123):
    x(123456789), y(362436069), z(521288629), w(w_) {};

  unsigned long next();
  double zero2one();
  void uniform(MatD& mat, const double scale = 1.0);
  double gauss(double sigma, double mu = 0.0);
  void gauss(MatD& mat, double sigma, double mu = 0.0);
  template <typename T> void shuffle(std::vector<T>& data);

private:
  unsigned long x;
  unsigned long y;
  unsigned long z;
  unsigned long w;
  unsigned long t; //tmp
};

inline unsigned long Rand::next(){
  this->t=(this->x^(this->x<<11));
  this->x=this->y;
  this->y=this->z;
  this->z=this->w;
  return (this->w=(this->w^(this->w>>19))^(this->t^(this->t>>8)));
}

inline double Rand::zero2one(){
  return ((this->next()&0xFFFF)+1)/65536.0;
}

inline void Rand::uniform(MatD& mat, const double scale){
  for (int i = 0; i < mat.rows(); ++i){
    for (int j = 0; j < mat.cols(); ++j){
      mat.coeffRef(i, j) = 2.0*this->zero2one()-1.0;
    }
  }

  mat *= scale;
}

inline double Rand::gauss(double sigma, double mu){
  return
    mu+
    sigma*
    sqrt(-2.0*log(this->zero2one()))*
    sin(2.0*M_PI*this->zero2one());
}

inline void Rand::gauss(MatD& mat, double sigma, double mu){
  for (int i = 0; i < mat.rows(); ++i){
    for (int j = 0; j < mat.cols(); ++j){
      mat.coeffRef(i, j) = this->gauss(sigma, mu);
    }
  }
}

template <typename T> inline void Rand::shuffle(std::vector<T>& data){
  T tmp;

  for (int i = data.size(), a, b; i > 1; --i){
    a = i-1;
    b = this->next()%i;
    tmp = data[a];
    data[a] = data[b];
    data[b] = tmp;
  }
}
