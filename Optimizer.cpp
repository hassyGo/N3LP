#include "Optimizer.hpp"

void Optimizer::sgd(const MatD& grad, const Real learningRate, MatD& param){
  param -= learningRate*grad;
}

void Optimizer::sgd(const VecD& grad, const Real learningRate, VecD& param){
  param -= learningRate*grad;
}

void Optimizer::adagrad(MatD& grad, const Real learningRate, MatD& gradHist, MatD& param){
  gradHist.array() += grad.array().square();
  grad.array() /= gradHist.array().sqrt();
  Optimizer::sgd(grad, learningRate, param);
}

void Optimizer::adagrad(VecD& grad, const Real learningRate, VecD& gradHist, VecD& param){
  gradHist.array() += grad.array().square();
  grad.array() /= gradHist.array().sqrt();
  Optimizer::sgd(grad, learningRate, param);
}

void Optimizer::momentum(MatD& grad, const Real learningRate, const Real m, MatD& gradHist, MatD& param){
  gradHist.array() *= m;
  Optimizer::sgd(grad, -learningRate, gradHist);
  Optimizer::sgd(gradHist, 1.0, param);
}

void Optimizer::momentum(VecD& grad, const Real learningRate, const Real m, VecD& gradHist, VecD& param){
  gradHist.array() *= m;
  Optimizer::sgd(grad, -learningRate, gradHist);
  Optimizer::sgd(gradHist, 1.0, param);
}
