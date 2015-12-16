#pragma once

#include "Matrix.hpp"

class SoftMax{
public:
  SoftMax(){};
  SoftMax(const int inputDim, const int classNum):
    weight(MatD::Zero(classNum, inputDim)), bias(MatD::Zero(classNum, 1))
  {};

  class Grad;

  MatD weight, bias;

  void calcDist(const MatD& input, MatD& output);
  double calcLoss(const MatD& output, const int label);
  void backward(const MatD& input, const MatD& output, const int label, MatD& deltaFeature, SoftMax::Grad& grad);
  void sgd(const SoftMax::Grad& grad, const double learningRate);
};

class SoftMax::Grad{
public:
  Grad(const SoftMax& softmax){
    this->weight = MatD::Zero(softmax.weight.rows(), softmax.weight.cols());
    this->bias = MatD::Zero(softmax.bias.rows(), softmax.bias.cols());
  }

  MatD weight, bias;

  void init(){
    this->weight.setZero();
    this->bias.setZero();
  }

  double norm(){
    return this->weight.squaredNorm()+this->bias.squaredNorm();
  }
};
