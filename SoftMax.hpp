#pragma once

#include "Matrix.hpp"

class SoftMax{
public:
  SoftMax(){};
  SoftMax(const int inputDim, const int classNum):
    weight(MatD::Zero(classNum, inputDim)), bias(VecD::Zero(classNum))
  {};

  class Grad;

  MatD weight; VecD bias;

  void calcDist(const VecD& input, VecD& output);
  Real calcLoss(const VecD& output, const int label);
  void backward(const VecD& input, const VecD& output, const int label, VecD& deltaFeature, SoftMax::Grad& grad);
  void sgd(const SoftMax::Grad& grad, const Real learningRate);
  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);

};

class SoftMax::Grad{
public:
  Grad(){}
  Grad(const SoftMax& softmax){
    this->weight = MatD::Zero(softmax.weight.rows(), softmax.weight.cols());
    this->bias = VecD::Zero(softmax.bias.rows());
  }

  MatD weight; VecD bias;

  void init(){
    this->weight.setZero();
    this->bias.setZero();
  }

  Real norm(){
    return this->weight.squaredNorm()+this->bias.squaredNorm();
  }

  void operator += (const SoftMax::Grad& grad){
    this->weight += grad.weight;
    this->bias += grad.bias;
  }

  void operator /= (const Real val){
    this->weight /= val;
    this->bias /= val;
  }
};
