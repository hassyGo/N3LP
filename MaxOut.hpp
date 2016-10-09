#pragma once

#include "Rand.hpp"

class MaxOut{
public:
  class Grad;
  class State;

  std::vector<MatD> weight;
  std::vector<VecD> bias;

  MaxOut(){}
  MaxOut(const unsigned int inputDim, const unsigned int hiddenDim, const unsigned int slice){
    for (unsigned int i = 0; i < slice; ++i){
      this->weight.push_back(MatD(hiddenDim, inputDim));
      this->bias.push_back(VecD(hiddenDim));
    }
  }

  void forward(const VecD& input, VecD& output);
  void forward(const VecD& input, VecD& output, MaxOut::State& state);
  void backward(const VecD& input, const VecD& output, const VecD& deltaOutput, VecD& deltaInput, MaxOut::State& state, MaxOut::Grad& grad);
  void init(Rand& rnd, const Real scale);
  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);

  void operator += (const MaxOut& maxout);
  void operator /= (const Real val);
};

class MaxOut::Grad{
public:
  Grad() {}
  Grad(const MaxOut& mx)
  {
    for (unsigned int i = 0; i < mx.weight.size(); ++i){
      this->weightGrad.push_back(MatD::Zero(mx.weight[i].rows(), mx.weight[i].cols()));
      this->biasGrad.push_back(VecD::Zero(mx.bias[i].rows()));
    }
  }

  std::vector<MatD> weightGrad;
  std::vector<VecD> biasGrad;

  void init();
  Real norm();
  void l2reg(const Real lambda, const MaxOut& mx);
  void sgd(const Real learningRate, MaxOut& af);
  void operator += (const MaxOut::Grad& grad);
  void operator /= (const Real val);
};

class MaxOut::State{
public:
  State() {}
  State(const MaxOut& mx){
    this->maxIndex = VecI(mx.bias[0].rows());
    this->preOutput = MatD(mx.bias[0].rows(), mx.bias.size());
  }

  VecI maxIndex;
  MatD preOutput;
};
