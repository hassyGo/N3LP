#pragma once

#include "Rand.hpp"

class Affine{
public:
  class Grad;

  enum ACT{
    TANH,
    RELU,
  };

  Affine::ACT act;
  MatD weight;
  VecD bias;

  Affine(){}
  Affine(const unsigned int inputDim, const unsigned int hiddenDim):
    act(Affine::RELU)
  {
    this->weight = MatD(hiddenDim, inputDim);
    this->bias = VecD(hiddenDim);
  }

  void forward(const VecD& input, VecD& output);
  void backward(const VecD& input, const VecD& output, const VecD& deltaOutput, VecD& deltaInput, Affine::Grad& grad);
  void init(Rand& rnd, const Real scale);
  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);

  void operator += (const Affine& affine);
  void operator /= (const Real val);
};

class Affine::Grad{
public:
  Grad(): gradHist(0){}
  Grad(const Affine& af):
  gradHist(0)
  {
    this->weightGrad = MatD::Zero(af.weight.rows(), af.weight.cols());
    this->biasGrad = VecD::Zero(af.bias.rows());
  }

  Affine::Grad* gradHist;

  MatD weightGrad;
  VecD biasGrad;

  void init();
  Real norm();
  void l2reg(const Real lambda, const Affine& af);
  void l2reg(const Real lambda, const Affine& af, const Affine& target);
  void sgd(const Real learningRate, Affine& af);
  void adagrad(const Real learningRate, Affine& affine, const Real initVal = 1.0);
  void momentum(const Real learningRate, const Real m, Affine& affine);
  void operator += (const Affine::Grad& grad);
  void operator /= (const Real val);
};
