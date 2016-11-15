#pragma once

#include "Matrix.hpp"
#include "Optimizer.hpp"

class SoftMax{
public:
  SoftMax(){};
  SoftMax(const int inputDim, const int classNum):
    weight(MatD::Zero(inputDim, classNum)), bias(VecD::Zero(classNum))
  {}
  SoftMax(const int inputDim, const int classNum, const int exception_, const Real gamma_, const Real mPlus_, const Real mMinus_):
    weight(MatD::Zero(inputDim, classNum)), bias(VecD::Zero(classNum)), exception(exception_), gamma(gamma_), mPlus(mPlus_), mMinus(mMinus_)
  {}

  class Grad;

  MatD weight; VecD bias;

  void calcDist(const VecD& input, VecD& output);
  Real calcLoss(const VecD& output, const int label);
  Real calcLoss(const VecD& output, const VecD& goldOutput);
  void backward(const VecD& input, const VecD& output, const int label, VecD& deltaFeature, SoftMax::Grad& grad);
  void backward(const VecD& input, const VecD& output, const VecD& goldOutput, VecD& deltaFeature, SoftMax::Grad& grad);
  void backwardAttention(const VecD& input, const VecD& output, const VecD& deltaOut, VecD& deltaFeature, SoftMax::Grad& grad);
  void sgd(const SoftMax::Grad& grad, const Real learningRate);
  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);

  void operator += (const SoftMax& softmax);
  void operator /= (const Real val);

  //for ranking loss
  int exception;
  Real gamma, mPlus, mMinus;
  void calcScore(const VecD& input, VecD& output);
  Real calcRankingLoss(const VecD& output, const int label);
  void backwardRankingLoss(const VecD& input, const VecD output, const int label, VecD& deltaFeature, SoftMax::Grad& grad);
};

class SoftMax::Grad{
public:
  Grad(): gradHist(0){}
  Grad(const SoftMax& softmax):
    gradHist(0)
  {
    this->weight = MatD::Zero(softmax.weight.rows(), softmax.weight.cols());
    this->bias = VecD::Zero(softmax.bias.rows());
  }

  SoftMax::Grad* gradHist;
  
  MatD weight; VecD bias;

  void init(){
    this->weight.setZero();
    this->bias.setZero();
  }

  Real norm(){
    return this->weight.squaredNorm()+this->bias.squaredNorm();
  }

  void l2reg(const Real lambda, const SoftMax& s){
    this->weight += lambda*s.weight;
  }

  void l2reg(const Real lambda, const SoftMax& s, const SoftMax& target){
    this->weight += lambda*(s.weight-target.weight);
    this->bias += lambda*(s.bias-target.bias);
  }

  void sgd(const Real learningRate, SoftMax& softmax){
    Optimizer::sgd(this->weight, learningRate, softmax.weight);
    Optimizer::sgd(this->bias, learningRate, softmax.bias);
  }

  void adagrad(const Real learningRate, SoftMax& softmax, const Real initVal = 1.0){
    if (this->gradHist == 0){
      this->gradHist = new SoftMax::Grad(softmax);
      this->gradHist->weight.fill(initVal);
      this->gradHist->bias.fill(initVal);
    }

    Optimizer::adagrad(this->weight, learningRate, this->gradHist->weight, softmax.weight);
    Optimizer::adagrad(this->bias, learningRate, this->gradHist->bias, softmax.bias);
  }

  void momentum(const Real learningRate, const Real m, SoftMax& softmax){
    if (this->gradHist == 0){
      this->gradHist = new SoftMax::Grad(softmax);
      this->gradHist->weight.fill(0.0);
      this->gradHist->bias.fill(0.0);
    }

    Optimizer::momentum(this->weight, learningRate, m, this->gradHist->weight, softmax.weight);
    Optimizer::momentum(this->bias, learningRate, m, this->gradHist->bias, softmax.bias);
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
