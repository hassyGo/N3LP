#pragma once

#include "LSTM.hpp"

class DeepLSTM{
public:
  DeepLSTM(){};
  DeepLSTM(const int inputDim, const int hiddenDim, const int depth);
  DeepLSTM(const int inputDim, const int additionalInputDim, const int hiddenDim, const int depth);

  class State;
  class Grad;

  std::vector<LSTM> lstms;

  void init(Rand& rnd, const Real scale = 1.0);
  void forward(const VecD& xt, const DeepLSTM::State* prev, DeepLSTM::State* cur, int startDepth  = -1, int endDepth = -1);
  void forward(const VecD& xt, DeepLSTM::State* cur, int startDepth  = -1, int endDepth = -1);
  void backward(DeepLSTM::State* prev, DeepLSTM::State* cur, DeepLSTM::Grad& grad, const VecD& xt, int startDepth  = -1, int endDepth = -1);
  void backward(DeepLSTM::State* cur, DeepLSTM::Grad& grad, const VecD& xt, int startDepth  = -1, int endDepth = -1);
  void sgd(const DeepLSTM::Grad& grad, const Real learningRate);
  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);

  void forward(const VecD& xt, const VecD& at, const DeepLSTM::State* prev, DeepLSTM::State* cur, int startDepth  = -1, int endDepth = -1);
  void forward(const VecD& xt, const VecD& at, DeepLSTM::State* cur, int startDepth  = -1, int endDepth = -1);
  void backward(DeepLSTM::State* prev, DeepLSTM::State* cur, DeepLSTM::Grad& grad, const VecD& xt, const VecD& at, int startDepth  = -1, int endDepth = -1);
  void backward(DeepLSTM::State* cur, DeepLSTM::Grad& grad, const VecD& xt, const VecD& at, int startDepth  = -1, int endDepth = -1);

  void operator += (const DeepLSTM& lstm);
  void operator /= (const Real val);
};

class DeepLSTM::State{
public:
  ~State() {this->clear();}
  State(const DeepLSTM& dlstm);

  std::vector<LSTM::State*> lstm;

  void clear();
};

class DeepLSTM::Grad{
public:
  Grad(){}
  Grad(const DeepLSTM& dlstm);

  std::vector<LSTM::Grad> lstm;

  void init(int depth = -1);
  Real norm(int depth = -1);
  void sgd(const Real learningRate, const unsigned int depth, DeepLSTM& lstm);
  void adagrad(const Real learningRate, DeepLSTM& lstm, const Real initVal = 1.0);
  void momentum(const Real learningRate, const Real m, DeepLSTM& lstm);

  void operator += (const DeepLSTM::Grad& grad);
  void operator /= (const Real val);
};
