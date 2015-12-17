#pragma once

#include "LSTM.hpp"

class DeepLSTM{
public:
  DeepLSTM(){};
  DeepLSTM(const int inputDim, const int hiddenDim, const int depth);

  class State;
  class Grad;

  std::vector<LSTM> lstms;

  void init(Rand& rnd, const double scale = 1.0);
  void forward(const MatD& xt, const DeepLSTM::State* prev, DeepLSTM::State* cur);
  void backward(DeepLSTM::State* prev, DeepLSTM::State* cur, DeepLSTM::Grad& grad, const MatD& xt);
  void sgd(const DeepLSTM::Grad& grad, const double learningRate);
  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);
};

class DeepLSTM::State{
public:
  State(const DeepLSTM& dlstm);

  std::vector<LSTM::State*> lstm;

  void clear();
};

class DeepLSTM::Grad{
public:
  Grad(const DeepLSTM& dlstm);

  std::vector<LSTM::Grad> lstm;

  void init();
  double norm();
};
