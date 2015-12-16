#pragma once

#include "Matrix.hpp"
#include "Rand.hpp"
#include <fstream>

class LSTM{
public:
  LSTM(){};
  LSTM(const int inputDim, const int hiddenDim);

  class State;
  class Grad;

  MatD Wxi, Whi, bi; //for the input gate
  MatD Wxf, Whf, bf; //for the forget gate
  MatD Wxo, Who, bo; //for the output gate
  MatD Wxu, Whu, bu; //for the memory cell

  void init(Rand& rnd, const double scale = 1.0);
  virtual void forward(const MatD& xt, const LSTM::State* prev, LSTM::State* cur);
  virtual void backward(LSTM::State* prev, LSTM::State* cur, LSTM::Grad& grad, const MatD& xt);
  void sgd(const LSTM::Grad& grad, const double learningRate);
  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);
};

class LSTM::State{
public:
  MatD h, c, u, i, f, o;
  MatD cTanh;

  MatD delh, delc, delx; //for backprop

  void clear();
};

class LSTM::Grad{
public:
  Grad(const LSTM& lstm);

  MatD Wxi, Whi, bi;
  MatD Wxf, Whf, bf;
  MatD Wxo, Who, bo;
  MatD Wxu, Whu, bu;

  void init();
  double norm();
};
