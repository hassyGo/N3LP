#pragma once

#include "Matrix.hpp"
#include "Rand.hpp"
#include <fstream>

class LSTM{
public:
  LSTM(){};
  LSTM(const int inputDim, const int hiddenDim);
  LSTM(const int inputDim, const int additionalInputDim, const int hiddenDim);

  class State;
  class Grad;

  MatD Wxi, Whi; VecD bi; //for the input gate
  MatD Wxf, Whf; VecD bf; //for the forget gate
  MatD Wxo, Who; VecD bo; //for the output gate
  MatD Wxu, Whu; VecD bu; //for the memory cell

  void init(Rand& rnd, const Real scale = 1.0);
  virtual void forward(const VecD& xt, const LSTM::State* prev, LSTM::State* cur);
  virtual void forward(const VecD& xt, LSTM::State* cur);
  virtual void backward(LSTM::State* prev, LSTM::State* cur, LSTM::Grad& grad, const VecD& xt);
  virtual void backward(LSTM::State* cur, LSTM::Grad& grad, const VecD& xt);
  void sgd(const LSTM::Grad& grad, const Real learningRate);
  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);

  MatD Wai, Waf, Wao, Wau; //for additional input
  virtual void forward(const VecD& xt, const VecD& at, const LSTM::State* prev, LSTM::State* cur);
  virtual void backward(LSTM::State* prev, LSTM::State* cur, LSTM::Grad& grad, const VecD& xt, const VecD& at);
};

class LSTM::State{
public:
  virtual ~State() {this->clear();};

  VecD h, c, u, i, f, o;
  VecD cTanh;

  VecD delh, delc, delx, dela; //for backprop

  virtual void clear();
};

class LSTM::Grad{
public:
  Grad(){}
  Grad(const LSTM& lstm);

  MatD Wxi, Whi; VecD bi;
  MatD Wxf, Whf; VecD bf;
  MatD Wxo, Who; VecD bo;
  MatD Wxu, Whu; VecD bu;

  MatD Wai, Waf, Wao, Wau;

  void init();
  Real norm();

  void operator += (const LSTM::Grad& grad);
  void operator /= (const Real val);
};
