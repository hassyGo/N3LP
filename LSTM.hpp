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

  Real dropoutRateX;
  Real dropoutRateA;
  Real dropoutRateH;
  
  MatD Wxi, Whi; VecD bi; //for the input gate
  MatD Wxf, Whf; VecD bf; //for the forget gate
  MatD Wxo, Who; VecD bo; //for the output gate
  MatD Wxu, Whu; VecD bu; //for the memory cell

  void init(Rand& rnd, const Real scale = 1.0);
  void activate(LSTM::State* cur);
  void activate(const LSTM::State* prev, LSTM::State* cur);
  virtual void forward(const VecD& xt, const LSTM::State* prev, LSTM::State* cur);
  virtual void forward(const VecD& xt, LSTM::State* cur);
  virtual void backward(LSTM::State* prev, LSTM::State* cur, LSTM::Grad& grad, const VecD& xt);
  virtual void backward(LSTM::State* cur, LSTM::Grad& grad, const VecD& xt);
  void sgd(const LSTM::Grad& grad, const Real learningRate);
  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);

  MatD Wai, Waf, Wao, Wau; //for additional input
  virtual void forward(const VecD& xt, const VecD& at, const LSTM::State* prev, LSTM::State* cur);
  virtual void forward(const VecD& xt, const VecD& at, LSTM::State* cur);
  virtual void backward(LSTM::State* prev, LSTM::State* cur, LSTM::Grad& grad, const VecD& xt, const VecD& at);
  virtual void backward(LSTM::State* cur, LSTM::Grad& grad, const VecD& xt, const VecD& at);

  void dropout(bool isTest);
  void operator += (const LSTM& lstm);
  void operator /= (const Real val);
};

class LSTM::State{
public:
  virtual ~State() {this->clear();};

  VecD h, c, u, i, f, o;
  VecD cTanh;
  VecD maskXt, maskAt, maskHt; //for dropout

  VecD delh, delc, delx, dela; //for backprop

  virtual void clear();
};

class LSTM::Grad{
public:
  Grad(): gradHist(0) {}
  Grad(const LSTM& lstm);

  LSTM::Grad* gradHist;
  
  MatD Wxi, Whi; VecD bi;
  MatD Wxf, Whf; VecD bf;
  MatD Wxo, Who; VecD bo;
  MatD Wxu, Whu; VecD bu;

  MatD Wai, Waf, Wao, Wau;

  void init();
  Real norm();
  void l2reg(const Real lambda, const LSTM& lstm);
  void l2reg(const Real lambda, const LSTM& lstm, const LSTM& target);
  void sgd(const Real learningRate, LSTM& lstm);
  void adagrad(const Real learningRate, LSTM& lstm, const Real initVal = 1.0);
  void momentum(const Real learningRate, const Real m, LSTM& lstm);

  void operator += (const LSTM::Grad& grad);
  void operator /= (const Real val);
};
