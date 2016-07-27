#pragma once

#include "LSTM.hpp"
#include "LayerNormalizer.hpp"

class LnLSTM : public LSTM{
public:
  LnLSTM(){}
  LnLSTM(const int inputDim, const int hiddenDim);
  LnLSTM(const int inputDim, const int additionalInputDim, const int hiddenDim);

  class State;
  class Grad;

  LayerNormalizer lnh, lnx, lnc, lna;

  void init(Rand& rnd, const Real scale = 1.0);
  void forward(const VecD& xt, const LSTM::State* prev, LSTM::State* cur);
  void forward(const VecD& xt, LSTM::State* cur);
  void backward(LSTM::State* prev, LSTM::State* cur, LSTM::Grad& grad, const VecD& xt);
  void backward(LSTM::State* cur, LSTM::Grad& grad, const VecD& xt);
  void sgd(const LnLSTM::Grad& grad, const Real learningRate);
  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);

  void forward(const VecD& xt, const VecD& at, const LSTM::State* prev, LSTM::State* cur);
  void backward(LSTM::State* prev, LSTM::State* cur, LSTM::Grad& grad, const VecD& xt, const VecD& at);
};

class LnLSTM::State: public LSTM::State{
public:
  State(): lnsh(new LayerNormalizer::State), lnsx(new LayerNormalizer::State), lnsc(new LayerNormalizer::State), lnsa(new LayerNormalizer::State){}
  ~State() {
    this->clear();
    delete this->lnsh;
    delete this->lnsx;
    delete this->lnsc;
    delete this->lnsa;}

  LayerNormalizer::State* lnsh;
  LayerNormalizer::State* lnsx;
  LayerNormalizer::State* lnsc;
  LayerNormalizer::State* lnsa;
  VecD lnhConcat, lnxConcat, lnaConcat;
  VecD delConcat;

  void clear();
};

class LnLSTM::Grad: public LSTM::Grad{
public:
  Grad(){}
  Grad(const LnLSTM& lnlstm);

  void init();
  Real norm();

  void operator += (const LnLSTM::Grad& grad);

  LayerNormalizer::Grad lnh, lnx, lnc, lna;
};
