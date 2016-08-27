#pragma once

#include "Matrix.hpp"
#include "Rand.hpp"
#include <fstream>
#include "LSTM.hpp"

class TreeLSTM{
public:
  TreeLSTM(){};
  TreeLSTM(const int inputDim, const int hiddenDim);

  class State;
  class Grad;

  MatD Wxi, WhiL, WhiR; VecD bi; //for the input gate
  MatD Wxfl, WhflL, WhflR; VecD bfl; //for the left forget gate
  MatD Wxfr, WhfrL, WhfrR; VecD bfr; //for the right forget gate
  MatD Wxo, WhoL, WhoR; VecD bo; //for the output gate
  MatD Wxu, WhuL, WhuR; VecD bu; //for the memory cell

  void init(Rand& rnd, const Real scale = 1.0);
  void forward(const VecD& xt, TreeLSTM::State* parent, LSTM::State* left, LSTM::State* right);
  void backward(TreeLSTM::State* parent, LSTM::State* left, LSTM::State* right, TreeLSTM::Grad& grad, const VecD& xt);
  void forward(TreeLSTM::State* parent, LSTM::State* left, LSTM::State* right);
  void backward(TreeLSTM::State* parent, LSTM::State* left, LSTM::State* right, TreeLSTM::Grad& grad);
  void sgd(const TreeLSTM::Grad& grad, const Real learningRate);
  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);
};

class TreeLSTM::State: public LSTM::State{
public:
  ~State() {this->clear();};

  VecD fl, fr;

  void clear();
};

class TreeLSTM::Grad{
public:
  Grad(){}
  Grad(const TreeLSTM& tlstm);

  MatD Wxi, WhiL, WhiR; VecD bi;
  MatD Wxfl, WhflL, WhflR; VecD bfl;
  MatD Wxfr, WhfrL, WhfrR; VecD bfr;
  MatD Wxo, WhoL, WhoR; VecD bo;
  MatD Wxu, WhuL, WhuR; VecD bu;

  void init();
  Real norm();

  void operator += (const TreeLSTM::Grad& grad);
};
