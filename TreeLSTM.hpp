#pragma once

#include "Matrix.hpp"
#include "Rand.hpp"
#include <fstream>

class TreeLSTM{
public:
  TreeLSTM(){};
  TreeLSTM(const int inputDim, const int hiddenDim);

  class State;
  class Grad;

  MatD Wxi, WhiL, WhiR, bi; //for the input gate
  MatD Wxfl, WhflL, WhflR, bfl; //for the left forget gate
  MatD Wxfr, WhfrL, WhfrR, bfr; //for the right forget gate
  MatD Wxo, WhoL, WhoR, bo; //for the output gate
  MatD Wxu, WhuL, WhuR, bu; //for the memory cell

  void init(Rand& rnd, const double scale = 1.0);
  void forward(const MatD& xt, TreeLSTM::State* parent, TreeLSTM::State* left, TreeLSTM::State* right);
  void backward(TreeLSTM::State* parent, TreeLSTM::State* left, TreeLSTM::State* right, TreeLSTM::Grad& grad, const MatD& xt);
  void sgd(const TreeLSTM::Grad& grad, const double learningRate);
  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);
};

class TreeLSTM::State{
public:
  MatD h, c, u, i, fl, fr, o;
  MatD cTanh;

  MatD delh, delc, delx; //for backprop

  void clear();
};

class TreeLSTM::Grad{
public:
  Grad(const TreeLSTM& tlstm);

  MatD Wxi, WhiL, WhiR, bi;
  MatD Wxfl, WhflL, WhflR, bfl;
  MatD Wxfr, WhfrL, WhfrR, bfr;
  MatD Wxo, WhoL, WhoR, bo;
  MatD Wxu, WhuL, WhuR, bu;

  void init();
  double norm();
};
