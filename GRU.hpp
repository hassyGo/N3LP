#pragma once

#include "Matrix.hpp"
#include "Rand.hpp"
#include <fstream>

class GRU{
public:
  GRU(){};
  GRU(const int inputDim, const int hiddenDim);

  class State;
  class Grad;

  MatD Wxr, Whr, br;
  MatD Wxz, Whz, bz;
  MatD Wxu, Whu, bu;

  void init(Rand& rnd, const double scale = 1.0);
  virtual void forward(const MatD& xt, const GRU::State* prev, GRU::State* cur);
  virtual void backward(GRU::State* prev, GRU::State* cur, GRU::Grad& grad, const MatD& xt);
  void sgd(const GRU::Grad& grad, const double learningRate);
  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);
};

class GRU::State{
public:
  MatD h, u, r, z;
  MatD rh;

  MatD delh, delx; //for backprop

  void clear();
};

class GRU::Grad{
public:
  Grad(){}
  Grad(const GRU& gru);

  MatD Wxr, Whr, br;
  MatD Wxz, Whz, bz;
  MatD Wxu, Whu, bu;

  void init();
  double norm();

  void operator += (const GRU::Grad& grad);
  void operator /= (const double val);
};
