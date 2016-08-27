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

  MatD Wxr, Whr; VecD br;
  MatD Wxz, Whz; VecD bz;
  MatD Wxu, Whu; VecD bu;

  void init(Rand& rnd, const Real scale = 1.0);
  virtual void forward(const VecD& xt, const GRU::State* prev, GRU::State* cur);
  virtual void backward(GRU::State* prev, GRU::State* cur, GRU::Grad& grad, const VecD& xt);
  void sgd(const GRU::Grad& grad, const Real learningRate);
  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);
};

class GRU::State{
public:
  VecD h, u, r, z;
  VecD rh;

  VecD delh, delx; //for backprop

  void clear();
};

class GRU::Grad{
public:
  Grad(){}
  Grad(const GRU& gru);

  MatD Wxr, Whr; VecD br;
  MatD Wxz, Whz; VecD bz;
  MatD Wxu, Whu; VecD bu;

  void init();
  Real norm();

  void operator += (const GRU::Grad& grad);
};
