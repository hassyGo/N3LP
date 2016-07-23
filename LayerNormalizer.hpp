#pragma once

#include "Matrix.hpp"
#include "Rand.hpp"

class LayerNormalizer{
public:
  LayerNormalizer(){}
  LayerNormalizer(const int dim);

  class State;
  class Grad;

  VecD g, b;

  void init(Rand& rnd, const Real scale = 1.0);
  void forward(VecD& at, LayerNormalizer::State* state);
  void backward(const VecD& delat, VecD& delatOrig, LayerNormalizer::State* state, LayerNormalizer::Grad& grad);
  void sgd(const LayerNormalizer::Grad& grad, const Real learningRate);

  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);
};

class LayerNormalizer::State{
public:
  Real sigma;
  VecD xt, yt;

  void clear();
};

class LayerNormalizer::Grad{
public:
  Grad(){}
  Grad(const LayerNormalizer& ln);

  VecD g, b;

  void init();
  Real norm();

  void operator += (const LayerNormalizer::Grad& grad);
};
