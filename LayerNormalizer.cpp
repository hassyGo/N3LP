#include "LayerNormalizer.hpp"
#include "Utils.hpp"
#include <iostream>

LayerNormalizer::LayerNormalizer(const int dim){
  this->g = VecD(dim);
  this->b = VecD(dim);
}

void LayerNormalizer::init(){
  this->g.fill(1.0);
  this->b.fill(0.0);
}

void LayerNormalizer::forward(VecD& at, LayerNormalizer::State* state){
  const unsigned int H = at.rows();

  state->yt = at.array()-at.sum()/H;
  state->sigma = sqrt(state->yt.squaredNorm()/H);
  state->xt = (1.0/state->sigma)*this->g;
  at = this->b;
  at.array() += state->xt.array()*state->yt.array();
}

void LayerNormalizer::backward(const VecD& delat, VecD& delatOrig, LayerNormalizer::State* state, LayerNormalizer::Grad& grad){
  const unsigned int H = delat.rows();
  VecD delxt, delyt;
  Real delmu, delsigma;

  delxt = delat.array()*state->yt.array();
  delyt = delat.array()*state->xt.array();

  grad.b += delat;
  grad.g += (1.0/state->sigma)*delxt;

  delsigma = -delxt.dot(this->g)/(state->sigma*state->sigma);
  delatOrig = (delsigma/(state->sigma*H))*state->yt;
  delmu = -delyt.sum()-delatOrig.sum();
  delatOrig += delyt;
  delatOrig.array() += delmu/H;
}

void LayerNormalizer::sgd(const LayerNormalizer::Grad& grad, const Real learningRate){
  this->g -= learningRate*grad.g;
  this->b -= learningRate*grad.b;
}

void LayerNormalizer::save(std::ofstream& ofs){
  Utils::save(ofs, this->g);
  Utils::save(ofs, this->b);
}

void LayerNormalizer::load(std::ifstream& ifs){
  Utils::load(ifs, this->g);
  Utils::load(ifs, this->b);
}

void LayerNormalizer::State::clear(){
  this->xt = VecD();
  this->yt = VecD();
}

LayerNormalizer::Grad::Grad(const LayerNormalizer& ln){
  this->g = VecD::Zero(ln.g.rows());
  this->b = VecD::Zero(ln.b.rows());
}

void LayerNormalizer::Grad::init(){
  this->g.setZero();
  this->b.setZero();
}

Real LayerNormalizer::Grad::norm(){
  return this->g.squaredNorm()+this->b.squaredNorm();
}

void LayerNormalizer::Grad::operator += (const LayerNormalizer::Grad& grad){
  this->g += grad.g;
  this->b += grad.b;
}
