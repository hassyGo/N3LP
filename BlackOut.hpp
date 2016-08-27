#pragma once

#include "Matrix.hpp"
#include "Rand.hpp"
#include <vector>
#include <unordered_map>
#include <fstream>

class BlackOut{
public:
  BlackOut(){}
  BlackOut(const int inputDim, const int classNum, const int numSample_):
    weight(MatD::Zero(inputDim, classNum)), bias(VecD::Zero(classNum)),
    numSample(numSample_)
  {}

  class State;
  class Grad;

  Rand rnd;
  MatD weight; VecD bias;
  int numSample;
  std::vector<int> sampleDist;
  VecD distWeight;

  void initSampling(const VecD& freq, const Real alpha);
  void sampling(const int label, BlackOut::State& state);
  void calcDist(const VecD& input, VecD& output);
  void calcSampledDist(const VecD& input, VecD& output, BlackOut::State& state);
  Real calcLoss(const VecD& output, const int label);
  Real calcSampledLoss(const VecD& output);
  void backward(const VecD& input, const VecD& output, BlackOut::State& state, VecD& deltaFeature, BlackOut::Grad& grad);
  void sgd(const BlackOut::Grad& grad, const Real learningRate);
  void save(std::ofstream& ofs);
  void load(std::ifstream& ifs);
};

class BlackOut::State{
public:
  State(){};
  State(BlackOut& blackout):
    rnd(Rand(blackout.rnd.next())),
    sample(std::vector<int>(blackout.numSample+1))
  {};

  Rand rnd;
  std::vector<int> sample;
};

class BlackOut::Grad{
public:
  std::unordered_map<int, VecD> weight;
  std::unordered_map<int, Real> bias;

  void init(){
    this->weight.clear();
    this->bias.clear();
  }

  Real norm(){
    Real res = 0.0;

    for (auto it = this->weight.begin(); it != this->weight.end(); ++it){
      res += it->second.squaredNorm();
    }
    for (auto it = this->bias.begin(); it != this->bias.end(); ++it){
      res += it->second*it->second;
    }

    return res;
  }

  void operator += (const BlackOut::Grad& grad){
    for (auto it = grad.weight.begin(); it != grad.weight.end(); ++it){
      if (this->weight.count(it->first)){
	this->weight.at(it->first) += it->second;
      }
      else {
	this->weight[it->first] = it->second;
      }
    }
    for (auto it = grad.bias.begin(); it != grad.bias.end(); ++it){
      if (this->bias.count(it->first)){
	this->bias.at(it->first) += it->second;
      }
      else {
	this->bias[it->first] = it->second;
      }
    }
  }
};
