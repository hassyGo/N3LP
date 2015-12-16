#pragma once

#include "LSTM.hpp"
#include "Vocabulary.hpp"
#include "SoftMax.hpp"

class EncDec{
public:
  class Data;
  class Grad;

  EncDec(Vocabulary& sourceVoc_, Vocabulary& targetVoc_, std::vector<EncDec::Data*>& trainData_, const int inputDim, const int hiddenDim);

  Rand rnd;
  Vocabulary& sourceVoc;
  Vocabulary& targetVoc;
  std::vector<EncDec::Data*>& trainData;
  LSTM enc, dec;
  SoftMax softmax;
  MatD sourceEmbed;
  MatD targetEmbed;
  MatD zeros;

  void encode(const std::vector<int>& src, std::vector<LSTM::State*>& encState);
  void translate(const std::vector<int>& src);
  void train(EncDec::Data* data, LSTM::Grad& lstmSrcGrad, LSTM::Grad& lstmTgtGrad, SoftMax::Grad& softmaxGrad, EncDec::Grad& embedGrad, double& loss);
  void train(const double learningRate);
  static void demo(const std::string& src, const std::string& tgt);
};

class EncDec::Data{
public:
  std::vector<int> src, tgt;
  std::vector<LSTM::State*> encState, decState;
};

class EncDec::Grad{
public:
  std::unordered_map<int, MatD> sourceEmbed, targetEmbed;

  void init(){
    this->sourceEmbed.clear();
    this->targetEmbed.clear();
  }

  double norm(){
    double res = 0.0;

    for (auto it = this->sourceEmbed.begin(); it != this->sourceEmbed.end(); ++it){
      res += it->second.squaredNorm();
    }
    for (auto it = this->targetEmbed.begin(); it != this->targetEmbed.end(); ++it){
      res += it->second.squaredNorm();
    }

    return res;
  }
};
