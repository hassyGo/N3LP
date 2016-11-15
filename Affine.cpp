#include "Affine.hpp"
#include "Optimizer.hpp"
#include "Utils.hpp"
#include "ActFunc.hpp"

void Affine::forward(const VecD& input, VecD& output){
  output = this->bias;
  output.noalias() += this->weight*input;
  if (this->act == Affine::TANH){
    ActFunc::tanh(output);
  }
  else if (this->act == Affine::RELU){
    ActFunc::relu(output);
  }
  else {
    assert(false);
  }
}

void Affine::backward(const VecD& input, const VecD& output, const VecD& deltaOutput, VecD& deltaInput, Affine::Grad& grad){
  VecD del;
  if (this->act == Affine::TANH){
    del = ActFunc::tanhPrime(output).array()*deltaOutput.array();
  }
  else if (this->act == Affine::RELU){
    del = ActFunc::reluPrime(output).array()*deltaOutput.array();
  }
  else {
    assert(false);
  }

  deltaInput = this->weight.transpose()*del;
  grad.biasGrad += del;
  grad.weightGrad += del*input.transpose();
}
  
void Affine::init(Rand& rnd, const Real scale){
  rnd.uniform(this->weight, scale);
  this->bias.setZero();
}

void Affine::save(std::ofstream& ofs){
  Utils::save(ofs, this->weight);
  Utils::save(ofs, this->bias);
}

void Affine::load(std::ifstream& ifs){
  Utils::load(ifs, this->weight);
  Utils::load(ifs, this->bias);
}

void Affine::operator += (const Affine& affine){
  this->weight += affine.weight;
  this->bias += affine.bias;
}

void Affine::operator /= (const Real val){
  this->weight /= val;
  this->bias /= val;
}

void Affine::Grad::init(){
  this->weightGrad.setZero();
  this->biasGrad.setZero();
}

Real Affine::Grad::norm(){
  return this->weightGrad.squaredNorm()+this->biasGrad.squaredNorm();
}

void Affine::Grad::l2reg(const Real lambda, const Affine& af){
  this->weightGrad += lambda*af.weight;
}

void Affine::Grad::l2reg(const Real lambda, const Affine& af, const Affine& target){
  this->weightGrad += lambda*(af.weight-target.weight);
  this->biasGrad += lambda*(af.bias-target.bias);
}

void Affine::Grad::sgd(const Real learningRate, Affine& af){
  Optimizer::sgd(this->weightGrad, learningRate, af.weight);
  Optimizer::sgd(this->biasGrad, learningRate, af.bias);
}

void Affine::Grad::adagrad(const Real learningRate, Affine& affine, const Real initVal){
  if (this->gradHist == 0){
    this->gradHist = new Affine::Grad(affine);
    this->gradHist->weightGrad.fill(initVal);
    this->gradHist->biasGrad.fill(initVal);
  }

  Optimizer::adagrad(this->weightGrad, learningRate, this->gradHist->weightGrad, affine.weight);
  Optimizer::adagrad(this->biasGrad, learningRate, this->gradHist->biasGrad, affine.bias);
}

void Affine::Grad::momentum(const Real learningRate, const Real m, Affine& affine){
  if (this->gradHist == 0){
    this->gradHist = new Affine::Grad(affine);
    this->gradHist->weightGrad.fill(0);
    this->gradHist->biasGrad.fill(0);
  }

  Optimizer::momentum(this->weightGrad, learningRate, m, this->gradHist->weightGrad, affine.weight);
  Optimizer::momentum(this->biasGrad, learningRate, m, this->gradHist->biasGrad,affine.bias);
}

void Affine::Grad::operator += (const Affine::Grad& grad){
  this->weightGrad += grad.weightGrad;
  this->biasGrad += grad.biasGrad;
}

void Affine::Grad::operator /= (const Real val){
  this->weightGrad /= val;
  this->biasGrad /= val;
}
