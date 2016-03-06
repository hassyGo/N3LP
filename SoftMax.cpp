#include "SoftMax.hpp"
#include "Utils.hpp"

void SoftMax::calcDist(const VecD& input, VecD& output){
  output = this->bias;
  output.noalias() += this->weight*input;
  output.array() -= output.maxCoeff(); //for numerical stability
  output = output.array().exp();
  output /= output.array().sum();
}

Real SoftMax::calcLoss(const VecD& output, const int label){
  return -log(output.coeff(label, 0));
}

void SoftMax::backward(const VecD& input, const VecD& output, const int label, VecD& deltaFeature, SoftMax::Grad& grad){
  VecD delta = output;

  delta.coeffRef(label, 0) -= 1.0;
  deltaFeature = this->weight.transpose()*delta;
  grad.weight += delta*input.transpose();
  grad.bias += delta;
}

void SoftMax::sgd(const SoftMax::Grad& grad, const Real learningRate){
  this->weight -= learningRate*grad.weight;
  this->bias -= learningRate*grad.bias;
}

void SoftMax::save(std::ofstream& ofs){
  Utils::save(ofs, this->weight);
  Utils::save(ofs, this->bias);
}

void SoftMax::load(std::ifstream& ifs){
  Utils::load(ifs, this->weight);
  Utils::load(ifs, this->bias);
}
