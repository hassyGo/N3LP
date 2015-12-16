#include "SoftMax.hpp"

void SoftMax::calcDist(const MatD& input, MatD& output){
  output = this->bias;
  output.noalias() += this->weight*input;
  output = output.array().exp();
  output /= output.array().sum();
}

double SoftMax::calcLoss(const MatD& output, const int label){
  return -log(output.coeff(label, 0));
}

void SoftMax::backward(const MatD& input, const MatD& output, const int label, MatD& deltaFeature, SoftMax::Grad& grad){
  MatD delta = output;

  delta.coeffRef(label, 0) -= 1.0;
  deltaFeature = this->weight.transpose()*delta;
  grad.weight += delta*input.transpose();
  grad.bias += delta;
}

void SoftMax::sgd(const SoftMax::Grad& grad, const double learningRate){
  this->weight -= learningRate*grad.weight;
  this->bias -= learningRate*grad.bias;
}
