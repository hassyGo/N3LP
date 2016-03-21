#include "BlackOut.hpp"
#include "Utils.hpp"
#include <iostream>

void BlackOut::initSampling(const VecD& freq, const Real alpha){
  const int sum = freq.array().sum();
  const int total = sum;

  this->distWeight = freq/sum;
  this->distWeight = this->distWeight.array().pow(alpha);
  this->distWeight /= this->distWeight.sum();

  for (int i = 0; i < this->distWeight.rows(); ++i){
    for (int j = 0, num = (int)(total*this->distWeight.coeff(i, 0)); j < num; ++j){
      this->sampleDist.push_back(i);
    }
  }

  this->rnd.shuffle(this->sampleDist);
  this->distWeight = this->distWeight.array().inverse();
}

void BlackOut::sampling(const int label, BlackOut::State& state){
  const unsigned int SIZE = this->sampleDist.size();

  state.sample[0] = label;

  for (int i = 1, neg; i <= this->numSample; ++i){
    do {
      neg = this->sampleDist[(state.rnd.next() >> 16)%SIZE];
    } while (neg == label);

    state.sample[i] = neg;
  }
}

void BlackOut::calcDist(const VecD& input, VecD& output){
  output = this->bias;
  output.noalias() += this->weight.transpose()*input;
  output.array() -= output.maxCoeff(); //for numerical stability
  output = output.array().exp();
  output /= output.array().sum();
}

void BlackOut::calcSampledDist(const VecD& input, VecD& output, BlackOut::State& state){
  output = VecD(this->numSample+1);

  for (int i = 0; i < this->numSample+1; ++i){
    output.coeffRef(i, 0) =
      this->bias.coeff(state.sample[i], 0)+
      this->weight.col(state.sample[i]).dot(input);
  }

  output.array() -= output.maxCoeff();

  for (int i = 0; i < this->numSample+1; ++i){
    output.coeffRef(i, 0) =
      this->distWeight.coeff(state.sample[i], 0)*
      exp(output.coeff(i, 0));
  }

  output /= output.array().sum();
}

Real BlackOut::calcLoss(const VecD& output, const int label){
  return -log(output.coeff(label, 0));
}

Real BlackOut::calcSampledLoss(const VecD& output){
  Real loss = -log(output.coeff(0, 0));

  for (int i = 1; i < output.rows(); ++i){
    loss -= log(1.0-output.coeff(i, 0));
  }

  return loss;
}

void BlackOut::backward(const VecD& input, const VecD& output, BlackOut::State& state, VecD& deltaFeature, BlackOut::Grad& grad){
  const VecD fragment = (1.0-output.block(1, 0, this->numSample, 1).array()).inverse();
  const Real sum = fragment.array().sum();
  VecD delta(this->numSample+1);

  delta.coeffRef(0, 0) = (this->numSample+1-sum)*output.coeff(0, 0)-1.0;

  for (int i = 1; i < this->numSample+1; ++i){
    delta.coeffRef(i, 0) = (this->numSample+1-(sum-fragment.coeff(i-1, 0)))*output.coeff(i, 0);
  }

  deltaFeature.noalias() = delta.coeff(0, 0)*this->weight.col(state.sample[0]);

  for (int i = 1; i < this->numSample+1; ++i){
    deltaFeature.noalias() += delta.coeff(i, 0)*this->weight.col(state.sample[i]);
  }
  
  for (int i = 0; i < this->numSample+1; ++i){
    if (grad.weight.count(state.sample[i])){
      grad.weight.at(state.sample[i]).noalias() += delta.coeff(i, 0)*input;
      grad.bias.at(state.sample[i]) += delta.coeff(i, 0);
    }
    else {
      grad.weight[state.sample[i]].noalias() = delta.coeff(i, 0)*input;
      grad.bias[state.sample[i]] = delta.coeff(i, 0);
    }
  }
}

void BlackOut::sgd(const BlackOut::Grad& grad, const Real learningRate){
  for (auto it = grad.weight.begin(); it != grad.weight.end(); ++it){
    this->weight.col(it->first) -= learningRate*it->second;
  }
  for (auto it = grad.bias.begin(); it != grad.bias.end(); ++it){
    this->bias.coeffRef(it->first, 0) -= learningRate*it->second;
  }
}

void BlackOut::save(std::ofstream& ofs){
  Utils::save(ofs, this->weight);
  Utils::save(ofs, this->bias);
}

void BlackOut::load(std::ifstream& ifs){
  Utils::load(ifs, this->weight);
  Utils::load(ifs, this->bias);
}
