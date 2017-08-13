#include "SoftMax.hpp"
#include "ActFunc.hpp"
#include "Utils.hpp"

void SoftMax::calcDist(const VecD& input, VecD& output){
  output = this->bias;
  output.noalias() += this->weight.transpose()*input;
  output.array() -= output.maxCoeff(); //for numerical stability
  output = output.array().exp();
  output /= output.array().sum();
}

Real SoftMax::calcLoss(const VecD& output, const int label){
  return -log(output.coeff(label, 0));
}

Real SoftMax::calcLoss(const VecD& output, const VecD& goldOutput){
  return -(goldOutput.array()*output.array().log()).sum();
}

void SoftMax::backward(const VecD& input, const VecD& output, const int label, VecD& deltaFeature, SoftMax::Grad& grad){
  VecD delta = output;

  delta.coeffRef(label, 0) -= 1.0;
  deltaFeature = this->weight*delta;
  grad.weight += input*delta.transpose();
  grad.bias += delta;
}

void SoftMax::backward(const VecD& input, const VecD& output, const VecD& goldOutput, VecD& deltaFeature, SoftMax::Grad& grad){
  VecD delta = output-goldOutput;

  deltaFeature = this->weight*delta;
  grad.weight += input*delta.transpose();
  grad.bias += delta;
}

void SoftMax::backwardAttention(const VecD& input, const VecD& output, const VecD& deltaOut, VecD& deltaFeature, SoftMax::Grad& grad){
  VecD delta = output.array()*(deltaOut.array()-output.dot(deltaOut));

  deltaFeature = this->weight*delta;
  grad.weight += input*delta.transpose();
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

void SoftMax::operator += (const SoftMax& softmax){
  this->weight += softmax.weight;
  this->bias += softmax.bias;
}

void SoftMax::operator /= (const Real val){
  this->weight /= val;
  this->bias /= val;
}

void SoftMax::calcScore(const VecD& input, VecD& output){
  output.noalias() = this->weight.transpose()*input+this->bias;

  if (this->exception >= 0){
    output.coeffRef(this->exception, 0) = -1.0e+10;
  }
}

Real SoftMax::calcRankingLoss(const VecD& output, const int label){
  int row, col;
  VecD tmp = output;

  while (true){
    tmp.maxCoeff(&row, &col);

    if (row != label){
      break;
    }

    tmp.coeffRef(row, col) = -1.0e+10;
  }

  if (label == this->exception){
    return log(1.0+exp(this->gamma*(this->mMinus+output.coeff(row, 0))));
  }
  else {
    return
      log(1.0+exp(this->gamma*(this->mPlus-output.coeff(label, 0))))+
      log(1.0+exp(this->gamma*(this->mMinus+output.coeff(row, 0))));
  }
}

void SoftMax::backwardRankingLoss(const VecD& input, const VecD output, const int label, VecD& deltaFeature, SoftMax::Grad& grad){
  int row, col;
  VecD tmp = output;

  while (true){
    tmp.maxCoeff(&row, &col);

    if (row != label){
      break;
    }

    tmp.coeffRef(row, col) = -1.0e+10;
  }

  Real delMinus = this->gamma*ActFunc::logistic(this->gamma*(this->mMinus+output.coeff(row, 0)));

  deltaFeature = delMinus*this->weight.col(row);
  grad.weight.col(row) += delMinus*input;
  //grad.bias.coeffRef(row, 0) += delMinus;

  if (label == this->exception){
    return;
  }

  Real delPlus = -this->gamma*ActFunc::logistic(this->gamma*(this->mPlus-output.coeff(label, 0)));

  deltaFeature += delPlus*this->weight.col(label);
  grad.weight.col(label) += delPlus*input;
  //grad.bias.coeffRef(label, 0) += delPlus;
}
