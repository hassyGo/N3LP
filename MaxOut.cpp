#include "MaxOut.hpp"
#include "Optimizer.hpp"
#include "Utils.hpp"

void MaxOut::forward(const VecD& input, VecD& output){
  MatD tmp(this->bias[0].rows(), this->bias.size());

  for (unsigned int i = 0; i < this->bias.size(); ++i){
    tmp.col(i) = this->bias[i];
    tmp.col(i).noalias() += this->weight[i]*input;
  }

  output = VecD(tmp.rows());
  for (unsigned int i = 0; i < output.rows(); ++i){
    output.coeffRef(i, 0) = tmp.row(i).maxCoeff();
  }
}

void MaxOut::forward(const VecD& input, VecD& output, MaxOut::State& state){
  for (unsigned int i = 0; i < this->bias.size(); ++i){
    state.preOutput.col(i) = this->bias[i];
    state.preOutput.col(i).noalias() += this->weight[i]*input;
  }

  output = VecD(state.preOutput.rows());
  for (unsigned int i = 0, row, col; i < output.rows(); ++i){
    output.coeffRef(i, 0) = state.preOutput.row(i).maxCoeff(&row, &col);
    state.maxIndex.coeffRef(i, 0) = col;
  }
}

void MaxOut::backward(const VecD& input, const VecD& output, const VecD& deltaOutput, VecD& deltaInput, MaxOut::State& state, MaxOut::Grad& grad){
  VecD tmp = deltaOutput;
  deltaInput = VecD::Zero(input.rows());

  for (unsigned int i = 0, index; i < output.rows(); ++i){
    index = state.maxIndex.coeff(i, 0);
    deltaInput += this->weight[index].row(i).transpose()*tmp.coeff(i, 0);
    grad.biasGrad[index].coeffRef(i, 0) += tmp.coeff(i, 0);
    grad.weightGrad[index].row(i) += tmp.coeff(i, 0)*input.transpose();
  }
}

void MaxOut::init(Rand& rnd, const Real scale){
  for (unsigned int i = 0; i < this->weight.size(); ++i){
    rnd.uniform(this->weight[i], scale);
    this->bias[i].setZero();
  }
}

void MaxOut::save(std::ofstream& ofs){
  for (unsigned int i = 0; i < this->weight.size(); ++i){
    Utils::save(ofs, this->weight[i]);
    Utils::save(ofs, this->bias[i]);
  }
}

void MaxOut::load(std::ifstream& ifs){
  for (unsigned int i = 0; i < this->weight.size(); ++i){
    Utils::load(ifs, this->weight[i]);
    Utils::load(ifs, this->bias[i]);
  }
}

void MaxOut::Grad::init(){
  for (unsigned int i = 0; i < this->weightGrad.size(); ++i){
    this->weightGrad[i].setZero();
    this->biasGrad[i].setZero();
  }
}

Real MaxOut::Grad::norm(){
  Real res = 0.0;

  for (unsigned int i = 0; i < this->weightGrad.size(); ++i){
    res += (this->weightGrad[i].squaredNorm()+this->biasGrad[i].squaredNorm());
  }

  return res;
}

void MaxOut::Grad::l2reg(const Real lambda, const MaxOut& mx){
  for (unsigned int i = 0; i < mx.weight.size(); ++i){
    this->weightGrad[i] += lambda*mx.weight[i];
  }
}

void MaxOut::Grad::sgd(const Real learningRate, MaxOut& mx){
  for (unsigned int i = 0; i < mx.weight.size(); ++i){
    Optimizer::sgd(this->weightGrad[i], learningRate, mx.weight[i]);
    Optimizer::sgd(this->biasGrad[i], learningRate, mx.bias[i]);
  }
}

void MaxOut::Grad::operator += (const MaxOut::Grad& grad){
  for (unsigned int i = 0; i < this->weightGrad.size(); ++i){
    this->weightGrad[i] += grad.weightGrad[i];
    this->biasGrad[i] += grad.biasGrad[i];
  }
}

void MaxOut::Grad::operator /= (const Real val){
  for (unsigned int i = 0; i < this->weightGrad.size(); ++i){
    this->weightGrad[i] /= val;
    this->biasGrad[i] /= val;
  }
}
