#include "GRU.hpp"
#include "ActFunc.hpp"
#include "Utils.hpp"
#include <Eigen/SVD>

GRU::GRU(const int inputDim, const int hiddenDim){
  this->Wxr = MatD(hiddenDim, inputDim);
  this->Whr = MatD(hiddenDim, hiddenDim);
  this->br = VecD::Zero(hiddenDim);

  this->Wxz = MatD(hiddenDim, inputDim);
  this->Whz = MatD(hiddenDim, hiddenDim);
  this->bz = VecD::Zero(hiddenDim);

  this->Wxu = MatD(hiddenDim, inputDim);
  this->Whu = MatD(hiddenDim, hiddenDim);
  this->bu = VecD::Zero(hiddenDim);
}

void GRU::init(Rand& rnd, const Real scale){
  rnd.uniform(this->Wxr, scale);
  rnd.uniform(this->Whr, scale);

  rnd.uniform(this->Wxz, scale);
  rnd.uniform(this->Whz, scale);
  
  rnd.uniform(this->Wxu, scale);
  rnd.uniform(this->Whu, scale);

  this->Whr = Eigen::JacobiSVD<MatD>(this->Whr, Eigen::ComputeFullV|Eigen::ComputeFullU).matrixU();
  this->Whz = Eigen::JacobiSVD<MatD>(this->Whz, Eigen::ComputeFullV|Eigen::ComputeFullU).matrixU();
  this->Whu = Eigen::JacobiSVD<MatD>(this->Whu, Eigen::ComputeFullV|Eigen::ComputeFullU).matrixU();
}

void GRU::forward(const VecD& xt, const GRU::State* prev, GRU::State* cur){
  cur->r = this->br + this->Wxr*xt + this->Whr*prev->h;
  cur->z = this->bz + this->Wxz*xt + this->Whz*prev->h;

  ActFunc::logistic(cur->r);
  ActFunc::logistic(cur->z);

  cur->rh = cur->r.array()*prev->h.array();
  cur->u = this->bu + this->Wxu*xt + this->Whu*cur->rh;
  ActFunc::tanh(cur->u);
  cur->h = (1.0-cur->z.array())*prev->h.array() + cur->z.array()*cur->u.array();
}

void GRU::backward(GRU::State* prev, GRU::State* cur, GRU::Grad& grad, const VecD& xt){
  VecD delr, delz, delu, delrh;

  delz = ActFunc::logisticPrime(cur->z).array()*cur->delh.array()*(cur->u-prev->h).array();
  delu = ActFunc::tanhPrime(cur->u).array()*cur->delh.array()*cur->z.array();
  delrh = this->Whu.transpose()*delu;
  delr = ActFunc::logisticPrime(cur->r).array()*delrh.array()*prev->h.array();

  cur->delx =
    this->Wxr.transpose()*delr+
    this->Wxz.transpose()*delz+
    this->Wxu.transpose()*delu;

  prev->delh.noalias() +=
    this->Whr.transpose()*delr+
    this->Whz.transpose()*delz;
  prev->delh.array() +=
    delrh.array()*cur->r.array()+
    cur->delh.array()*(1.0-cur->z.array());

  grad.Wxr.noalias() += delr*xt.transpose();
  grad.Whr.noalias() += delr*prev->h.transpose();

  grad.Wxz.noalias() += delz*xt.transpose();
  grad.Whz.noalias() += delz*prev->h.transpose();

  grad.Wxu.noalias() += delu*xt.transpose();
  grad.Whu.noalias() += delu*cur->rh.transpose();

  grad.br += delr;
  grad.bz += delz;
  grad.bu += delu;
}

void GRU::sgd(const GRU::Grad& grad, const Real learningRate){
  this->Wxr -= learningRate*grad.Wxr;
  this->Whr -= learningRate*grad.Whr;
  this->br -= learningRate*grad.br;

  this->Wxz -= learningRate*grad.Wxz;
  this->Whz -= learningRate*grad.Whz;
  this->bz -= learningRate*grad.bz;

  this->Wxu -= learningRate*grad.Wxu;
  this->Whu -= learningRate*grad.Whu;
  this->bu -= learningRate*grad.bu;
}

void GRU::save(std::ofstream& ofs){
  Utils::save(ofs, this->Wxr); Utils::save(ofs, this->Whr); Utils::save(ofs, this->br);
  Utils::save(ofs, this->Wxz); Utils::save(ofs, this->Whz); Utils::save(ofs, this->bz);
  Utils::save(ofs, this->Wxu); Utils::save(ofs, this->Whu); Utils::save(ofs, this->bu);
}

void GRU::load(std::ifstream& ifs){
  Utils::load(ifs, this->Wxr); Utils::load(ifs, this->Whr); Utils::load(ifs, this->br);
  Utils::load(ifs, this->Wxz); Utils::load(ifs, this->Whz); Utils::load(ifs, this->bz);
  Utils::load(ifs, this->Wxu); Utils::load(ifs, this->Whu); Utils::load(ifs, this->bu);
}

void GRU::State::clear(){
  this->h = VecD();
  this->u = VecD();
  this->r = VecD();
  this->z = VecD();
  this->rh = VecD();
  this->delh = VecD();
  this->delx = VecD();
}

GRU::Grad::Grad(const GRU& gru){
  this->Wxr = MatD::Zero(gru.Wxr.rows(), gru.Wxr.cols());
  this->Whr = MatD::Zero(gru.Whr.rows(), gru.Whr.cols());
  this->br = VecD::Zero(gru.br.rows());

  this->Wxz = MatD::Zero(gru.Wxz.rows(), gru.Wxz.cols());
  this->Whz = MatD::Zero(gru.Whz.rows(), gru.Whz.cols());
  this->bz = VecD::Zero(gru.bz.rows());

  this->Wxu = MatD::Zero(gru.Wxu.rows(), gru.Wxu.cols());
  this->Whu = MatD::Zero(gru.Whu.rows(), gru.Whu.cols());
  this->bu = VecD::Zero(gru.bu.rows());
};

void GRU::Grad::init(){
  this->Wxr.setZero(); this->Whr.setZero(); this->br.setZero();
  this->Wxz.setZero(); this->Whz.setZero(); this->bz.setZero();
  this->Wxu.setZero(); this->Whu.setZero(); this->bu.setZero();
}

Real GRU::Grad::norm(){
  return
    this->Wxr.squaredNorm()+this->Whr.squaredNorm()+this->br.squaredNorm()+
    this->Wxz.squaredNorm()+this->Whz.squaredNorm()+this->bz.squaredNorm()+
    this->Wxu.squaredNorm()+this->Whu.squaredNorm()+this->bu.squaredNorm();
}

void GRU::Grad::operator += (const GRU::Grad& grad){
  this->Wxr += grad.Wxr; this->Whr += grad.Whr; this->br += grad.br;
  this->Wxz += grad.Wxz; this->Whz += grad.Whz; this->bz += grad.bz;
  this->Wxu += grad.Wxu; this->Whu += grad.Whu; this->bu += grad.bu;
}
