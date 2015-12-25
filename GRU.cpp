#include "GRU.hpp"
#include "ActFunc.hpp"
#include "Utils.hpp"
#include <Eigen/SVD>

GRU::GRU(const int inputDim, const int hiddenDim){
  this->Wxr = MatD(hiddenDim, inputDim);
  this->Whr = MatD(hiddenDim, hiddenDim);
  this->br = MatD::Zero(hiddenDim, 1);

  this->Wxz = MatD(hiddenDim, inputDim);
  this->Whz = MatD(hiddenDim, hiddenDim);
  this->bz = MatD::Zero(hiddenDim, 1);

  this->Wxu = MatD(hiddenDim, inputDim);
  this->Whu = MatD(hiddenDim, hiddenDim);
  this->bu = MatD::Zero(hiddenDim, 1);
}

void GRU::init(Rand& rnd, const double scale){
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

void GRU::forward(const MatD& xt, const GRU::State* prev, GRU::State* cur){
  cur->r = this->br + this->Wxr*xt + this->Whr*prev->h;
  cur->z = this->bz + this->Wxz*xt + this->Whz*prev->h;

  ActFunc::logistic(cur->r);
  ActFunc::logistic(cur->z);

  cur->rh = cur->r.array()*prev->h.array();
  cur->u = this->bu + this->Wxu*xt + this->Whu*cur->rh;
  ActFunc::tanh(cur->u);
  cur->h = (1.0-cur->z.array())*prev->h.array() + cur->z.array()*cur->u.array();
}

void GRU::backward(GRU::State* prev, GRU::State* cur, GRU::Grad& grad, const MatD& xt){
  MatD delr, delz, delu, delrh;

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

void GRU::sgd(const GRU::Grad& grad, const double learningRate){
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
  this->h = MatD();
  this->u = MatD();
  this->r = MatD();
  this->z = MatD();
  this->rh = MatD();
  this->delh = MatD();
  this->delx = MatD();
}

GRU::Grad::Grad(const GRU& gru){
  this->Wxr = MatD::Zero(gru.Wxr.rows(), gru.Wxr.cols());
  this->Whr = MatD::Zero(gru.Whr.rows(), gru.Whr.cols());
  this->br = MatD::Zero(gru.br.rows(), gru.br.cols());

  this->Wxz = MatD::Zero(gru.Wxz.rows(), gru.Wxz.cols());
  this->Whz = MatD::Zero(gru.Whz.rows(), gru.Whz.cols());
  this->bz = MatD::Zero(gru.bz.rows(), gru.bz.cols());

  this->Wxu = MatD::Zero(gru.Wxu.rows(), gru.Wxu.cols());
  this->Whu = MatD::Zero(gru.Whu.rows(), gru.Whu.cols());
  this->bu = MatD::Zero(gru.bu.rows(), gru.bu.cols());
};

void GRU::Grad::init(){
  this->Wxr.setZero(); this->Whr.setZero(); this->br.setZero();
  this->Wxz.setZero(); this->Whz.setZero(); this->bz.setZero();
  this->Wxu.setZero(); this->Whu.setZero(); this->bu.setZero();
}

double GRU::Grad::norm(){
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

void GRU::Grad::operator /= (const double val){
  this->Wxr /= val; this->Whr /= val; this->br /= val;
  this->Wxz /= val; this->Whz /= val; this->bz /= val;
  this->Wxu /= val; this->Whu /= val; this->bu /= val;
}
