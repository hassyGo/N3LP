#include "LSTM.hpp"
#include "ActFunc.hpp"
#include "Utils.hpp"

LSTM::LSTM(const int inputDim, const int hiddenDim){
  this->Wxi = MatD(hiddenDim, inputDim);
  this->Whi = MatD(hiddenDim, hiddenDim);
  this->bi = MatD::Zero(hiddenDim, 1);

  this->Wxf = MatD(hiddenDim, inputDim);
  this->Whf = MatD(hiddenDim, hiddenDim);
  this->bf = MatD::Zero(hiddenDim, 1);

  this->Wxo = MatD(hiddenDim, inputDim);
  this->Who = MatD(hiddenDim, hiddenDim);
  this->bo = MatD::Zero(hiddenDim, 1);

  this->Wxu = MatD(hiddenDim, inputDim);
  this->Whu = MatD(hiddenDim, hiddenDim);
  this->bu = MatD::Zero(hiddenDim, 1);
}

void LSTM::init(Rand& rnd, const double scale){
  rnd.uniform(this->Wxi, scale);
  rnd.uniform(this->Whi, scale);

  rnd.uniform(this->Wxf, scale);
  rnd.uniform(this->Whf, scale);
  
  rnd.uniform(this->Wxo, scale);
  rnd.uniform(this->Who, scale);

  rnd.uniform(this->Wxu, scale);
  rnd.uniform(this->Whu, scale);
}

void LSTM::forward(const MatD& xt, const LSTM::State* prev, LSTM::State* cur){
  cur->i = this->bi + this->Wxi*xt + this->Whi*prev->h;
  cur->f = this->bf + this->Wxf*xt + this->Whf*prev->h;
  cur->o = this->bo + this->Wxo*xt + this->Who*prev->h;
  cur->u = this->bu + this->Wxu*xt + this->Whu*prev->h;

  ActFunc::logistic(cur->i);
  ActFunc::logistic(cur->f);
  ActFunc::logistic(cur->o);
  ActFunc::tanh(cur->u);
  cur->c = cur->i.array()*cur->u.array() + cur->f.array()*prev->c.array();
  cur->cTanh = cur->c;
  ActFunc::tanh(cur->cTanh);
  cur->h = cur->o.array()*cur->cTanh.array();
}

void LSTM::backward(LSTM::State* prev, LSTM::State* cur, LSTM::Grad& grad, const MatD& xt){
  MatD delo, deli, delu, delf;

  cur->delc.array() += ActFunc::tanhPrime(cur->cTanh).array()*cur->delh.array()*cur->o.array();
  prev->delc.array() += cur->delc.array()*cur->f.array();
  delo = ActFunc::logisticPrime(cur->o).array()*cur->delh.array()*cur->cTanh.array();
  deli = ActFunc::logisticPrime(cur->i).array()*cur->delc.array()*cur->u.array();
  delf = ActFunc::logisticPrime(cur->f).array()*cur->delc.array()*prev->c.array();
  delu = ActFunc::tanhPrime(cur->u).array()*cur->delc.array()*cur->i.array();
  
  cur->delx =
    this->Wxi.transpose()*deli+
    this->Wxf.transpose()*delf+
    this->Wxo.transpose()*delo+
    this->Wxu.transpose()*delu;

  prev->delh.noalias() +=
    this->Whi.transpose()*deli+
    this->Whf.transpose()*delf+
    this->Who.transpose()*delo+
    this->Whu.transpose()*delu;

  grad.Wxi.noalias() += deli*xt.transpose();
  grad.Whi.noalias() += deli*prev->h.transpose();

  grad.Wxf.noalias() += delf*xt.transpose();
  grad.Whf.noalias() += delf*prev->h.transpose();

  grad.Wxo.noalias() += delo*xt.transpose();
  grad.Who.noalias() += delo*prev->h.transpose();

  grad.Wxu.noalias() += delu*xt.transpose();
  grad.Whu.noalias() += delu*prev->h.transpose();

  grad.bi += deli;
  grad.bf += delf;
  grad.bo += delo;
  grad.bu += delu;
}

void LSTM::sgd(const LSTM::Grad& grad, const double learningRate){
  this->Wxi -= learningRate*grad.Wxi;
  this->Whi -= learningRate*grad.Whi;
  this->bi -= learningRate*grad.bi;

  this->Wxf -= learningRate*grad.Wxf;
  this->Whf -= learningRate*grad.Whf;
  this->bf -= learningRate*grad.bf;

  this->Wxo -= learningRate*grad.Wxo;
  this->Who -= learningRate*grad.Who;
  this->bo -= learningRate*grad.bo;

  this->Wxu -= learningRate*grad.Wxu;
  this->Whu -= learningRate*grad.Whu;
  this->bu -= learningRate*grad.bu;
}

void LSTM::save(std::ofstream& ofs){
  Utils::save(ofs, this->Wxi); Utils::save(ofs, this->Whi); Utils::save(ofs, this->bi);
  Utils::save(ofs, this->Wxf); Utils::save(ofs, this->Whf); Utils::save(ofs, this->bf);
  Utils::save(ofs, this->Wxo); Utils::save(ofs, this->Who); Utils::save(ofs, this->bo);
  Utils::save(ofs, this->Wxu); Utils::save(ofs, this->Whu); Utils::save(ofs, this->bu);
}

void LSTM::load(std::ifstream& ifs){
  Utils::load(ifs, this->Wxi); Utils::load(ifs, this->Whi); Utils::load(ifs, this->bi);
  Utils::load(ifs, this->Wxf); Utils::load(ifs, this->Whf); Utils::load(ifs, this->bf);
  Utils::load(ifs, this->Wxo); Utils::load(ifs, this->Who); Utils::load(ifs, this->bo);
  Utils::load(ifs, this->Wxu); Utils::load(ifs, this->Whu); Utils::load(ifs, this->bu);
}

void LSTM::State::clear(){
  this->h = MatD();
  this->c = MatD();
  this->u = MatD();
  this->i = MatD();
  this->f = MatD();
  this->o = MatD();
  this->cTanh = MatD();
  this->delh = MatD();
  this->delc = MatD();
  this->delx = MatD();
}

LSTM::Grad::Grad(const LSTM& lstm){
  this->Wxi = MatD::Zero(lstm.Wxi.rows(), lstm.Wxi.cols());
  this->Whi = MatD::Zero(lstm.Whi.rows(), lstm.Whi.cols());
  this->bi = MatD::Zero(lstm.bi.rows(), lstm.bi.cols());

  this->Wxf = MatD::Zero(lstm.Wxf.rows(), lstm.Wxf.cols());
  this->Whf = MatD::Zero(lstm.Whf.rows(), lstm.Whf.cols());
  this->bf = MatD::Zero(lstm.bf.rows(), lstm.bf.cols());

  this->Wxo = MatD::Zero(lstm.Wxo.rows(), lstm.Wxo.cols());
  this->Who = MatD::Zero(lstm.Who.rows(), lstm.Who.cols());
  this->bo = MatD::Zero(lstm.bo.rows(), lstm.bo.cols());

  this->Wxu = MatD::Zero(lstm.Wxu.rows(), lstm.Wxu.cols());
  this->Whu = MatD::Zero(lstm.Whu.rows(), lstm.Whu.cols());
  this->bu = MatD::Zero(lstm.bu.rows(), lstm.bu.cols());
};

void LSTM::Grad::init(){
  this->Wxi.setZero(); this->Whi.setZero(); this->bi.setZero();
  this->Wxf.setZero(); this->Whf.setZero(); this->bf.setZero();
  this->Wxo.setZero(); this->Who.setZero(); this->bo.setZero();
  this->Wxu.setZero(); this->Whu.setZero(); this->bu.setZero();
}

double LSTM::Grad::norm(){
  return
    this->Wxi.squaredNorm()+this->Whi.squaredNorm()+this->bi.squaredNorm()+
    this->Wxf.squaredNorm()+this->Whf.squaredNorm()+this->bf.squaredNorm()+
    this->Wxo.squaredNorm()+this->Who.squaredNorm()+this->bo.squaredNorm()+
    this->Wxu.squaredNorm()+this->Whu.squaredNorm()+this->bu.squaredNorm();
}
