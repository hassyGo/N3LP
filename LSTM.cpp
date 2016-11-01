#include "LSTM.hpp"
#include "ActFunc.hpp"
#include "Utils.hpp"

LSTM::LSTM(const int inputDim, const int hiddenDim){
  this->Wxi = MatD(hiddenDim, inputDim);
  this->Whi = MatD(hiddenDim, hiddenDim);
  this->bi = VecD::Zero(hiddenDim);

  this->Wxf = MatD(hiddenDim, inputDim);
  this->Whf = MatD(hiddenDim, hiddenDim);
  this->bf = VecD::Zero(hiddenDim);

  this->Wxo = MatD(hiddenDim, inputDim);
  this->Who = MatD(hiddenDim, hiddenDim);
  this->bo = VecD::Zero(hiddenDim);

  this->Wxu = MatD(hiddenDim, inputDim);
  this->Whu = MatD(hiddenDim, hiddenDim);
  this->bu = VecD::Zero(hiddenDim);
}

LSTM::LSTM(const int inputDim, const int additionalInputDim, const int hiddenDim){
  this->Wxi = MatD(hiddenDim, inputDim);
  this->Whi = MatD(hiddenDim, hiddenDim);
  this->bi = VecD::Zero(hiddenDim);

  this->Wxf = MatD(hiddenDim, inputDim);
  this->Whf = MatD(hiddenDim, hiddenDim);
  this->bf = VecD::Zero(hiddenDim);

  this->Wxo = MatD(hiddenDim, inputDim);
  this->Who = MatD(hiddenDim, hiddenDim);
  this->bo = VecD::Zero(hiddenDim);

  this->Wxu = MatD(hiddenDim, inputDim);
  this->Whu = MatD(hiddenDim, hiddenDim);
  this->bu = VecD::Zero(hiddenDim);

  this->Wai = MatD(hiddenDim, additionalInputDim);
  this->Waf = MatD(hiddenDim, additionalInputDim);
  this->Wao = MatD(hiddenDim, additionalInputDim);
  this->Wau = MatD(hiddenDim, additionalInputDim);
}

void LSTM::init(Rand& rnd, const Real scale){
  rnd.uniform(this->Wxi, scale);
  rnd.uniform(this->Whi, scale);

  rnd.uniform(this->Wxf, scale);
  rnd.uniform(this->Whf, scale);
  
  rnd.uniform(this->Wxo, scale);
  rnd.uniform(this->Who, scale);

  rnd.uniform(this->Wxu, scale);
  rnd.uniform(this->Whu, scale);

  rnd.uniform(this->Wai, scale);
  rnd.uniform(this->Waf, scale);
  rnd.uniform(this->Wao, scale);
  rnd.uniform(this->Wau, scale);
}

void LSTM::activate(const LSTM::State* prev, LSTM::State* cur){
  ActFunc::logistic(cur->i);
  ActFunc::logistic(cur->f);
  ActFunc::logistic(cur->o);
  ActFunc::tanh(cur->u);
  cur->c = cur->i.array()*cur->u.array() + cur->f.array()*prev->c.array();
  cur->cTanh = cur->c;
  ActFunc::tanh(cur->cTanh);
  cur->h = cur->o.array()*cur->cTanh.array();
}

void LSTM::activate(LSTM::State* cur){
  ActFunc::logistic(cur->i);
  ActFunc::logistic(cur->o);
  ActFunc::tanh(cur->u);
  cur->c = cur->i.array()*cur->u.array();
  cur->cTanh = cur->c;
  ActFunc::tanh(cur->cTanh);
  cur->h = cur->o.array()*cur->cTanh.array();
}

void LSTM::forward(const VecD& xt, const LSTM::State* prev, LSTM::State* cur){
  cur->i = this->bi;
  cur->i.noalias() += this->Wxi*xt + this->Whi*prev->h;
  cur->f = this->bf;
  cur->f.noalias() += this->Wxf*xt + this->Whf*prev->h;
  cur->o = this->bo;
  cur->o.noalias() += this->Wxo*xt + this->Who*prev->h;
  cur->u = this->bu;
  cur->u.noalias() += this->Wxu*xt + this->Whu*prev->h;

  this->activate(prev, cur);
}
void LSTM::forward(const VecD& xt, LSTM::State* cur){
  cur->i = this->bi;
  cur->i.noalias() += this->Wxi*xt;
  cur->o = this->bo;
  cur->o.noalias() += this->Wxo*xt;
  cur->u = this->bu;
  cur->u.noalias() += this->Wxu*xt;

  this->activate(cur);
}
void LSTM::backward(LSTM::State* prev, LSTM::State* cur, LSTM::Grad& grad, const VecD& xt){
  VecD delo, deli, delu, delf;

  cur->delc.array() += ActFunc::tanhPrime(cur->cTanh).array()*cur->delh.array()*cur->o.array();
  prev->delc.array() += cur->delc.array()*cur->f.array();
  delo = ActFunc::logisticPrime(cur->o).array()*cur->delh.array()*cur->cTanh.array();
  deli = ActFunc::logisticPrime(cur->i).array()*cur->delc.array()*cur->u.array();
  delf = ActFunc::logisticPrime(cur->f).array()*cur->delc.array()*prev->c.array();
  delu = ActFunc::tanhPrime(cur->u).array()*cur->delc.array()*cur->i.array();
  
  cur->delx.noalias() =
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
void LSTM::backward(LSTM::State* cur, LSTM::Grad& grad, const VecD& xt){
  VecD delo, deli, delu;

  cur->delc.array() += ActFunc::tanhPrime(cur->cTanh).array()*cur->delh.array()*cur->o.array();
  delo = ActFunc::logisticPrime(cur->o).array()*cur->delh.array()*cur->cTanh.array();
  deli = ActFunc::logisticPrime(cur->i).array()*cur->delc.array()*cur->u.array();
  delu = ActFunc::tanhPrime(cur->u).array()*cur->delc.array()*cur->i.array();
  
  cur->delx.noalias() =
    this->Wxi.transpose()*deli+
    this->Wxo.transpose()*delo+
    this->Wxu.transpose()*delu;

  grad.Wxi.noalias() += deli*xt.transpose();
  grad.Wxo.noalias() += delo*xt.transpose();
  grad.Wxu.noalias() += delu*xt.transpose();

  grad.bi += deli;
  grad.bo += delo;
  grad.bu += delu;
}

void LSTM::forward(const VecD& xt, const VecD& at, const LSTM::State* prev, LSTM::State* cur){
  cur->i = this->bi;
  cur->i.noalias() += this->Wxi*xt + this->Whi*prev->h + this->Wai*at;
  cur->f = this->bf;
  cur->f.noalias() += this->Wxf*xt + this->Whf*prev->h + this->Waf*at;
  cur->o = this->bo;
  cur->o.noalias() += this->Wxo*xt + this->Who*prev->h + this->Wao*at;
  cur->u = this->bu;
  cur->u.noalias() += this->Wxu*xt + this->Whu*prev->h + this->Wau*at;

  this->activate(prev, cur);
}
void LSTM::forward(const VecD& xt, const VecD& at, LSTM::State* cur){
  cur->i = this->bi;
  cur->i.noalias() += this->Wxi*xt + this->Wai*at;
  cur->f = this->bf;
  cur->f.noalias() += this->Wxf*xt + this->Waf*at;
  cur->o = this->bo;
  cur->o.noalias() += this->Wxo*xt + this->Wao*at;
  cur->u = this->bu;
  cur->u.noalias() += this->Wxu*xt + this->Wau*at;

  this->activate(cur);
}
void LSTM::backward(LSTM::State* prev, LSTM::State* cur, LSTM::Grad& grad, const VecD& xt, const VecD& at){
  VecD delo, deli, delu, delf;

  cur->delc.array() += ActFunc::tanhPrime(cur->cTanh).array()*cur->delh.array()*cur->o.array();
  prev->delc.array() += cur->delc.array()*cur->f.array();
  delo = ActFunc::logisticPrime(cur->o).array()*cur->delh.array()*cur->cTanh.array();
  deli = ActFunc::logisticPrime(cur->i).array()*cur->delc.array()*cur->u.array();
  delf = ActFunc::logisticPrime(cur->f).array()*cur->delc.array()*prev->c.array();
  delu = ActFunc::tanhPrime(cur->u).array()*cur->delc.array()*cur->i.array();
  
  cur->delx.noalias() =
    this->Wxi.transpose()*deli+
    this->Wxf.transpose()*delf+
    this->Wxo.transpose()*delo+
    this->Wxu.transpose()*delu;

  prev->delh.noalias() +=
    this->Whi.transpose()*deli+
    this->Whf.transpose()*delf+
    this->Who.transpose()*delo+
    this->Whu.transpose()*delu;

  cur->dela.noalias() =
    this->Wai.transpose()*deli+
    this->Waf.transpose()*delf+
    this->Wao.transpose()*delo+
    this->Wau.transpose()*delu;

  grad.Wxi.noalias() += deli*xt.transpose();
  grad.Whi.noalias() += deli*prev->h.transpose();

  grad.Wxf.noalias() += delf*xt.transpose();
  grad.Whf.noalias() += delf*prev->h.transpose();

  grad.Wxo.noalias() += delo*xt.transpose();
  grad.Who.noalias() += delo*prev->h.transpose();

  grad.Wxu.noalias() += delu*xt.transpose();
  grad.Whu.noalias() += delu*prev->h.transpose();

  grad.Wai.noalias() += deli*at.transpose();
  grad.Waf.noalias() += delf*at.transpose();
  grad.Wao.noalias() += delo*at.transpose();
  grad.Wau.noalias() += delu*at.transpose();

  grad.bi += deli;
  grad.bf += delf;
  grad.bo += delo;
  grad.bu += delu;
}
void LSTM::backward(LSTM::State* cur, LSTM::Grad& grad, const VecD& xt, const VecD& at){
  VecD delo, deli, delu;

  cur->delc.array() += ActFunc::tanhPrime(cur->cTanh).array()*cur->delh.array()*cur->o.array();
  delo = ActFunc::logisticPrime(cur->o).array()*cur->delh.array()*cur->cTanh.array();
  deli = ActFunc::logisticPrime(cur->i).array()*cur->delc.array()*cur->u.array();
  delu = ActFunc::tanhPrime(cur->u).array()*cur->delc.array()*cur->i.array();
  
  cur->delx.noalias() =
    this->Wxi.transpose()*deli+
    this->Wxo.transpose()*delo+
    this->Wxu.transpose()*delu;

  cur->dela.noalias() =
    this->Wai.transpose()*deli+
    this->Wao.transpose()*delo+
    this->Wau.transpose()*delu;

  grad.Wxi.noalias() += deli*xt.transpose();
  grad.Wxo.noalias() += delo*xt.transpose();
  grad.Wxu.noalias() += delu*xt.transpose();

  grad.Wai.noalias() += deli*at.transpose();
  grad.Wao.noalias() += delo*at.transpose();
  grad.Wau.noalias() += delu*at.transpose();

  grad.bi += deli;
  grad.bo += delo;
  grad.bu += delu;
}

void LSTM::sgd(const LSTM::Grad& grad, const Real learningRate){
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

  this->Wai -= learningRate*grad.Wai;
  this->Waf -= learningRate*grad.Waf;
  this->Wao -= learningRate*grad.Wao;
  this->Wau -= learningRate*grad.Wau;
}

void LSTM::save(std::ofstream& ofs){
  Utils::save(ofs, this->Wxi); Utils::save(ofs, this->Whi); Utils::save(ofs, this->bi);
  Utils::save(ofs, this->Wxf); Utils::save(ofs, this->Whf); Utils::save(ofs, this->bf);
  Utils::save(ofs, this->Wxo); Utils::save(ofs, this->Who); Utils::save(ofs, this->bo);
  Utils::save(ofs, this->Wxu); Utils::save(ofs, this->Whu); Utils::save(ofs, this->bu);
  Utils::save(ofs, this->Wai); Utils::save(ofs, this->Waf); Utils::save(ofs, this->Wao); Utils::save(ofs, this->Wau);
}

void LSTM::load(std::ifstream& ifs){
  Utils::load(ifs, this->Wxi); Utils::load(ifs, this->Whi); Utils::load(ifs, this->bi);
  Utils::load(ifs, this->Wxf); Utils::load(ifs, this->Whf); Utils::load(ifs, this->bf);
  Utils::load(ifs, this->Wxo); Utils::load(ifs, this->Who); Utils::load(ifs, this->bo);
  Utils::load(ifs, this->Wxu); Utils::load(ifs, this->Whu); Utils::load(ifs, this->bu);
  Utils::load(ifs, this->Wai); Utils::load(ifs, this->Waf); Utils::load(ifs, this->Wao); Utils::load(ifs, this->Wau);
}

void LSTM::State::clear(){
  this->h = VecD();
  this->c = VecD();
  this->u = VecD();
  this->i = VecD();
  this->f = VecD();
  this->o = VecD();
  this->cTanh = VecD();
  this->delh = VecD();
  this->delc = VecD();
  this->delx = VecD();
  this->dela = VecD();
}

LSTM::Grad::Grad(const LSTM& lstm){
  this->Wxi = MatD::Zero(lstm.Wxi.rows(), lstm.Wxi.cols());
  this->Whi = MatD::Zero(lstm.Whi.rows(), lstm.Whi.cols());
  this->bi = VecD::Zero(lstm.bi.rows());

  this->Wxf = MatD::Zero(lstm.Wxf.rows(), lstm.Wxf.cols());
  this->Whf = MatD::Zero(lstm.Whf.rows(), lstm.Whf.cols());
  this->bf = VecD::Zero(lstm.bf.rows());

  this->Wxo = MatD::Zero(lstm.Wxo.rows(), lstm.Wxo.cols());
  this->Who = MatD::Zero(lstm.Who.rows(), lstm.Who.cols());
  this->bo = VecD::Zero(lstm.bo.rows());

  this->Wxu = MatD::Zero(lstm.Wxu.rows(), lstm.Wxu.cols());
  this->Whu = MatD::Zero(lstm.Whu.rows(), lstm.Whu.cols());
  this->bu = VecD::Zero(lstm.bu.rows());

  this->Wai = MatD::Zero(lstm.Wai.rows(), lstm.Wai.cols());
  this->Waf = MatD::Zero(lstm.Waf.rows(), lstm.Waf.cols());
  this->Wao = MatD::Zero(lstm.Wao.rows(), lstm.Wao.cols());
  this->Wau = MatD::Zero(lstm.Wau.rows(), lstm.Wau.cols());
};

void LSTM::Grad::init(){
  this->Wxi.setZero(); this->Whi.setZero(); this->bi.setZero();
  this->Wxf.setZero(); this->Whf.setZero(); this->bf.setZero();
  this->Wxo.setZero(); this->Who.setZero(); this->bo.setZero();
  this->Wxu.setZero(); this->Whu.setZero(); this->bu.setZero();
  this->Wai.setZero(); this->Waf.setZero(); this->Wao.setZero(); this->Wau.setZero();
}

Real LSTM::Grad::norm(){
  return
    this->Wxi.squaredNorm()+this->Whi.squaredNorm()+this->bi.squaredNorm()+
    this->Wxf.squaredNorm()+this->Whf.squaredNorm()+this->bf.squaredNorm()+
    this->Wxo.squaredNorm()+this->Who.squaredNorm()+this->bo.squaredNorm()+
    this->Wxu.squaredNorm()+this->Whu.squaredNorm()+this->bu.squaredNorm()+
    this->Wai.squaredNorm()+this->Waf.squaredNorm()+this->Wao.squaredNorm()+this->Wau.squaredNorm();
}

void LSTM::Grad::operator += (const LSTM::Grad& grad){
  this->Wxi += grad.Wxi; this->Whi += grad.Whi; this->bi += grad.bi;
  this->Wxf += grad.Wxf; this->Whf += grad.Whf; this->bf += grad.bf;
  this->Wxo += grad.Wxo; this->Who += grad.Who; this->bo += grad.bo;
  this->Wxu += grad.Wxu; this->Whu += grad.Whu; this->bu += grad.bu;
  this->Wai += grad.Wai; this->Waf += grad.Waf; this->Wao += grad.Wao; this->Wau += grad.Wau;
}

void LSTM::Grad::operator /= (const Real val){
  this->Wxi /= val; this->Whi /= val; this->bi /= val;
  this->Wxf /= val; this->Whf /= val; this->bf /= val;
  this->Wxo /= val; this->Who /= val; this->bo /= val;
  this->Wxu /= val; this->Whu /= val; this->bu /= val;
  this->Wai /= val; this->Waf /= val; this->Wao /= val; this->Wau /= val;
}
