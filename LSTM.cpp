#include "LSTM.hpp"
#include "ActFunc.hpp"
#include "Utils.hpp"
#include "Optimizer.hpp"

LSTM::LSTM(const int inputDim, const int hiddenDim):
  dropoutRateX(-1.0), dropoutRateA(-1.0), dropoutRateH(-1.0)
{
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

LSTM::LSTM(const int inputDim, const int additionalInputDim, const int hiddenDim):
  dropoutRateX(-1.0), dropoutRateA(-1.0), dropoutRateH(-1.0)
{
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
  cur->f = this->bf;
  cur->o = this->bo;
  cur->u = this->bu;

  if (this->dropoutRateX > 0.0){
    VecD masked = xt.array()*cur->maskXt.array();
    cur->i.noalias() += this->Wxi*masked;
    cur->f.noalias() += this->Wxf*masked;
    cur->o.noalias() += this->Wxo*masked;
    cur->u.noalias() += this->Wxu*masked;
  }
  else {
    cur->i.noalias() += this->Wxi*xt;
    cur->f.noalias() += this->Wxf*xt;
    cur->o.noalias() += this->Wxo*xt;
    cur->u.noalias() += this->Wxu*xt;
  }

  if (this->dropoutRateH > 0.0){
    VecD masked = prev->h.array()*cur->maskHt.array();
    cur->i.noalias() += this->Whi*masked;
    cur->f.noalias() += this->Whf*masked;
    cur->o.noalias() += this->Who*masked;
    cur->u.noalias() += this->Whu*masked;
  }
  else {
    cur->i.noalias() += this->Whi*prev->h;
    cur->f.noalias() += this->Whf*prev->h;
    cur->o.noalias() += this->Who*prev->h;
    cur->u.noalias() += this->Whu*prev->h;
  }
  
  this->activate(prev, cur);
}
void LSTM::forward(const VecD& xt, LSTM::State* cur){
  cur->i = this->bi;
  cur->o = this->bo;
  cur->u = this->bu;

  if (this->dropoutRateX > 0.0){
    VecD masked = xt.array()*cur->maskXt.array();
    cur->i.noalias() += this->Wxi*masked;
    cur->o.noalias() += this->Wxo*masked;
    cur->u.noalias() += this->Wxu*masked;
  }
  else {
    cur->i.noalias() += this->Wxi*xt;
    cur->o.noalias() += this->Wxo*xt;
    cur->u.noalias() += this->Wxu*xt;
  }

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

  if (this->dropoutRateX > 0.0){
    VecD masked = xt.array()*cur->maskXt.array();
    grad.Wxi.noalias() += deli*masked.transpose();
    grad.Wxf.noalias() += delf*masked.transpose();
    grad.Wxo.noalias() += delo*masked.transpose();
    grad.Wxu.noalias() += delu*masked.transpose();
    cur->delx.array() *= cur->maskXt.array();
  }
  else {
    grad.Wxi.noalias() += deli*xt.transpose();
    grad.Wxf.noalias() += delf*xt.transpose();
    grad.Wxo.noalias() += delo*xt.transpose();
    grad.Wxu.noalias() += delu*xt.transpose();
  }

  if (this->dropoutRateH > 0.0){
    VecD masked = prev->h.array()*cur->maskHt.array();
    grad.Whi.noalias() += deli*masked.transpose();
    grad.Whf.noalias() += delf*masked.transpose();
    grad.Who.noalias() += delo*masked.transpose();
    grad.Whu.noalias() += delu*masked.transpose();
    prev->delh.array() *= cur->maskHt.array();
  }
  else {  
    grad.Whi.noalias() += deli*prev->h.transpose();
    grad.Whf.noalias() += delf*prev->h.transpose();
    grad.Who.noalias() += delo*prev->h.transpose();
    grad.Whu.noalias() += delu*prev->h.transpose();
  }

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

  if (this->dropoutRateX > 0.0){
    VecD masked = xt.array()*cur->maskXt.array();
    grad.Wxi.noalias() += deli*masked.transpose();
    grad.Wxo.noalias() += delo*masked.transpose();
    grad.Wxu.noalias() += delu*masked.transpose();
    cur->delx.array() *= cur->maskXt.array();
  }
  else {
    grad.Wxi.noalias() += deli*xt.transpose();
    grad.Wxo.noalias() += delo*xt.transpose();
    grad.Wxu.noalias() += delu*xt.transpose();
  }
  
  grad.bi += deli;
  grad.bo += delo;
  grad.bu += delu;
}

void LSTM::forward(const VecD& xt, const VecD& at, const LSTM::State* prev, LSTM::State* cur){
  cur->i = this->bi;
  cur->f = this->bf;
  cur->o = this->bo;
  cur->u = this->bu;

  if (this->dropoutRateX > 0.0){
    VecD maskedXt = xt.array()*cur->maskXt.array();
    cur->i.noalias() += this->Wxi*maskedXt;
    cur->f.noalias() += this->Wxf*maskedXt;
    cur->o.noalias() += this->Wxo*maskedXt;
    cur->u.noalias() += this->Wxu*maskedXt;
  }
  else {
    cur->i.noalias() += this->Wxi*xt;
    cur->f.noalias() += this->Wxf*xt;
    cur->o.noalias() += this->Wxo*xt;
    cur->u.noalias() += this->Wxu*xt;
  }

  if (this->dropoutRateA > 0.0){
    VecD maskedAt = at.array()*cur->maskAt.array();
    cur->i.noalias() += this->Wai*maskedAt;
    cur->f.noalias() += this->Waf*maskedAt;
    cur->o.noalias() += this->Wao*maskedAt;
    cur->u.noalias() += this->Wau*maskedAt;
  }
  else {
    cur->i.noalias() += this->Wai*at;
    cur->f.noalias() += this->Waf*at;
    cur->o.noalias() += this->Wao*at;
    cur->u.noalias() += this->Wau*at;
  }

  if (this->dropoutRateH > 0.0){
    VecD maskedHt = prev->h.array()*cur->maskHt.array();
    cur->i.noalias() += this->Whi*maskedHt;
    cur->f.noalias() += this->Whf*maskedHt;
    cur->o.noalias() += this->Who*maskedHt;
    cur->u.noalias() += this->Whu*maskedHt;
  }
  else {
    cur->i.noalias() += this->Whi*prev->h;
    cur->f.noalias() += this->Whf*prev->h;
    cur->o.noalias() += this->Who*prev->h;
    cur->u.noalias() += this->Whu*prev->h;
  }
  
  this->activate(prev, cur);
}
void LSTM::forward(const VecD& xt, const VecD& at, LSTM::State* cur){
  cur->i = this->bi;
  cur->f = this->bf;
  cur->o = this->bo;
  cur->u = this->bu;

  if (this->dropoutRateX > 0.0){
    VecD maskedXt = xt.array()*cur->maskXt.array();
    cur->i.noalias() += this->Wxi*maskedXt;
    cur->f.noalias() += this->Wxf*maskedXt;
    cur->o.noalias() += this->Wxo*maskedXt;
    cur->u.noalias() += this->Wxu*maskedXt;
  }
  else {
    cur->i.noalias() += this->Wxi*xt;
    cur->f.noalias() += this->Wxf*xt;
    cur->o.noalias() += this->Wxo*xt;
    cur->u.noalias() += this->Wxu*xt;
  }

  if (this->dropoutRateA > 0.0){
    VecD maskedAt = at.array()*cur->maskAt.array();
    cur->i.noalias() += this->Wai*maskedAt;
    cur->f.noalias() += this->Waf*maskedAt;
    cur->o.noalias() += this->Wao*maskedAt;
    cur->u.noalias() += this->Wau*maskedAt;
  }
  else {
    cur->i.noalias() += this->Wai*at;
    cur->f.noalias() += this->Waf*at;
    cur->o.noalias() += this->Wao*at;
    cur->u.noalias() += this->Wau*at;
  }
  
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

  if (this->dropoutRateX > 0.0){
    VecD maskedXt = xt.array()*cur->maskXt.array();
    grad.Wxi.noalias() += deli*maskedXt.transpose();
    grad.Wxf.noalias() += delf*maskedXt.transpose();
    grad.Wxo.noalias() += delo*maskedXt.transpose();
    grad.Wxu.noalias() += delu*maskedXt.transpose();
    
    cur->delx.array() *= cur->maskXt.array();
  }
  else {
    grad.Wxi.noalias() += deli*xt.transpose();
    grad.Wxf.noalias() += delf*xt.transpose();
    grad.Wxo.noalias() += delo*xt.transpose();
    grad.Wxu.noalias() += delu*xt.transpose();
  }

  if (this->dropoutRateA > 0.0){
    VecD maskedAt = at.array()*cur->maskAt.array();
    grad.Wai.noalias() += deli*maskedAt.transpose();
    grad.Waf.noalias() += delf*maskedAt.transpose();
    grad.Wao.noalias() += delo*maskedAt.transpose();
    grad.Wau.noalias() += delu*maskedAt.transpose();
    
    cur->dela.array() *= cur->maskAt.array();
  }
  else {
    grad.Wai.noalias() += deli*at.transpose();
    grad.Waf.noalias() += delf*at.transpose();
    grad.Wao.noalias() += delo*at.transpose();
    grad.Wau.noalias() += delu*at.transpose();
  }

  if (this->dropoutRateH > 0.0){
    VecD maskedHt = prev->h.array()*cur->maskHt.array();
    grad.Whi.noalias() += deli*maskedHt.transpose();
    grad.Whf.noalias() += delf*maskedHt.transpose();
    grad.Who.noalias() += delo*maskedHt.transpose();
    grad.Whu.noalias() += delu*maskedHt.transpose();
    
    prev->delh.array() *= cur->maskHt.array();
  }
  else {
    grad.Whi.noalias() += deli*prev->h.transpose();
    grad.Whf.noalias() += delf*prev->h.transpose();
    grad.Who.noalias() += delo*prev->h.transpose();
    grad.Whu.noalias() += delu*prev->h.transpose();
  }
  
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

  if (this->dropoutRateX > 0.0){
    VecD maskedXt = xt.array()*cur->maskXt.array();
    grad.Wxi.noalias() += deli*maskedXt.transpose();
    grad.Wxo.noalias() += delo*maskedXt.transpose();
    grad.Wxu.noalias() += delu*maskedXt.transpose();
    
    cur->delx.array() *= cur->maskXt.array();
  }
  else {
    grad.Wxi.noalias() += deli*xt.transpose();
    grad.Wxo.noalias() += delo*xt.transpose();
    grad.Wxu.noalias() += delu*xt.transpose();
  }

  if (this->dropoutRateA > 0.0){
    VecD maskedAt = at.array()*cur->maskAt.array();
    grad.Wai.noalias() += deli*maskedAt.transpose();
    grad.Wao.noalias() += delo*maskedAt.transpose();
    grad.Wau.noalias() += delu*maskedAt.transpose();
    
    cur->dela.array() *= cur->maskAt.array();
  }
  else {
    grad.Wai.noalias() += deli*at.transpose();
    grad.Wao.noalias() += delo*at.transpose();
    grad.Wau.noalias() += delu*at.transpose();
  }
  
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

void LSTM::dropout(bool isTest){
  if (isTest){
    this->Wxi *= this->dropoutRateX;
    this->Wxf *= this->dropoutRateX;
    this->Wxo *= this->dropoutRateX;
    this->Wxu *= this->dropoutRateX;

    this->Wai *= this->dropoutRateA;
    this->Waf *= this->dropoutRateA;
    this->Wao *= this->dropoutRateA;
    this->Wau *= this->dropoutRateA;

    this->Whi *= this->dropoutRateH;
    this->Whf *= this->dropoutRateH;
    this->Who *= this->dropoutRateH;
    this->Whu *= this->dropoutRateH;
  }
  else {
    this->Wxi *= 1.0/this->dropoutRateX;
    this->Wxf *= 1.0/this->dropoutRateX;
    this->Wxo *= 1.0/this->dropoutRateX;

    this->Wxu *= 1.0/this->dropoutRateX;
    this->Wai *= 1.0/this->dropoutRateA;
    this->Waf *= 1.0/this->dropoutRateA;
    this->Wao *= 1.0/this->dropoutRateA;
    this->Wau *= 1.0/this->dropoutRateA;

    this->Whi *= 1.0/this->dropoutRateH;
    this->Whf *= 1.0/this->dropoutRateH;
    this->Who *= 1.0/this->dropoutRateH;
    this->Whu *= 1.0/this->dropoutRateH;
  }
}

void LSTM::operator += (const LSTM& lstm){
  this->Wxi += lstm.Wxi; this->Wxf += lstm.Wxf; this->Wxo += lstm.Wxo; this->Wxu += lstm.Wxu;
  this->Whi += lstm.Whi; this->Whf += lstm.Whf; this->Who += lstm.Who; this->Whu += lstm.Whu;
  this->Wai += lstm.Wai; this->Waf += lstm.Waf; this->Wao += lstm.Wao; this->Wau += lstm.Wau;
  this->bi += lstm.bi; this->bf += lstm.bf; this->bo += lstm.bo; this->bu += lstm.bu;
}

void LSTM::operator /= (const Real val){
  this->Wxi /= val; this->Wxf /= val; this->Wxo /= val; this->Wxu /= val;
  this->Whi /= val; this->Whf /= val; this->Who /= val; this->Whu /= val;
  this->Wai /= val; this->Waf /= val; this->Wao /= val; this->Wau /= val;
  this->bi /= val; this->bf /= val; this->bo /= val; this->bu /= val;
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

LSTM::Grad::Grad(const LSTM& lstm):
  gradHist(0)
{
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

void LSTM::Grad::l2reg(const Real lambda, const LSTM& lstm){
  this->Wxi += lambda*lstm.Wxi; this->Whi += lambda*lstm.Whi; this->Wai += lambda*lstm.Wai;
  this->Wxf += lambda*lstm.Wxf; this->Whf += lambda*lstm.Whf; this->Waf += lambda*lstm.Waf;
  this->Wxo += lambda*lstm.Wxo; this->Who += lambda*lstm.Who; this->Wao += lambda*lstm.Wao;
  this->Wxu += lambda*lstm.Wxu; this->Whu += lambda*lstm.Whu; this->Wau += lambda*lstm.Wau;
}

void LSTM::Grad::l2reg(const Real lambda, const LSTM& lstm, const LSTM& target){
  this->Wxi += lambda*(lstm.Wxi-target.Wxi); this->Whi += lambda*(lstm.Whi-target.Whi); this->Wai += lambda*(lstm.Wai-target.Wai); this->bi += lambda*(lstm.bi-target.bi);
  this->Wxf += lambda*(lstm.Wxf-target.Wxf); this->Whf += lambda*(lstm.Whf-target.Whf); this->Waf += lambda*(lstm.Waf-target.Waf); this->bf += lambda*(lstm.bf-target.bf);
  this->Wxo += lambda*(lstm.Wxo-target.Wxo); this->Who += lambda*(lstm.Who-target.Who); this->Wao += lambda*(lstm.Wao-target.Wao); this->bo += lambda*(lstm.bo-target.bo);
  this->Wxu += lambda*(lstm.Wxu-target.Wxu); this->Whu += lambda*(lstm.Whu-target.Whu); this->Wau += lambda*(lstm.Wau-target.Wau); this->bu += lambda*(lstm.bu-target.bu);
}

void LSTM::Grad::sgd(const Real learningRate, LSTM& lstm){
  Optimizer::sgd(this->Wxi, learningRate, lstm.Wxi);
  Optimizer::sgd(this->Wxf, learningRate, lstm.Wxf);
  Optimizer::sgd(this->Wxo, learningRate, lstm.Wxo);
  Optimizer::sgd(this->Wxu, learningRate, lstm.Wxu);

  Optimizer::sgd(this->Whi, learningRate, lstm.Whi);
  Optimizer::sgd(this->Whf, learningRate, lstm.Whf);
  Optimizer::sgd(this->Who, learningRate, lstm.Who);
  Optimizer::sgd(this->Whu, learningRate, lstm.Whu);

  Optimizer::sgd(this->Wai, learningRate, lstm.Wai);
  Optimizer::sgd(this->Waf, learningRate, lstm.Waf);
  Optimizer::sgd(this->Wao, learningRate, lstm.Wao);
  Optimizer::sgd(this->Wau, learningRate, lstm.Wau);

  Optimizer::sgd(this->bi, learningRate, lstm.bi);
  Optimizer::sgd(this->bf, learningRate, lstm.bf);
  Optimizer::sgd(this->bo, learningRate, lstm.bo);
  Optimizer::sgd(this->bu, learningRate, lstm.bu);
}

void LSTM::Grad::adagrad(const Real learningRate, LSTM& lstm, const Real initVal){
  if (this->gradHist == 0){
    this->gradHist = new LSTM::Grad(lstm);
    this->gradHist->Wxi.fill(initVal);
    this->gradHist->Wxf.fill(initVal);
    this->gradHist->Wxo.fill(initVal);
    this->gradHist->Wxu.fill(initVal);

    this->gradHist->Whi.fill(initVal);
    this->gradHist->Whf.fill(initVal);
    this->gradHist->Who.fill(initVal);
    this->gradHist->Whu.fill(initVal);
    
    this->gradHist->Wai.fill(initVal);
    this->gradHist->Waf.fill(initVal);
    this->gradHist->Wao.fill(initVal);
    this->gradHist->Wau.fill(initVal);

    this->gradHist->bi.fill(initVal);
    this->gradHist->bf.fill(initVal);
    this->gradHist->bo.fill(initVal);
    this->gradHist->bu.fill(initVal);
  }

  Optimizer::adagrad(this->Wxi, learningRate, this->gradHist->Wxi, lstm.Wxi);
  Optimizer::adagrad(this->Wxf, learningRate, this->gradHist->Wxf, lstm.Wxf);
  Optimizer::adagrad(this->Wxo, learningRate, this->gradHist->Wxo, lstm.Wxo);
  Optimizer::adagrad(this->Wxu, learningRate, this->gradHist->Wxu, lstm.Wxu);

  Optimizer::adagrad(this->Whi, learningRate, this->gradHist->Whi, lstm.Whi);
  Optimizer::adagrad(this->Whf, learningRate, this->gradHist->Whf, lstm.Whf);
  Optimizer::adagrad(this->Who, learningRate, this->gradHist->Who, lstm.Who);
  Optimizer::adagrad(this->Whu, learningRate, this->gradHist->Whu, lstm.Whu);

  Optimizer::adagrad(this->Wai, learningRate, this->gradHist->Wai, lstm.Wai);
  Optimizer::adagrad(this->Waf, learningRate, this->gradHist->Waf, lstm.Waf);
  Optimizer::adagrad(this->Wao, learningRate, this->gradHist->Wao, lstm.Wao);
  Optimizer::adagrad(this->Wau, learningRate, this->gradHist->Wau, lstm.Wau);

  Optimizer::adagrad(this->bi, learningRate, this->gradHist->bi, lstm.bi);
  Optimizer::adagrad(this->bf, learningRate, this->gradHist->bf, lstm.bf);
  Optimizer::adagrad(this->bo, learningRate, this->gradHist->bo, lstm.bo);
  Optimizer::adagrad(this->bu, learningRate, this->gradHist->bu, lstm.bu);
}

void LSTM::Grad::momentum(const Real learningRate, const Real m, LSTM& lstm){
  if (this->gradHist == 0){
    const Real initVal = 0.0;
    
    this->gradHist = new LSTM::Grad(lstm);
    this->gradHist->Wxi.fill(initVal);
    this->gradHist->Wxf.fill(initVal);
    this->gradHist->Wxo.fill(initVal);
    this->gradHist->Wxu.fill(initVal);

    this->gradHist->Whi.fill(initVal);
    this->gradHist->Whf.fill(initVal);
    this->gradHist->Who.fill(initVal);
    this->gradHist->Whu.fill(initVal);
    
    this->gradHist->Wai.fill(initVal);
    this->gradHist->Waf.fill(initVal);
    this->gradHist->Wao.fill(initVal);
    this->gradHist->Wau.fill(initVal);

    this->gradHist->bi.fill(initVal);
    this->gradHist->bf.fill(initVal);
    this->gradHist->bo.fill(initVal);
    this->gradHist->bu.fill(initVal);
  }

  Optimizer::momentum(this->Wxi, learningRate, m, this->gradHist->Wxi, lstm.Wxi);
  Optimizer::momentum(this->Wxf, learningRate, m, this->gradHist->Wxf, lstm.Wxf);
  Optimizer::momentum(this->Wxo, learningRate, m, this->gradHist->Wxo, lstm.Wxo);
  Optimizer::momentum(this->Wxu, learningRate, m, this->gradHist->Wxu, lstm.Wxu);

  Optimizer::momentum(this->Whi, learningRate, m, this->gradHist->Whi, lstm.Whi);
  Optimizer::momentum(this->Whf, learningRate, m, this->gradHist->Whf, lstm.Whf);
  Optimizer::momentum(this->Who, learningRate, m, this->gradHist->Who, lstm.Who);
  Optimizer::momentum(this->Whu, learningRate, m, this->gradHist->Whu, lstm.Whu);

  Optimizer::momentum(this->Wai, learningRate, m, this->gradHist->Wai, lstm.Wai);
  Optimizer::momentum(this->Waf, learningRate, m, this->gradHist->Waf, lstm.Waf);
  Optimizer::momentum(this->Wao, learningRate, m, this->gradHist->Wao, lstm.Wao);
  Optimizer::momentum(this->Wau, learningRate, m, this->gradHist->Wau, lstm.Wau);

  Optimizer::momentum(this->bi, learningRate, m, this->gradHist->bi, lstm.bi);
  Optimizer::momentum(this->bf, learningRate, m, this->gradHist->bf, lstm.bf);
  Optimizer::momentum(this->bo, learningRate, m, this->gradHist->bo, lstm.bo);
  Optimizer::momentum(this->bu, learningRate, m, this->gradHist->bu, lstm.bu);
}

void LSTM::Grad::operator += (const LSTM::Grad& grad){
  this->Wxi += grad.Wxi; this->Whi += grad.Whi; this->bi += grad.bi;
  this->Wxf += grad.Wxf; this->Whf += grad.Whf; this->bf += grad.bf;
  this->Wxo += grad.Wxo; this->Who += grad.Who; this->bo += grad.bo;
  this->Wxu += grad.Wxu; this->Whu += grad.Whu; this->bu += grad.bu;
  this->Wai += grad.Wai; this->Waf += grad.Waf; this->Wao += grad.Wao; this->Wau += grad.Wau;
}

//NOT USED!!
void LSTM::Grad::operator /= (const Real val){
  this->Wxi /= val; this->Whi /= val; this->bi /= val;
  this->Wxf /= val; this->Whf /= val; this->bf /= val;
  this->Wxo /= val; this->Who /= val; this->bo /= val;
  this->Wxu /= val; this->Whu /= val; this->bu /= val;
  this->Wai /= val; this->Waf /= val; this->Wao /= val; this->Wau /= val;
}
