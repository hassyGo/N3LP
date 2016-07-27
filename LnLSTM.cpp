#include "LnLSTM.hpp"
#include "ActFunc.hpp"

LnLSTM::LnLSTM(const int inputDim, const int hiddenDim):
  LSTM(inputDim, hiddenDim)
{
  this->lnh = LayerNormalizer(4*hiddenDim);
  this->lnx = LayerNormalizer(4*hiddenDim);
  this->lnc = LayerNormalizer(hiddenDim);
}

LnLSTM::LnLSTM(const int inputDim, const int additionalInputDim, const int hiddenDim):
  LSTM(inputDim, additionalInputDim, hiddenDim)
{
  this->lnh = LayerNormalizer(4*hiddenDim);
  this->lnx = LayerNormalizer(4*hiddenDim);
  this->lnc = LayerNormalizer(hiddenDim);
  this->lna = LayerNormalizer(4*hiddenDim);
}

void LnLSTM::init(Rand& rnd, const Real scale){
  LSTM::init(rnd, scale);
  this->lnh.init();
  this->lnx.init();
  this->lnc.init();
  this->lna.init();
}

void LnLSTM::forward(const VecD& xt, const LSTM::State* prev, LSTM::State* cur){
  const unsigned int H = this->bi.rows();
  LnLSTM::State* state = (LnLSTM::State*)cur;

  state->lnhConcat = VecD(4*H);
  state->lnxConcat = VecD(4*H);

  state->lnhConcat.segment(0*H, H).noalias() = this->Whi*prev->h;
  state->lnhConcat.segment(1*H, H).noalias() = this->Whf*prev->h;
  state->lnhConcat.segment(2*H, H).noalias() = this->Who*prev->h;
  state->lnhConcat.segment(3*H, H).noalias() = this->Whu*prev->h;
  this->lnh.forward(state->lnhConcat, state->lnsh);

  state->lnxConcat.segment(0*H, H).noalias() = this->Wxi*xt;
  state->lnxConcat.segment(1*H, H).noalias() = this->Wxf*xt;
  state->lnxConcat.segment(2*H, H).noalias() = this->Wxo*xt;
  state->lnxConcat.segment(3*H, H).noalias() = this->Wxu*xt;
  this->lnx.forward(state->lnxConcat, state->lnsx);

  cur->i = this->bi+state->lnhConcat.segment(0*H, H)+state->lnxConcat.segment(0*H, H);
  cur->f = this->bf+state->lnhConcat.segment(1*H, H)+state->lnxConcat.segment(1*H, H);
  cur->o = this->bo+state->lnhConcat.segment(2*H, H)+state->lnxConcat.segment(2*H, H);
  cur->u = this->bu+state->lnhConcat.segment(3*H, H)+state->lnxConcat.segment(3*H, H);

  ActFunc::logistic(cur->i);
  ActFunc::logistic(cur->f);
  ActFunc::logistic(cur->o);
  ActFunc::tanh(cur->u);

  cur->c = cur->i.array()*cur->u.array() + cur->f.array()*prev->c.array();
  cur->cTanh = cur->c;
  this->lnc.forward(cur->cTanh, state->lnsc);
  ActFunc::tanh(cur->cTanh);
  cur->h = cur->o.array()*cur->cTanh.array();
}

void LnLSTM::forward(const VecD& xt, LSTM::State* cur){
  const unsigned int H = this->bi.rows();
  LnLSTM::State* state = (LnLSTM::State*)cur;

  state->lnxConcat = VecD(4*H);

  state->lnxConcat.segment(0*H, H).noalias() = this->Wxi*xt;
  state->lnxConcat.segment(1*H, H).noalias() = this->Wxf*xt;
  state->lnxConcat.segment(2*H, H).noalias() = this->Wxo*xt;
  state->lnxConcat.segment(3*H, H).noalias() = this->Wxu*xt;
  this->lnx.forward(state->lnxConcat, state->lnsx);

  cur->i = this->bi+state->lnxConcat.segment(0*H, H);
  cur->f = this->bf+state->lnxConcat.segment(1*H, H);
  cur->o = this->bo+state->lnxConcat.segment(2*H, H);
  cur->u = this->bu+state->lnxConcat.segment(3*H, H);

  ActFunc::logistic(cur->i);
  ActFunc::logistic(cur->f);
  ActFunc::logistic(cur->o);
  ActFunc::tanh(cur->u);

  cur->c = cur->i.array()*cur->u.array();
  cur->cTanh = cur->c;
  this->lnc.forward(cur->cTanh, state->lnsc);
  ActFunc::tanh(cur->cTanh);
  cur->h = cur->o.array()*cur->cTanh.array();
}

void LnLSTM::backward(LSTM::State* prev, LSTM::State* cur, LSTM::Grad& grad, const VecD& xt){
  const unsigned int H = this->bi.rows();
  LnLSTM::State* state = (LnLSTM::State*)cur;
  LnLSTM::Grad& gg = (LnLSTM::Grad&)grad;
  VecD delc;
  VecD delhConcat, delxConcat;

  this->lnc.backward(ActFunc::tanhPrime(cur->cTanh).array()*cur->delh.array()*cur->o.array(), delc, state->lnsc, gg.lnc);
  cur->delc += delc;
  prev->delc.array() += cur->delc.array()*cur->f.array();

  state->delConcat = VecD(4*H);
  state->delConcat.segment(0*H, H) = ActFunc::logisticPrime(cur->i).array()*cur->delc.array()*cur->u.array();
  state->delConcat.segment(1*H, H) = ActFunc::logisticPrime(cur->f).array()*cur->delc.array()*prev->c.array();
  state->delConcat.segment(2*H, H) = ActFunc::logisticPrime(cur->o).array()*cur->delh.array()*cur->cTanh.array();
  state->delConcat.segment(3*H, H) = ActFunc::tanhPrime(cur->u).array()*cur->delc.array()*cur->i.array();
  this->lnh.backward(state->delConcat, delhConcat, state->lnsh, gg.lnh);
  this->lnx.backward(state->delConcat, delxConcat, state->lnsx, gg.lnx);

  cur->delx.noalias() =
    this->Wxi.transpose()*delxConcat.segment(H*0, H)+
    this->Wxf.transpose()*delxConcat.segment(H*1, H)+
    this->Wxo.transpose()*delxConcat.segment(H*2, H)+
    this->Wxu.transpose()*delxConcat.segment(H*3, H);

  prev->delh.noalias() +=
    this->Whi.transpose()*delhConcat.segment(H*0, H)+
    this->Whf.transpose()*delhConcat.segment(H*1, H)+
    this->Who.transpose()*delhConcat.segment(H*2, H)+
    this->Whu.transpose()*delhConcat.segment(H*3, H);
  
  grad.Wxi.noalias() += delxConcat.segment(H*0, H)*xt.transpose();
  grad.Whi.noalias() += delhConcat.segment(H*0, H)*prev->h.transpose();

  grad.Wxf.noalias() += delxConcat.segment(H*1, H)*xt.transpose();
  grad.Whf.noalias() += delhConcat.segment(H*1, H)*prev->h.transpose();

  grad.Wxo.noalias() += delxConcat.segment(H*2, H)*xt.transpose();
  grad.Who.noalias() += delhConcat.segment(H*2, H)*prev->h.transpose();

  grad.Wxu.noalias() += delxConcat.segment(H*3, H)*xt.transpose();
  grad.Whu.noalias() += delhConcat.segment(H*3, H)*prev->h.transpose();

  grad.bi += state->delConcat.segment(H*0, H);
  grad.bf += state->delConcat.segment(H*1, H);
  grad.bo += state->delConcat.segment(H*2, H);
  grad.bu += state->delConcat.segment(H*3, H);
}

void LnLSTM::backward(LSTM::State* cur, LSTM::Grad& grad, const VecD& xt){
  const unsigned int H = this->bi.rows();
  LnLSTM::State* state = (LnLSTM::State*)cur;
  LnLSTM::Grad& gg= (LnLSTM::Grad&)grad;
  VecD delc;
  VecD delhConcat, delxConcat;

  this->lnc.backward(ActFunc::tanhPrime(cur->cTanh).array()*cur->delh.array()*cur->o.array(), delc, state->lnsc, gg.lnc);
  cur->delc += delc;

  state->delConcat = VecD(4*H);
  state->delConcat.segment(0*H, H) = ActFunc::logisticPrime(cur->i).array()*cur->delc.array()*cur->u.array();
  state->delConcat.segment(1*H, H).setZero();
  state->delConcat.segment(2*H, H) = ActFunc::logisticPrime(cur->o).array()*cur->delh.array()*cur->cTanh.array();
  state->delConcat.segment(3*H, H) = ActFunc::tanhPrime(cur->u).array()*cur->delc.array()*cur->i.array();
  this->lnx.backward(state->delConcat, delxConcat, state->lnsx, gg.lnx);
  
  cur->delx.noalias() =
    this->Wxi.transpose()*delxConcat.segment(H*0, H)+
    this->Wxf.transpose()*delxConcat.segment(H*1, H)+
    this->Wxo.transpose()*delxConcat.segment(H*2, H)+
    this->Wxu.transpose()*delxConcat.segment(H*3, H);
  
  grad.Wxi.noalias() += delxConcat.segment(H*0, H)*xt.transpose();
  grad.Wxf.noalias() += delxConcat.segment(H*1, H)*xt.transpose();
  grad.Wxo.noalias() += delxConcat.segment(H*2, H)*xt.transpose();
  grad.Wxu.noalias() += delxConcat.segment(H*3, H)*xt.transpose();

  grad.bi += state->delConcat.segment(H*0, H);
  grad.bf += state->delConcat.segment(H*1, H);
  grad.bo += state->delConcat.segment(H*2, H);
  grad.bu += state->delConcat.segment(H*3, H);
}

void LnLSTM::forward(const VecD& xt, const VecD& at, const LSTM::State* prev, LSTM::State* cur){
  const unsigned int H = this->bi.rows();
  LnLSTM::State* state = (LnLSTM::State*)cur;

  state->lnhConcat = VecD(4*H);
  state->lnxConcat = VecD(4*H);
  state->lnaConcat = VecD(4*H);

  state->lnhConcat.segment(0*H, H).noalias() = this->Whi*prev->h;
  state->lnhConcat.segment(1*H, H).noalias() = this->Whf*prev->h;
  state->lnhConcat.segment(2*H, H).noalias() = this->Who*prev->h;
  state->lnhConcat.segment(3*H, H).noalias() = this->Whu*prev->h;
  this->lnh.forward(state->lnhConcat, state->lnsh);

  state->lnxConcat.segment(0*H, H).noalias() = this->Wxi*xt;
  state->lnxConcat.segment(1*H, H).noalias() = this->Wxf*xt;
  state->lnxConcat.segment(2*H, H).noalias() = this->Wxo*xt;
  state->lnxConcat.segment(3*H, H).noalias() = this->Wxu*xt;
  this->lnx.forward(state->lnxConcat, state->lnsx);

  state->lnaConcat.segment(0*H, H).noalias() = this->Wai*at;
  state->lnaConcat.segment(1*H, H).noalias() = this->Waf*at;
  state->lnaConcat.segment(2*H, H).noalias() = this->Wao*at;
  state->lnaConcat.segment(3*H, H).noalias() = this->Wau*at;
  this->lna.forward(state->lnaConcat, state->lnsa);

  cur->i = this->bi+state->lnhConcat.segment(0*H, H)+state->lnxConcat.segment(0*H, H)+state->lnaConcat.segment(0*H, H);
  cur->f = this->bf+state->lnhConcat.segment(1*H, H)+state->lnxConcat.segment(1*H, H)+state->lnaConcat.segment(1*H, H);
  cur->o = this->bo+state->lnhConcat.segment(2*H, H)+state->lnxConcat.segment(2*H, H)+state->lnaConcat.segment(2*H, H);
  cur->u = this->bu+state->lnhConcat.segment(3*H, H)+state->lnxConcat.segment(3*H, H)+state->lnaConcat.segment(3*H, H);

  ActFunc::logistic(cur->i);
  ActFunc::logistic(cur->f);
  ActFunc::logistic(cur->o);
  ActFunc::tanh(cur->u);

  cur->c = cur->i.array()*cur->u.array() + cur->f.array()*prev->c.array();
  cur->cTanh = cur->c;
  this->lnc.forward(cur->cTanh, state->lnsc);
  ActFunc::tanh(cur->cTanh);
  cur->h = cur->o.array()*cur->cTanh.array();
}

void LnLSTM::backward(LSTM::State* prev, LSTM::State* cur, LSTM::Grad& grad, const VecD& xt, const VecD& at){
  const unsigned int H = this->bi.rows();
  LnLSTM::State* state = (LnLSTM::State*)cur;
  LnLSTM::Grad& gg = (LnLSTM::Grad&)grad;
  VecD delc;
  VecD delhConcat, delxConcat, delaConcat;

  this->lnc.backward(ActFunc::tanhPrime(cur->cTanh).array()*cur->delh.array()*cur->o.array(), delc, state->lnsc, gg.lnc);
  cur->delc += delc;
  prev->delc.array() += cur->delc.array()*cur->f.array();

  state->delConcat = VecD(4*H);
  state->delConcat.segment(0*H, H) = ActFunc::logisticPrime(cur->i).array()*cur->delc.array()*cur->u.array();
  state->delConcat.segment(1*H, H) = ActFunc::logisticPrime(cur->f).array()*cur->delc.array()*prev->c.array();
  state->delConcat.segment(2*H, H) = ActFunc::logisticPrime(cur->o).array()*cur->delh.array()*cur->cTanh.array();
  state->delConcat.segment(3*H, H) = ActFunc::tanhPrime(cur->u).array()*cur->delc.array()*cur->i.array();
  this->lnh.backward(state->delConcat, delhConcat, state->lnsh, gg.lnh);
  this->lnx.backward(state->delConcat, delxConcat, state->lnsx, gg.lnx);
  this->lna.backward(state->delConcat, delaConcat, state->lnsa, gg.lna);

  cur->delx.noalias() =
    this->Wxi.transpose()*delxConcat.segment(H*0, H)+
    this->Wxf.transpose()*delxConcat.segment(H*1, H)+
    this->Wxo.transpose()*delxConcat.segment(H*2, H)+
    this->Wxu.transpose()*delxConcat.segment(H*3, H);

  prev->delh.noalias() +=
    this->Whi.transpose()*delhConcat.segment(H*0, H)+
    this->Whf.transpose()*delhConcat.segment(H*1, H)+
    this->Who.transpose()*delhConcat.segment(H*2, H)+
    this->Whu.transpose()*delhConcat.segment(H*3, H);

  cur->dela.noalias() =
    this->Wai.transpose()*delaConcat.segment(H*0, H)+
    this->Waf.transpose()*delaConcat.segment(H*1, H)+
    this->Wao.transpose()*delaConcat.segment(H*2, H)+
    this->Wau.transpose()*delaConcat.segment(H*3, H);
  
  grad.Wxi.noalias() += delxConcat.segment(H*0, H)*xt.transpose();
  grad.Whi.noalias() += delhConcat.segment(H*0, H)*prev->h.transpose();

  grad.Wxf.noalias() += delxConcat.segment(H*1, H)*xt.transpose();
  grad.Whf.noalias() += delhConcat.segment(H*1, H)*prev->h.transpose();

  grad.Wxo.noalias() += delxConcat.segment(H*2, H)*xt.transpose();
  grad.Who.noalias() += delhConcat.segment(H*2, H)*prev->h.transpose();

  grad.Wxu.noalias() += delxConcat.segment(H*3, H)*xt.transpose();
  grad.Whu.noalias() += delhConcat.segment(H*3, H)*prev->h.transpose();

  grad.Wai.noalias() += delaConcat.segment(H*0, H)*at.transpose();
  grad.Waf.noalias() += delaConcat.segment(H*1, H)*at.transpose();
  grad.Wao.noalias() += delaConcat.segment(H*2, H)*at.transpose();
  grad.Wau.noalias() += delaConcat.segment(H*3, H)*at.transpose();

  grad.bi += state->delConcat.segment(H*0, H);
  grad.bf += state->delConcat.segment(H*1, H);
  grad.bo += state->delConcat.segment(H*2, H);
  grad.bu += state->delConcat.segment(H*3, H);
}

void LnLSTM::sgd(const LnLSTM::Grad& grad, const Real learningRate){
  LSTM::sgd(grad, learningRate);
  this->lnh.sgd(grad.lnh, learningRate);
  this->lnx.sgd(grad.lnx, learningRate);
  this->lnc.sgd(grad.lnc, learningRate);
  this->lna.sgd(grad.lna, learningRate);
}

void LnLSTM::save(std::ofstream& ofs){
  LSTM::save(ofs);
  this->lnh.save(ofs);
  this->lnx.save(ofs);
  this->lnc.save(ofs);
  this->lna.save(ofs);
}

void LnLSTM::load(std::ifstream& ifs){
  LSTM::load(ifs);
  this->lnh.load(ifs);
  this->lnx.load(ifs);
  this->lnc.load(ifs);
  this->lna.load(ifs);
}

void LnLSTM::State::clear(){
  LSTM::State::clear();

  this->lnsh->clear();
  this->lnsx->clear();
  this->lnsc->clear();
  this->lnsa->clear();
  this->lnhConcat = VecD();
  this->lnxConcat = VecD();
  this->lnaConcat = VecD();
  this->delConcat = VecD();
}

LnLSTM::Grad::Grad(const LnLSTM& lnlstm):
  LSTM::Grad(lnlstm)
{
  this->lnh = LayerNormalizer::Grad(lnlstm.lnh);
  this->lnx = LayerNormalizer::Grad(lnlstm.lnx);
  this->lnc = LayerNormalizer::Grad(lnlstm.lnc);
  this->lna = LayerNormalizer::Grad(lnlstm.lna);
}

void LnLSTM::Grad::init(){
  LSTM::Grad::init();
  this->lnh.init();
  this->lnx.init();
  this->lnc.init();
  this->lna.init();
}

Real LnLSTM::Grad::norm(){
  return LSTM::Grad::norm()+this->lnh.norm()+this->lnx.norm()+this->lnc.norm()+this->lna.norm();
}

void LnLSTM::Grad::operator += (const LnLSTM::Grad& grad){
  (LSTM::Grad)(*this) += (LSTM::Grad)grad; //???
  this->lnh += grad.lnh;
  this->lnx += grad.lnx;
  this->lnc += grad.lnc;
  this->lna += grad.lna;
}
