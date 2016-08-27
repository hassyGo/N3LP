#include "TreeLSTM.hpp"
#include "ActFunc.hpp"
#include "Utils.hpp"

TreeLSTM::TreeLSTM(const int inputDim, const int hiddenDim){
  this->Wxi = MatD(hiddenDim, inputDim);
  this->WhiL = MatD(hiddenDim, hiddenDim);
  this->WhiR = MatD(hiddenDim, hiddenDim);
  this->bi = VecD::Zero(hiddenDim);

  this->Wxfl = MatD(hiddenDim, inputDim);
  this->WhflL = MatD(hiddenDim, hiddenDim);
  this->WhflR = MatD(hiddenDim, hiddenDim);
  this->bfl = VecD::Zero(hiddenDim);

  this->Wxfr = MatD(hiddenDim, inputDim);
  this->WhfrL = MatD(hiddenDim, hiddenDim);
  this->WhfrR = MatD(hiddenDim, hiddenDim);
  this->bfr = VecD::Zero(hiddenDim);

  this->Wxo = MatD(hiddenDim, inputDim);
  this->WhoL = MatD(hiddenDim, hiddenDim);
  this->WhoR = MatD(hiddenDim, hiddenDim);
  this->bo = VecD::Zero(hiddenDim);

  this->Wxu = MatD(hiddenDim, inputDim);
  this->WhuL = MatD(hiddenDim, hiddenDim);
  this->WhuR = MatD(hiddenDim, hiddenDim);
  this->bu = VecD::Zero(hiddenDim);
}

void TreeLSTM::init(Rand& rnd, const Real scale){
  rnd.uniform(this->Wxi, scale);
  rnd.uniform(this->WhiL, scale);
  rnd.uniform(this->WhiR, scale);

  rnd.uniform(this->Wxfl, scale);
  rnd.uniform(this->WhflL, scale);
  rnd.uniform(this->WhflR, scale);

  rnd.uniform(this->Wxfr, scale);
  rnd.uniform(this->WhfrL, scale);
  rnd.uniform(this->WhfrR, scale);
  
  rnd.uniform(this->Wxo, scale);
  rnd.uniform(this->WhoL, scale);
  rnd.uniform(this->WhoR, scale);

  rnd.uniform(this->Wxu, scale);
  rnd.uniform(this->WhuL, scale);
  rnd.uniform(this->WhuR, scale);
}

void TreeLSTM::forward(const VecD& xt, TreeLSTM::State* parent, LSTM::State* left, LSTM::State* right){
  parent->i = this->bi;
  parent->i.noalias() += this->Wxi*xt + this->WhiL*left->h + this->WhiR*right->h;
  parent->fl = this->bfl;
  parent->fl.noalias() += this->Wxfl*xt + this->WhflL*left->h + this->WhflR*right->h;
  parent->fr = this->bfr;
  parent->fr.noalias() += this->Wxfr*xt + this->WhfrL*left->h + this->WhfrR*right->h;
  parent->o = this->bo;
  parent->o.noalias() += this->Wxo*xt + this->WhoL*left->h + this->WhoR*right->h;
  parent->u = this->bu;
  parent->u.noalias() += this->Wxu*xt + this->WhuL*left->h + this->WhuR*right->h;

  ActFunc::logistic(parent->i);
  ActFunc::logistic(parent->fl);
  ActFunc::logistic(parent->fr);
  ActFunc::logistic(parent->o);
  ActFunc::tanh(parent->u);
  parent->c = parent->i.array()*parent->u.array() + parent->fl.array()*left->c.array() + parent->fr.array()*right->c.array();
  parent->cTanh = parent->c;
  ActFunc::tanh(parent->cTanh);
  parent->h = parent->o.array()*parent->cTanh.array();
}

void TreeLSTM::forward(TreeLSTM::State* parent, LSTM::State* left, LSTM::State* right){
  parent->i = this->bi;
  parent->i.noalias() += this->WhiL*left->h + this->WhiR*right->h;
  parent->fl = this->bfl;
  parent->fl.noalias() += this->WhflL*left->h + this->WhflR*right->h;
  parent->fr = this->bfr;
  parent->fr.noalias() += this->WhfrL*left->h + this->WhfrR*right->h;
  parent->o = this->bo;
  parent->o.noalias() += this->WhoL*left->h + this->WhoR*right->h;
  parent->u = this->bu;
  parent->u.noalias() += this->WhuL*left->h + this->WhuR*right->h;

  ActFunc::logistic(parent->i);
  ActFunc::logistic(parent->fl);
  ActFunc::logistic(parent->fr);
  ActFunc::logistic(parent->o);
  ActFunc::tanh(parent->u);
  parent->c = parent->i.array()*parent->u.array() + parent->fl.array()*left->c.array() + parent->fr.array()*right->c.array();
  parent->cTanh = parent->c;
  ActFunc::tanh(parent->cTanh);
  parent->h = parent->o.array()*parent->cTanh.array();
}

void TreeLSTM::backward(TreeLSTM::State* parent, LSTM::State* left, LSTM::State* right, TreeLSTM::Grad& grad, const VecD& xt){
  VecD delo, deli, delu, delfl, delfr;

  parent->delc.array() += ActFunc::tanhPrime(parent->cTanh).array()*parent->delh.array()*parent->o.array();
  left->delc.array() += parent->delc.array()*parent->fl.array();
  right->delc.array() += parent->delc.array()*parent->fr.array();
  delo = ActFunc::logisticPrime(parent->o).array()*parent->delh.array()*parent->cTanh.array();
  deli = ActFunc::logisticPrime(parent->i).array()*parent->delc.array()*parent->u.array();
  delfl = ActFunc::logisticPrime(parent->fl).array()*parent->delc.array()*left->c.array();
  delfr = ActFunc::logisticPrime(parent->fr).array()*parent->delc.array()*right->c.array();
  delu = ActFunc::tanhPrime(parent->u).array()*parent->delc.array()*parent->i.array();

  parent->delx.noalias() =
    this->Wxi.transpose()*deli+
    this->Wxfl.transpose()*delfl+
    this->Wxfr.transpose()*delfr+
    this->Wxo.transpose()*delo+
    this->Wxu.transpose()*delu;

  left->delh.noalias() +=
    this->WhiL.transpose()*deli+
    this->WhflL.transpose()*delfl+
    this->WhfrL.transpose()*delfr+
    this->WhoL.transpose()*delo+
    this->WhuL.transpose()*delu;

  right->delh.noalias() +=
    this->WhiR.transpose()*deli+
    this->WhflR.transpose()*delfl+
    this->WhfrR.transpose()*delfr+
    this->WhoR.transpose()*delo+
    this->WhuR.transpose()*delu;

  grad.Wxi.noalias() += deli*xt.transpose();
  grad.WhiL.noalias() += deli*left->h.transpose();
  grad.WhiR.noalias() += deli*right->h.transpose();
  
  grad.Wxfl.noalias() += delfl*xt.transpose();
  grad.WhflL.noalias() += delfl*left->h.transpose();
  grad.WhflR.noalias() += delfl*right->h.transpose();

  grad.Wxfr.noalias() += delfr*xt.transpose();
  grad.WhfrL.noalias() += delfr*left->h.transpose();
  grad.WhfrR.noalias() += delfr*right->h.transpose();

  grad.Wxo.noalias() += delo*xt.transpose();
  grad.WhoL.noalias() += delo*left->h.transpose();
  grad.WhoR.noalias() += delo*right->h.transpose();

  grad.Wxu.noalias() += delu*xt.transpose();
  grad.WhuL.noalias() += delu*left->h.transpose();
  grad.WhuR.noalias() += delu*right->h.transpose();

  grad.bi += deli;
  grad.bfl += delfl;
  grad.bfr += delfr;
  grad.bo += delo;
  grad.bu += delu;
}

void TreeLSTM::backward(TreeLSTM::State* parent, LSTM::State* left, LSTM::State* right, TreeLSTM::Grad& grad){
  VecD delo, deli, delu, delfl, delfr;

  parent->delc.array() += ActFunc::tanhPrime(parent->cTanh).array()*parent->delh.array()*parent->o.array();
  left->delc.array() += parent->delc.array()*parent->fl.array();
  right->delc.array() += parent->delc.array()*parent->fr.array();
  delo = ActFunc::logisticPrime(parent->o).array()*parent->delh.array()*parent->cTanh.array();
  deli = ActFunc::logisticPrime(parent->i).array()*parent->delc.array()*parent->u.array();
  delfl = ActFunc::logisticPrime(parent->fl).array()*parent->delc.array()*left->c.array();
  delfr = ActFunc::logisticPrime(parent->fr).array()*parent->delc.array()*right->c.array();
  delu = ActFunc::tanhPrime(parent->u).array()*parent->delc.array()*parent->i.array();

  left->delh.noalias() +=
    this->WhiL.transpose()*deli+
    this->WhflL.transpose()*delfl+
    this->WhfrL.transpose()*delfr+
    this->WhoL.transpose()*delo+
    this->WhuL.transpose()*delu;

  right->delh.noalias() +=
    this->WhiR.transpose()*deli+
    this->WhflR.transpose()*delfl+
    this->WhfrR.transpose()*delfr+
    this->WhoR.transpose()*delo+
    this->WhuR.transpose()*delu;

  grad.WhiL.noalias() += deli*left->h.transpose();
  grad.WhiR.noalias() += deli*right->h.transpose();
  
  grad.WhflL.noalias() += delfl*left->h.transpose();
  grad.WhflR.noalias() += delfl*right->h.transpose();

  grad.WhfrL.noalias() += delfr*left->h.transpose();
  grad.WhfrR.noalias() += delfr*right->h.transpose();

  grad.WhoL.noalias() += delo*left->h.transpose();
  grad.WhoR.noalias() += delo*right->h.transpose();

  grad.WhuL.noalias() += delu*left->h.transpose();
  grad.WhuR.noalias() += delu*right->h.transpose();

  grad.bi += deli;
  grad.bfl += delfl;
  grad.bfr += delfr;
  grad.bo += delo;
  grad.bu += delu;
}

void TreeLSTM::sgd(const TreeLSTM::Grad& grad, const Real learningRate){
  this->Wxi -= learningRate*grad.Wxi;
  this->WhiL -= learningRate*grad.WhiL;
  this->WhiR -= learningRate*grad.WhiR;
  this->bi -= learningRate*grad.bi;

  this->Wxfl -= learningRate*grad.Wxfl;
  this->WhflL -= learningRate*grad.WhflL;
  this->WhflR -= learningRate*grad.WhflR;
  this->bfl -= learningRate*grad.bfl;

  this->Wxfr -= learningRate*grad.Wxfr;
  this->WhfrL -= learningRate*grad.WhfrL;
  this->WhfrR -= learningRate*grad.WhfrR;
  this->bfr -= learningRate*grad.bfr;

  this->Wxo -= learningRate*grad.Wxo;
  this->WhoL -= learningRate*grad.WhoL;
  this->WhoR -= learningRate*grad.WhoR;
  this->bo -= learningRate*grad.bo;

  this->Wxu -= learningRate*grad.Wxu;
  this->WhuL -= learningRate*grad.WhuL;
  this->WhuR -= learningRate*grad.WhuR;
  this->bu -= learningRate*grad.bu;
}

void TreeLSTM::save(std::ofstream& ofs){
  Utils::save(ofs, this->Wxi); Utils::save(ofs, this->WhiL); Utils::save(ofs, this->WhiR); Utils::save(ofs, this->bi);
  Utils::save(ofs, this->Wxfl); Utils::save(ofs, this->WhflL); Utils::save(ofs, this->WhflR); Utils::save(ofs, this->bfl);
  Utils::save(ofs, this->Wxfr); Utils::save(ofs, this->WhfrL); Utils::save(ofs, this->WhfrR); Utils::save(ofs, this->bfr);
  Utils::save(ofs, this->Wxo); Utils::save(ofs, this->WhoL); Utils::save(ofs, this->WhoR); Utils::save(ofs, this->bo);
  Utils::save(ofs, this->Wxu); Utils::save(ofs, this->WhuL); Utils::save(ofs, this->WhuR); Utils::save(ofs, this->bu);
}

void TreeLSTM::load(std::ifstream& ifs){
  Utils::load(ifs, this->Wxi); Utils::load(ifs, this->WhiL); Utils::load(ifs, this->WhiR); Utils::load(ifs, this->bi);
  Utils::load(ifs, this->Wxfl); Utils::load(ifs, this->WhflL); Utils::load(ifs, this->WhflR); Utils::load(ifs, this->bfl);
  Utils::load(ifs, this->Wxfr); Utils::load(ifs, this->WhfrL); Utils::load(ifs, this->WhfrR); Utils::load(ifs, this->bfr);
  Utils::load(ifs, this->Wxo); Utils::load(ifs, this->WhoL); Utils::load(ifs, this->WhoR); Utils::load(ifs, this->bo);
  Utils::load(ifs, this->Wxu); Utils::load(ifs, this->WhuL); Utils::load(ifs, this->WhuR); Utils::load(ifs, this->bu);
}

void TreeLSTM::State::clear(){
  LSTM::State::clear();
  this->fl = VecD();
  this->fr = VecD();
}

TreeLSTM::Grad::Grad(const TreeLSTM& tlstm){
  this->Wxi = MatD::Zero(tlstm.Wxi.rows(), tlstm.Wxi.cols());
  this->WhiL = MatD::Zero(tlstm.WhiL.rows(), tlstm.WhiL.cols());
  this->WhiR = MatD::Zero(tlstm.WhiR.rows(), tlstm.WhiR.cols());
  this->bi = MatD::Zero(tlstm.bi.rows(), tlstm.bi.cols());

  this->Wxfl = MatD::Zero(tlstm.Wxfl.rows(), tlstm.Wxfl.cols());
  this->WhflL = MatD::Zero(tlstm.WhflL.rows(), tlstm.WhflL.cols());
  this->WhflR = MatD::Zero(tlstm.WhflR.rows(), tlstm.WhflR.cols());
  this->bfl = MatD::Zero(tlstm.bfl.rows(), tlstm.bfl.cols());

  this->Wxfr = MatD::Zero(tlstm.Wxfr.rows(), tlstm.Wxfr.cols());
  this->WhfrL = MatD::Zero(tlstm.WhfrL.rows(), tlstm.WhfrL.cols());
  this->WhfrR = MatD::Zero(tlstm.WhfrR.rows(), tlstm.WhfrR.cols());
  this->bfr = MatD::Zero(tlstm.bfr.rows(), tlstm.bfr.cols());

  this->Wxo = MatD::Zero(tlstm.Wxo.rows(), tlstm.Wxo.cols());
  this->WhoL = MatD::Zero(tlstm.WhoL.rows(), tlstm.WhoL.cols());
  this->WhoR = MatD::Zero(tlstm.WhoR.rows(), tlstm.WhoR.cols());
  this->bo = MatD::Zero(tlstm.bo.rows(), tlstm.bo.cols());

  this->Wxu = MatD::Zero(tlstm.Wxu.rows(), tlstm.Wxu.cols());
  this->WhuL = MatD::Zero(tlstm.WhuL.rows(), tlstm.WhuL.cols());
  this->WhuR = MatD::Zero(tlstm.WhuR.rows(), tlstm.WhuR.cols());
  this->bu = MatD::Zero(tlstm.bu.rows(), tlstm.bu.cols());
}

void TreeLSTM::Grad::init(){
  this->Wxi.setZero(); this->WhiL.setZero(); this->WhiR.setZero(); this->bi.setZero();
  this->Wxfl.setZero(); this->WhflL.setZero(); this->WhflR.setZero(); this->bfl.setZero();
  this->Wxfr.setZero(); this->WhfrL.setZero(); this->WhfrR.setZero(); this->bfr.setZero();
  this->Wxo.setZero(); this->WhoL.setZero(); this->WhoR.setZero(); this->bo.setZero();
  this->Wxu.setZero(); this->WhuL.setZero(); this->WhuR.setZero(); this->bu.setZero();
}

Real TreeLSTM::Grad::norm(){
  return
    this->Wxi.squaredNorm()+this->WhiL.squaredNorm()+this->WhiR.squaredNorm()+this->bi.squaredNorm()+
    this->Wxfl.squaredNorm()+this->WhflL.squaredNorm()+this->WhflR.squaredNorm()+this->bfl.squaredNorm()+
    this->Wxfr.squaredNorm()+this->WhfrL.squaredNorm()+this->WhfrR.squaredNorm()+this->bfr.squaredNorm()+
    this->Wxo.squaredNorm()+this->WhoL.squaredNorm()+this->WhoR.squaredNorm()+this->bo.squaredNorm()+
    this->Wxu.squaredNorm()+this->WhuL.squaredNorm()+this->WhuR.squaredNorm()+this->bu.squaredNorm();
}

void TreeLSTM::Grad::operator += (const TreeLSTM::Grad& grad){
  this->Wxi += grad.Wxi; this->WhiL += grad.WhiL; this->WhiR += grad.WhiR; this->bi += grad.bi;
  this->Wxfl += grad.Wxfl; this->WhflL += grad.WhflL; this->WhflR += grad.WhflR; this->bfl += grad.bfl;
  this->Wxfr += grad.Wxfr; this->WhfrL += grad.WhfrL; this->WhfrR += grad.WhfrR; this->bfr += grad.bfr;
  this->Wxo += grad.Wxo; this->WhoL += grad.WhoL; this->WhoR += grad.WhoR; this->bo += grad.bo;
  this->Wxu += grad.Wxu; this->WhuL += grad.WhuL; this->WhuR += grad.WhuR; this->bu += grad.bu;
}
