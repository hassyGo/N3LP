#include "DeepLSTM.hpp"

DeepLSTM::DeepLSTM(const int inputDim, const int hiddenDim, const int depth){
  for (int i = 0; i < depth; ++i){
    if (i == 0){
      this->lstms.push_back(LSTM(inputDim, hiddenDim));
    }
    else {
      this->lstms.push_back(LSTM(hiddenDim, hiddenDim));
    }
  }
}

void DeepLSTM::init(Rand& rnd, const double scale){
  for (auto it = this->lstms.begin(); it != this->lstms.end(); ++it){
    it->init(rnd, scale);
  }
}

void DeepLSTM::forward(const MatD& xt, const DeepLSTM::State* prev, DeepLSTM::State* cur){
  this->lstms[0].forward(xt, prev->lstm[0], cur->lstm[0]);

  for (int i = 1; i < (int)this->lstms.size(); ++i){
    this->lstms[i].forward(cur->lstm[i-1]->h, prev->lstm[i], cur->lstm[i]);
  }
}

void DeepLSTM::backward(DeepLSTM::State* prev, DeepLSTM::State* cur, DeepLSTM::Grad& grad, const MatD& xt){
  for (int i = this->lstms.size()-1; i >= 1; --i){
    this->lstms[i].backward(prev->lstm[i], cur->lstm[i], grad.lstm[i], cur->lstm[i-1]->h);
    cur->lstm[i-1]->delh += cur->lstm[i]->delx;
  }

  this->lstms[0].backward(prev->lstm[0], cur->lstm[0], grad.lstm[0], xt);
}

void DeepLSTM::sgd(const DeepLSTM::Grad& grad, const double learningRate){
  for (int i = 0; i < (int)grad.lstm.size(); ++i){
    this->lstms[i].sgd(grad.lstm[i], learningRate);
  }
}

DeepLSTM::State::State(const DeepLSTM& dlstm){
  for (auto it = dlstm.lstms.begin(); it != dlstm.lstms.end(); ++it){
    this->lstm.push_back(new LSTM::State);
  }
}

void DeepLSTM::State::clear(){
  for (auto it = this->lstm.begin(); it != this->lstm.end(); ++it){
    (*it)->clear();
  }
}

DeepLSTM::Grad::Grad(const DeepLSTM& dlstm){
  for (auto it = dlstm.lstms.begin(); it != dlstm.lstms.end(); ++it){
    this->lstm.push_back(LSTM::Grad(*it));
  }
}

void DeepLSTM::Grad::init(){
  for (auto it = this->lstm.begin(); it != this->lstm.end(); ++it){
    it->init();
  }
}

double DeepLSTM::Grad::norm(){
  double res = 0.0;

  for (auto it = this->lstm.begin(); it != this->lstm.end(); ++it){
    res += it->norm();
  }

  return res;
}
