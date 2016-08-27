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

DeepLSTM::DeepLSTM(const int inputDim, const int additionalInputDim, const int hiddenDim, const int depth){
  for (int i = 0; i < depth; ++i){
    if (i == 0){
      this->lstms.push_back(LSTM(inputDim, additionalInputDim, hiddenDim));
    }
    else {
      this->lstms.push_back(LSTM(hiddenDim, additionalInputDim, hiddenDim));
    }
  }
}

void DeepLSTM::init(Rand& rnd, const Real scale){
  for (auto it = this->lstms.begin(); it != this->lstms.end(); ++it){
    it->init(rnd, scale);
  }
}

void DeepLSTM::forward(const VecD& xt, const DeepLSTM::State* prev, DeepLSTM::State* cur, int depth){
  if (depth == -1){
    depth = this->lstms.size();
  }
  
  this->lstms[0].forward(xt, prev->lstm[0], cur->lstm[0]);

  for (int i = 1; i < depth; ++i){
    this->lstms[i].forward(cur->lstm[i-1]->h, prev->lstm[i], cur->lstm[i]);
  }
}

void DeepLSTM::forward(const VecD& xt, const VecD& at, const DeepLSTM::State* prev, DeepLSTM::State* cur, int depth){
  if (depth == -1){
    depth = this->lstms.size();
  }
  
  this->lstms[0].forward(xt, prev->lstm[0], cur->lstm[0]);

  for (int i = 1; i < depth; ++i){
    this->lstms[i].forward(cur->lstm[i-1]->h, at, prev->lstm[i], cur->lstm[i]);
  }
}

void DeepLSTM::forward(const VecD& xt, DeepLSTM::State* cur, int depth){
  if (depth == -1){
    depth = this->lstms.size();
  }
    
  this->lstms[0].forward(xt, cur->lstm[0]);

  for (int i = 1; i < depth; ++i){
    this->lstms[i].forward(cur->lstm[i-1]->h, cur->lstm[i]);
  }
}

void DeepLSTM::forward(const VecD& xt, const VecD& at, DeepLSTM::State* cur, int depth){
  if (depth == -1){
    depth = this->lstms.size();
  }
    
  this->lstms[0].forward(xt, cur->lstm[0]);

  for (int i = 1; i < depth; ++i){
    this->lstms[i].forward(cur->lstm[i-1]->h, at, cur->lstm[i]);
  }
}

void DeepLSTM::backward(DeepLSTM::State* prev, DeepLSTM::State* cur, DeepLSTM::Grad& grad, const VecD& xt, int depth){
  if (depth == -1){
    depth = this->lstms.size();
  }
  
  for (int i = depth-1; i >= 1; --i){
    this->lstms[i].backward(prev->lstm[i], cur->lstm[i], grad.lstm[i], cur->lstm[i-1]->h);
    cur->lstm[i-1]->delh += cur->lstm[i]->delx;
  }

  this->lstms[0].backward(prev->lstm[0], cur->lstm[0], grad.lstm[0], xt);
}

void DeepLSTM::backward(DeepLSTM::State* prev, DeepLSTM::State* cur, DeepLSTM::Grad& grad, const VecD& xt, const VecD& at, int depth){
  if (depth == -1){
    depth = this->lstms.size();
  }
  
  for (int i = depth-1; i >= 1; --i){
    this->lstms[i].backward(prev->lstm[i], cur->lstm[i], grad.lstm[i], cur->lstm[i-1]->h, at);
    cur->lstm[i-1]->delh += cur->lstm[i]->delx;
  }

  this->lstms[0].backward(prev->lstm[0], cur->lstm[0], grad.lstm[0], xt);
}

void DeepLSTM::backward(DeepLSTM::State* cur, DeepLSTM::Grad& grad, const VecD& xt, int depth){
  if (depth == -1){
    depth = this->lstms.size();
  }
  
  for (int i = depth-1; i >= 1; --i){
    this->lstms[i].backward(cur->lstm[i], grad.lstm[i], cur->lstm[i-1]->h);
    cur->lstm[i-1]->delh += cur->lstm[i]->delx;
  }

  this->lstms[0].backward(cur->lstm[0], grad.lstm[0], xt);
}

void DeepLSTM::backward(DeepLSTM::State* cur, DeepLSTM::Grad& grad, const VecD& xt, const VecD& at, int depth){
  if (depth == -1){
    depth = this->lstms.size();
  }
  
  for (int i = depth-1; i >= 1; --i){
    this->lstms[i].backward(cur->lstm[i], grad.lstm[i], cur->lstm[i-1]->h, at);
    cur->lstm[i-1]->delh += cur->lstm[i]->delx;
  }

  this->lstms[0].backward(cur->lstm[0], grad.lstm[0], xt);
}

void DeepLSTM::sgd(const DeepLSTM::Grad& grad, const Real learningRate){
  for (int i = 0; i < (int)grad.lstm.size(); ++i){
    this->lstms[i].sgd(grad.lstm[i], learningRate);
  }
}

void DeepLSTM::save(std::ofstream& ofs){
  for (int i = 0; i < (int)this->lstms.size(); ++i){
    this->lstms[i].save(ofs);
  }
}

void DeepLSTM::load(std::ifstream& ifs){
  for (int i = 0; i < (int)this->lstms.size(); ++i){
    this->lstms[i].load(ifs);
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

Real DeepLSTM::Grad::norm(){
  Real res = 0.0;

  for (auto it = this->lstm.begin(); it != this->lstm.end(); ++it){
    res += it->norm();
  }

  return res;
}

void DeepLSTM::Grad::operator += (const DeepLSTM::Grad& grad){
  for (int i = 0; i < (int)grad.lstm.size(); ++i){
    this->lstm[i] += grad.lstm[i];
  }
}
