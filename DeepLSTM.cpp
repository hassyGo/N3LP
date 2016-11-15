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

void DeepLSTM::forward(const VecD& xt, const DeepLSTM::State* prev, DeepLSTM::State* cur, int startDepth, int endDepth){
  if (startDepth == -1){
    startDepth = 1;
  }
  if (endDepth == -1){
    endDepth = this->lstms.size();
  }

  for (int i = startDepth-1; i < endDepth; ++i){
    if (i == 0){
      this->lstms[0].forward(xt, prev->lstm[0], cur->lstm[0]);
      continue;
    }

    this->lstms[i].forward(cur->lstm[i-1]->h, prev->lstm[i], cur->lstm[i]);
  }
}

void DeepLSTM::forward(const VecD& xt, const VecD& at, const DeepLSTM::State* prev, DeepLSTM::State* cur, int startDepth, int endDepth){
  if (startDepth == -1){
    startDepth = 1;
  }
  if (endDepth == -1){
    endDepth = this->lstms.size();
  }

  for (int i = startDepth-1; i < endDepth; ++i){
    if (i == 0){
      this->lstms[0].forward(xt, prev->lstm[0], cur->lstm[0]);
      continue;
    }

    this->lstms[i].forward(cur->lstm[i-1]->h, at, prev->lstm[i], cur->lstm[i]);
  }
}

void DeepLSTM::forward(const VecD& xt, DeepLSTM::State* cur, int startDepth, int endDepth){
  if (startDepth == -1){
    startDepth = 1;
  }
  if (endDepth == -1){
    endDepth = this->lstms.size();
  }
  
  for (int i = startDepth-1; i < endDepth; ++i){
    if (i == 0){
      this->lstms[0].forward(xt, cur->lstm[0]);
      continue;
    }

    this->lstms[i].forward(cur->lstm[i-1]->h, cur->lstm[i]);
  }
}

void DeepLSTM::forward(const VecD& xt, const VecD& at, DeepLSTM::State* cur, int startDepth, int endDepth){
  if (startDepth == -1){
    startDepth = 1;
  }
  if (endDepth == -1){
    endDepth = this->lstms.size();
  }

  for (int i = startDepth-1; i < endDepth; ++i){
    if (i == 0){
      this->lstms[0].forward(xt, cur->lstm[0]);
      continue;
    }

    this->lstms[i].forward(cur->lstm[i-1]->h, at, cur->lstm[i]);
  }
}

void DeepLSTM::backward(DeepLSTM::State* prev, DeepLSTM::State* cur, DeepLSTM::Grad& grad, const VecD& xt, int startDepth, int endDepth){
  if (startDepth == -1){
    startDepth = 1;
  }
  if (endDepth == -1){
    endDepth = this->lstms.size();
  }
  
  for (int i = endDepth-1; i >= startDepth-1; --i){
    if (i == 0){
      this->lstms[0].backward(prev->lstm[0], cur->lstm[0], grad.lstm[0], xt);
    }
    else {
      this->lstms[i].backward(prev->lstm[i], cur->lstm[i], grad.lstm[i], cur->lstm[i-1]->h);
      cur->lstm[i-1]->delh += cur->lstm[i]->delx;
    }
  }
}

void DeepLSTM::backward(DeepLSTM::State* prev, DeepLSTM::State* cur, DeepLSTM::Grad& grad, const VecD& xt, const VecD& at, int startDepth, int endDepth){
  if (startDepth == -1){
    startDepth = 1;
  }
  if (endDepth == -1){
    endDepth = this->lstms.size();
  }
  
  for (int i = endDepth-1; i >= startDepth-1; --i){
    if (i == 0){
      this->lstms[0].backward(prev->lstm[0], cur->lstm[0], grad.lstm[0], xt);
    }
    else {
      this->lstms[i].backward(prev->lstm[i], cur->lstm[i], grad.lstm[i], cur->lstm[i-1]->h, at);
      cur->lstm[i-1]->delh += cur->lstm[i]->delx;
    }
  }
}

void DeepLSTM::backward(DeepLSTM::State* cur, DeepLSTM::Grad& grad, const VecD& xt, int startDepth, int endDepth){
  if (startDepth == -1){
    startDepth = 1;
  }
  if (endDepth == -1){
    endDepth = this->lstms.size();
  }
  
  for (int i = endDepth-1; i >= startDepth-1; --i){
    if (i == 0){
      this->lstms[0].backward(cur->lstm[0], grad.lstm[0], xt);
    }
    else {
      this->lstms[i].backward(cur->lstm[i], grad.lstm[i], cur->lstm[i-1]->h);
      cur->lstm[i-1]->delh += cur->lstm[i]->delx;
    }
  }
}

void DeepLSTM::backward(DeepLSTM::State* cur, DeepLSTM::Grad& grad, const VecD& xt, const VecD& at, int startDepth, int endDepth){
  if (startDepth == -1){
    startDepth = 1;
  }
  if (endDepth == -1){
    endDepth = this->lstms.size();
  }
  
  for (int i = endDepth-1; i >= startDepth-1; --i){
    if (i == 0){
      this->lstms[0].backward(cur->lstm[0], grad.lstm[0], xt);
    }
    else {
      this->lstms[i].backward(cur->lstm[i], grad.lstm[i], cur->lstm[i-1]->h, at);
      cur->lstm[i-1]->delh += cur->lstm[i]->delx;
    }
  }
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

void DeepLSTM::operator += (const DeepLSTM& lstm){
  for (unsigned int i = 0; i < this->lstms.size(); ++i){
    this->lstms[i] += lstm.lstms[i];
  }
}

void DeepLSTM::operator /= (const Real val){
  for (auto it = this->lstms.begin(); it != this->lstms.end(); ++it){
    (*it) /= val;
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

void DeepLSTM::Grad::init(int depth){
  if (depth == -1){
    depth = this->lstm.size()-1;
  }

  for (int i = 0; i <= depth; ++i){
    this->lstm[i].init();
  }
}

Real DeepLSTM::Grad::norm(int depth){
  Real res = 0.0;

  if (depth == -1){
    depth = this->lstm.size()-1;
  }

  for (int i = 0; i <= depth; ++i){
    res += this->lstm[i].norm();
  }

  return res;
}

void DeepLSTM::Grad::sgd(const Real learningRate, const unsigned int depth, DeepLSTM& lstm){
  for (unsigned int i = 0; i <= depth; ++i){
    this->lstm[i].sgd(learningRate, lstm.lstms[i]);
  }
}

void DeepLSTM::Grad::adagrad(const Real learningRate, DeepLSTM& lstm, const Real initVal){
  for (unsigned int i = 0; i < lstm.lstms.size(); ++i){
    this->lstm[i].adagrad(learningRate, lstm.lstms[i], initVal);
  }
}

void DeepLSTM::Grad::momentum(const Real learningRate, const Real m, DeepLSTM& lstm){
  for (unsigned int i = 0; i < lstm.lstms.size(); ++i){
    this->lstm[i].momentum(learningRate, m, lstm.lstms[i]);
  }
}

void DeepLSTM::Grad::operator += (const DeepLSTM::Grad& grad){
  for (int i = 0; i < (int)grad.lstm.size(); ++i){
    this->lstm[i] += grad.lstm[i];
  }
}

void DeepLSTM::Grad::operator /= (const Real val){
  for (int i = 0; i < (int)this->lstm.size(); ++i){
    this->lstm[i] /= val;
  }
}
