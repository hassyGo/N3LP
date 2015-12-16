#include "EncDec.hpp"
#include "Utils.hpp"
#include <iostream>
#include <fstream>

EncDec::EncDec(Vocabulary& sourceVoc_, Vocabulary& targetVoc_, std::vector<EncDec::Data*>& trainData_, const int inputDim, const int hiddenDim):
  sourceVoc(sourceVoc_), targetVoc(targetVoc_), trainData(trainData_)
{
  double scale = 0.1;

  this->enc = LSTM(inputDim, hiddenDim);
  this->enc.init(this->rnd, scale);
  this->dec = LSTM(inputDim, hiddenDim);
  this->dec.init(this->rnd, scale);
  this->sourceEmbed = MatD(inputDim, this->sourceVoc.tokenList.size());
  this->targetEmbed = MatD(inputDim, this->targetVoc.tokenList.size());
  this->rnd.uniform(this->sourceEmbed);
  this->rnd.uniform(this->targetEmbed);
  this->softmax = SoftMax(hiddenDim, this->targetVoc.tokenList.size());
  this->zeros = MatD::Zero(hiddenDim, 1);
}

void EncDec::encode(const std::vector<int>& src, std::vector<LSTM::State*>& encState){
  encState[0]->h = this->zeros;
  encState[0]->c = this->zeros;

  for (int i = 0; i < (int)src.size(); ++i){
    this->enc.forward(this->sourceEmbed.col(src[i]), encState[i], encState[i+1]);
  }
}

void EncDec::translate(const std::vector<int>& src){
  MatD targetDist;
  std::vector<LSTM::State*> encState, decState;
  std::vector<int> tgt;

  for (int i = 0; i <= (int)src.size(); ++i){
    encState.push_back(new LSTM::State);
  }

  this->encode(src, encState);

  for (int i = 0, row, col; i < 100; ++i){
    decState.push_back(new LSTM::State);

    if (i == 0){
      decState[i]->h = encState.back()->h;
      decState[i]->c = encState.back()->c;
    }
    else {
      this->dec.forward(this->targetEmbed.col(tgt[i-1]), decState[i-1], decState[i]);
    }

    this->softmax.calcDist(decState[i]->h, targetDist);
    targetDist.maxCoeff(&row, &col);
    tgt.push_back(row);

    if (row == this->targetVoc.eosIndex){
      break;
    }
  }

  for (auto it = src.begin(); it != src.end(); ++it){
    std::cout << this->sourceVoc.tokenList[*it]->str << " ";
  }
  std::cout << std::endl;

  for (auto it = tgt.begin(); it != tgt.end(); ++it){
    std::cout << this->targetVoc.tokenList[*it]->str << " ";
  }
  std::cout << std::endl;

  for (auto it = encState.begin(); it != encState.end(); ++it){
    delete *it;
  }
  for (auto it = decState.begin(); it != decState.end(); ++it){
    delete *it;
  }
}

void EncDec::train(EncDec::Data* data, LSTM::Grad& lstmSrcGrad, LSTM::Grad& lstmTgtGrad, SoftMax::Grad& softmaxGrad, EncDec::Grad& embedGrad, double& loss){
  MatD targetDist;

  loss = 0.0;
  this->encode(data->src, data->encState);
  
  for (int i = 0; i < (int)data->tgt.size(); ++i){
    if (i == 0){
      data->decState[0]->h = data->encState.back()->h;
      data->decState[0]->c = data->encState.back()->c;
    }
    else {
      this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), data->decState[i-1], data->decState[i]);
    }
    
    this->softmax.calcDist(data->decState[i]->h, targetDist);
    loss += this->softmax.calcLoss(targetDist, data->tgt[i]);
    this->softmax.backward(data->decState[i]->h, targetDist, data->tgt[i], data->decState[i]->delh, softmaxGrad);
  }

  loss /= data->tgt.size();
  data->decState.back()->delc = this->zeros;

  for (int i = data->tgt.size()-1; i >= 1; --i){
    this->dec.backward(data->decState[i-1], data->decState[i], lstmTgtGrad, this->targetEmbed.col(data->tgt[i-1]));

    if (embedGrad.targetEmbed.count(data->tgt[i-1])){
      embedGrad.targetEmbed.at(data->tgt[i-1]) += data->decState[i]->delx;
    }
    else {
      embedGrad.targetEmbed[data->tgt[i-1]] = data->decState[i]->delx;
    }
  }
  
  data->encState.back()->delc = data->decState[0]->delc;
  data->encState.back()->delh = data->decState[0]->delh;

  for (int i = data->src.size(); i >= 1; --i){
    data->encState[i-1]->delh = this->zeros;
    this->enc.backward(data->encState[i-1], data->encState[i], lstmSrcGrad, this->sourceEmbed.col(data->src[i-1]));

    if (embedGrad.sourceEmbed.count(data->src[i-1])){
      embedGrad.sourceEmbed.at(data->src[i-1]) += data->encState[i]->delx;
    }
    else {
      embedGrad.sourceEmbed[data->src[i-1]] = data->encState[i]->delx;
    }
  }
}

void EncDec::train(const double learningRate){
  LSTM::Grad lstmSrcGrad(this->enc);
  LSTM::Grad lstmTgtGrad(this->dec);
  SoftMax::Grad softmaxGrad(this->softmax);
  EncDec::Grad embedGrad;
  double loss = 0.0, lossTmp;
  double gradNorm, lr = learningRate;

  this->rnd.shuffle(this->trainData);

  for (int i = 0; i < (int)this->trainData.size(); ++i){
    this->train(this->trainData[i], lstmSrcGrad, lstmTgtGrad, softmaxGrad, embedGrad, lossTmp);
    loss += lossTmp;
    gradNorm = sqrt(lstmSrcGrad.norm()+lstmTgtGrad.norm()+softmaxGrad.norm()+embedGrad.norm());
    lr = (gradNorm > 5.0 ? 5.0*learningRate/gradNorm : learningRate);

    this->enc.sgd(lstmSrcGrad, lr);
    this->dec.sgd(lstmTgtGrad, lr);
    this->softmax.sgd(softmaxGrad, lr);

    for (auto it = embedGrad.sourceEmbed.begin(); it != embedGrad.sourceEmbed.end(); ++it){
      this->sourceEmbed.col(it->first) -= lr*it->second;
    }
    for (auto it = embedGrad.targetEmbed.begin(); it != embedGrad.targetEmbed.end(); ++it){
      this->targetEmbed.col(it->first) -= lr*it->second;
    }

    for (auto it = this->trainData[i]->encState.begin(); it != this->trainData[i]->encState.end(); ++it){
      (*it)->clear();
    }
    for (auto it = this->trainData[i]->decState.begin(); it != this->trainData[i]->decState.end(); ++it){
      (*it)->clear();
    }

    lstmSrcGrad.init();
    lstmTgtGrad.init();
    softmaxGrad.init();
    embedGrad.init();
  }

  std::cout << "Loss: " << loss/this->trainData.size() << std::endl;
}

void EncDec::demo(const std::string& src, const std::string& tgt){
  Vocabulary sourceVoc(src, 1);
  Vocabulary targetVoc(tgt, 1);
  std::vector<EncDec::Data*> trainData;
  std::ifstream ifsSrc(src.c_str());
  std::ifstream ifsTgt(tgt.c_str());
  std::vector<std::string> tokens;

  for (std::string line; std::getline(ifsSrc, line); ){
    trainData.push_back(new EncDec::Data);
    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it){
      trainData.back()->src.push_back(sourceVoc.tokenIndex.count(*it) ? sourceVoc.tokenIndex.at(*it) : sourceVoc.unkIndex);
      trainData.back()->encState.push_back(new LSTM::State);
    }

    std::reverse(trainData.back()->src.begin(), trainData.back()->src.end());
    trainData.back()->src.push_back(sourceVoc.eosIndex);
    trainData.back()->encState.push_back(new LSTM::State);
    trainData.back()->encState.push_back(new LSTM::State); //one for the initial state
  }

  int numLine = 0;

  for (std::string line; std::getline(ifsTgt, line); ){
    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it){
      trainData[numLine]->tgt.push_back(targetVoc.tokenIndex.count(*it) ? targetVoc.tokenIndex.at(*it) : targetVoc.unkIndex);
      trainData[numLine]->decState.push_back(new LSTM::State);
    }

    trainData[numLine]->tgt.push_back(targetVoc.eosIndex);
    trainData[numLine]->decState.push_back(new LSTM::State);
    ++numLine;
  }

  EncDec encdec(sourceVoc, targetVoc, trainData, 50, 50);

  std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
  std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;

  for (int i = 0; i < 50; ++i){
    std::cout << "Epoch " << i+1 << std::endl;
    encdec.train(0.5);
  }

  for (int i = 0; i < (int)trainData.size(); ++i){
    encdec.translate(trainData[i]->src);
  }
}
