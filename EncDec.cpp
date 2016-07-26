#include "EncDec.hpp"
#include "Utils.hpp"
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <omp.h>

EncDec::EncDec(Vocabulary& sourceVoc_, Vocabulary& targetVoc_, std::vector<EncDec::Data*>& trainData_, std::vector<EncDec::Data*>& devData_, const int inputDim, const int hiddenDim, const bool useBlackout_):
  useBlackout(useBlackout_), sourceVoc(sourceVoc_), targetVoc(targetVoc_), trainData(trainData_), devData(devData_)
{
  const Real scale = 0.1;

  this->enc = LSTM(inputDim, hiddenDim);
  this->enc.init(this->rnd, scale);
  this->dec = LSTM(inputDim, hiddenDim);
  this->dec.init(this->rnd, scale);
  this->sourceEmbed = MatD(inputDim, this->sourceVoc.tokenList.size());
  this->targetEmbed = MatD(inputDim, this->targetVoc.tokenList.size());
  this->rnd.uniform(this->sourceEmbed, scale);
  this->rnd.uniform(this->targetEmbed, scale);
  this->zeros = VecD::Zero(hiddenDim);

  //Bias 1
  this->enc.bf.fill(1.0);
  this->dec.bf.fill(1.0);

  if (!this->useBlackout){
    this->softmax = SoftMax(hiddenDim, this->targetVoc.tokenList.size());
  }
  else {
    const int sampleNum = 100;
    const Real alpha = 0.4;
    VecD freq = VecD(this->targetVoc.tokenList.size());
    
    for (int i = 0; i < (int)this->targetVoc.tokenList.size(); ++i){
      freq.coeffRef(i, 0) = this->targetVoc.tokenList[i]->count;
    }

    this->blackout = BlackOut(hiddenDim, this->targetVoc.tokenList.size(), sampleNum);
    this->blackout.initSampling(freq, alpha);
  }

  for (int j = 0; j < (int)this->devData.size(); ++j){
    this->encStateDev.push_back(std::vector<LSTM::State*>());
    this->decStateDev.push_back(std::vector<LSTM::State*>());
    
    for (int i = 0; i < 200; ++i){
      this->encStateDev.back().push_back(new LSTM::State);
      this->decStateDev.back().push_back(new LSTM::State);
    }
  }
}

void EncDec::encode(const std::vector<int>& src, std::vector<LSTM::State*>& encState){
  encState[0]->h = this->zeros;
  encState[0]->c = this->zeros;

  for (int i = 0; i < (int)src.size(); ++i){
    this->enc.forward(this->sourceEmbed.col(src[i]), encState[i], encState[i+1]);
  }
}

struct sort_pred {
  bool operator()(const EncDec::DecCandidate left, const EncDec::DecCandidate right) {
    return left.score > right.score;
  }

  bool operator()(const EncDec::Data* left, const EncDec::Data* right) {
    return (left->src.size()+left->tgt.size()) < (right->src.size()+right->tgt.size());
    //return left->tgt.size() < right->tgt.size();
  }
};

void EncDec::translate(const std::vector<int>& src, const int beam, const int maxLength, const int showNum){
  const Real minScore = -1.0e+05;
  MatD score(this->targetEmbed.cols(), beam);
  VecD targetDist;
  std::vector<LSTM::State*> encState, stateList;
  std::vector<EncDec::DecCandidate> candidate(beam), candidateTmp(beam);

  for (int i = 0; i <= (int)src.size(); ++i){
    encState.push_back(new LSTM::State);
    stateList.push_back(encState.back());
  }

  this->encode(src, encState);

  for (int i = 0; i < maxLength; ++i){
    for (int j = 0; j < beam; ++j){
      if (candidate[j].stop){
	score.col(j).fill(candidate[j].score);
	continue;
      }

      candidate[j].decState.push_back(new LSTM::State);
      stateList.push_back(candidate[j].decState.back());

      if (i == 0){
	candidate[j].decState[i]->h = encState.back()->h;
	candidate[j].decState[i]->c = encState.back()->c;
      }
      else {
	this->dec.forward(this->targetEmbed.col(candidate[j].tgt[i-1]), candidate[j].decState[i-1], candidate[j].decState[i]);
      }

      if (!this->useBlackout){
	this->softmax.calcDist(candidate[j].decState[i]->h, targetDist);
      }
      else {
	this->blackout.calcDist(candidate[j].decState[i]->h, targetDist);
      }

      score.col(j).array() = candidate[j].score+targetDist.array().log();
    }

    for (int j = 0, row, col; j < beam; ++j){
      score.maxCoeff(&row, &col);
      candidateTmp[j] = candidate[col];
      candidateTmp[j].score = score.coeff(row, col);

      if (candidateTmp[j].stop){
	score.col(col).fill(minScore);
	continue;
      }

      if (row == this->targetVoc.eosIndex){
	candidateTmp[j].stop = true;
      }

      candidateTmp[j].tgt.push_back(row);

      if (i == 0){
	score.row(row).fill(minScore);
      }
      else {
	score.coeffRef(row, col) = minScore;
      }
    }

    candidate = candidateTmp;
    std::sort(candidate.begin(), candidate.end(), sort_pred());

    if (candidate[0].tgt.back() == this->targetVoc.eosIndex){
      break;
    }
  }

  if (showNum <= 0){
    return;
  }

  for (auto it = src.begin(); it != src.end(); ++it){
    std::cout << this->sourceVoc.tokenList[*it]->str << " ";
  }
  std::cout << std::endl;

  for (int i = 0; i < showNum; ++i){
    std::cout << i+1 << " (" << candidate[i].score << "): ";
    for (auto it = candidate[i].tgt.begin(); it != candidate[i].tgt.end(); ++it){
      std::cout << this->targetVoc.tokenList[*it]->str << " ";
    }
    std::cout << std::endl;
  }

  for (auto it = stateList.begin(); it != stateList.end(); ++it){
    delete *it;
  }
}

bool EncDec::translate(std::vector<int>& output, const std::vector<int>& src, const int beam, const int maxLength){
  const Real minScore = -1.0e+05;
  MatD score(this->targetEmbed.cols(), beam);
  VecD targetDist;
  std::vector<LSTM::State*> encState, stateList;
  std::vector<EncDec::DecCandidate> candidate(beam), candidateTmp(beam);

  for (int i = 0; i <= (int)src.size(); ++i){
    encState.push_back(new LSTM::State);
    stateList.push_back(encState.back());
  }

  this->encode(src, encState);

  for (int i = 0; i < maxLength; ++i){
    for (int j = 0; j < beam; ++j){
      if (candidate[j].stop){
	score.col(j).fill(candidate[j].score);
	continue;
      }

      candidate[j].decState.push_back(new LSTM::State);
      stateList.push_back(candidate[j].decState.back());

      if (i == 0){
	candidate[j].decState[i]->h = encState.back()->h;
	candidate[j].decState[i]->c = encState.back()->c;
      }
      else {
	this->dec.forward(this->targetEmbed.col(candidate[j].tgt[i-1]), candidate[j].decState[i-1], candidate[j].decState[i]);
      }

      if (!this->useBlackout){
	this->softmax.calcDist(candidate[j].decState[i]->h, targetDist);
      }
      else {
	this->blackout.calcDist(candidate[j].decState[i]->h, targetDist);
      }

      score.col(j).array() = candidate[j].score+targetDist.array().log();
    }

    for (int j = 0, row, col; j < beam; ++j){
      score.maxCoeff(&row, &col);
      candidateTmp[j] = candidate[col];
      candidateTmp[j].score = score.coeff(row, col);

      if (candidateTmp[j].stop){
	score.col(col).fill(minScore);
	continue;
      }

      if (row == this->targetVoc.eosIndex){
	candidateTmp[j].stop = true;
      }

      candidateTmp[j].tgt.push_back(row);

      if (i == 0){
	score.row(row).fill(minScore);
      }
      else {
	score.coeffRef(row, col) = minScore;
      }
    }

    candidate = candidateTmp;
    std::sort(candidate.begin(), candidate.end(), sort_pred());

    if (candidate[0].tgt.back() == this->targetVoc.eosIndex){
      break;
    }
  }

  bool status;

  output.clear();

  if (candidate[0].tgt.back() == this->targetVoc.eosIndex){
    for (int i = 0; i < (int)candidate[0].tgt.size()-1; ++i){
      output.push_back(candidate[0].tgt[i]);
    }

    status = true;
  }
  else {
    output = candidate[0].tgt;
    status = false;
  }

  for (auto it = stateList.begin(); it != stateList.end(); ++it){
    delete *it;
  }

  return status;
}

Real EncDec::calcLoss(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState){
  VecD targetDist;
  Real loss = 0.0;

  this->encode(data->src, encState);

  for (int i = 0; i < (int)data->tgt.size(); ++i){
    if (i == 0){
      decState[0]->h = encState[data->src.size()]->h;
      decState[0]->c = encState[data->src.size()]->c;
    }
    else {
      this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), decState[i-1], decState[i]);
    }

    if (!this->useBlackout){
      this->softmax.calcDist(decState[i]->h, targetDist);
      loss += this->softmax.calcLoss(targetDist, data->tgt[i]);
    }
    else {
      this->blackout.calcDist(decState[i]->h, targetDist);
      loss += this->blackout.calcLoss(targetDist, data->tgt[i]);
    }
  }
  
  return loss;
}

Real EncDec::calcPerplexity(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState){
  VecD targetDist;
  Real perp = 0.0;

  this->encode(data->src, encState);

  for (int i = 0; i < (int)data->tgt.size(); ++i){
    if (i == 0){
      decState[0]->h = encState[data->src.size()]->h;
      decState[0]->c = encState[data->src.size()]->c;
    }
    else {
      this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), decState[i-1], decState[i]);
    }

    if (!this->useBlackout){
      this->softmax.calcDist(decState[i]->h, targetDist);
    }
    else {
      this->blackout.calcDist(decState[i]->h, targetDist);
    }

    perp -= log(targetDist.coeff(data->tgt[i], 0));
  }
  
  return exp(perp/data->tgt.size());
}

void EncDec::gradCheck(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, MatD& param, const MatD& grad){
  const Real EPS = 1.0e-04;
  Real val = 0.0, objPlus = 0.0, objMinus = 0.0;

  for (int i = 0; i < param.rows(); ++i){
    for (int j = 0; j < param.cols(); ++j){
      val = param.coeff(i, j);
      param.coeffRef(i, j) = val+EPS;
      objPlus = this->calcLoss(data, encState, decState);
      param.coeffRef(i, j) = val-EPS;
      objMinus = this->calcLoss(data, encState, decState);
      param.coeffRef(i, j) = val;

      std::cout << "Grad: " << grad.coeff(i, j) << std::endl;
      std::cout << "Enum: " << (objPlus-objMinus)/(2.0*EPS) << std::endl;
    }
  }
}

void EncDec::gradCheck(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad){
  std::cout << "Gradient checking..." << std::endl;
  this->gradCheck(data, encState, decState, this->enc.Whi, grad.lstmSrcGrad.Whi);
}

void EncDec::train(EncDec::Data* data, std::vector<LSTM::State*>& encState, std::vector<LSTM::State*>& decState, EncDec::Grad& grad, Real& loss){
  VecD targetDist;

  loss = 0.0;
  this->encode(data->src, encState);

  for (int i = 0; i < (int)data->tgt.size(); ++i){
    if (i == 0){
      decState[0]->h = encState[data->src.size()]->h;
      decState[0]->c = encState[data->src.size()]->c;
    }
    else {
      this->dec.forward(this->targetEmbed.col(data->tgt[i-1]), decState[i-1], decState[i]);
    }

    if (!this->useBlackout){
      this->softmax.calcDist(decState[i]->h, targetDist);
      loss += this->softmax.calcLoss(targetDist, data->tgt[i]);
      this->softmax.backward(decState[i]->h, targetDist, data->tgt[i], decState[i]->delh, grad.softmaxGrad);
    }
    else {
      this->blackout.sampling(data->tgt[i], grad.blackoutState);
      this->blackout.calcSampledDist(decState[i]->h, targetDist, grad.blackoutState);
      loss += this->blackout.calcSampledLoss(targetDist);
      this->blackout.backward(decState[i]->h, targetDist, grad.blackoutState, decState[i]->delh, grad.blackoutGrad);
    }
  }

  decState[data->tgt.size()-1]->delc = this->zeros;

  for (int i = data->tgt.size()-1; i >= 1; --i){
    decState[i-1]->delc = this->zeros;
    this->dec.backward(decState[i-1], decState[i], grad.lstmTgtGrad, this->targetEmbed.col(data->tgt[i-1]));

    if (grad.targetEmbed.count(data->tgt[i-1])){
      grad.targetEmbed.at(data->tgt[i-1]) += decState[i]->delx;
    }
    else {
      grad.targetEmbed[data->tgt[i-1]] = decState[i]->delx;
    }
  }
  
  encState[data->src.size()]->delc = decState[0]->delc;
  encState[data->src.size()]->delh = decState[0]->delh;

  for (int i = data->src.size(); i >= 1; --i){
    encState[i-1]->delh = this->zeros;
    encState[i-1]->delc = this->zeros;
    this->enc.backward(encState[i-1], encState[i], grad.lstmSrcGrad, this->sourceEmbed.col(data->src[i-1]));

    if (grad.sourceEmbed.count(data->src[i-1])){
      grad.sourceEmbed.at(data->src[i-1]) += encState[i]->delx;
    }
    else {
      grad.sourceEmbed[data->src[i-1]] = encState[i]->delx;
    }
  }
}

void EncDec::trainOpenMP(const Real learningRate, const int miniBatchSize, const int numThreads){
  static std::vector<EncDec::ThreadArg*> args;
  static std::vector<std::pair<int, int> > miniBatch;
  static EncDec::Grad grad;
  Real lossTrain = 0.0, perpDev = 0.0, denom = 0.0;
  Real gradNorm, lr = learningRate;
  const Real clipThreshold = 3.0;
  struct timeval start, end;

  if (args.empty()){
    for (int i = 0; i < numThreads; ++i){
      args.push_back(new EncDec::ThreadArg(*this));

      for (int j = 0; j < 200; ++j){
	args[i]->encState.push_back(new LSTM::State);
	args[i]->decState.push_back(new LSTM::State);
      }
    }

    for (int i = 0, step = this->trainData.size()/miniBatchSize; i < step; ++i){
      miniBatch.push_back(std::pair<int, int>(i*miniBatchSize, (i == step-1 ? this->trainData.size()-1 : (i+1)*miniBatchSize-1)));
    }

    grad.lstmSrcGrad = LSTM::Grad(this->enc);
    grad.lstmTgtGrad = LSTM::Grad(this->dec);
    grad.softmaxGrad = SoftMax::Grad(this->softmax);
    grad.blackoutGrad = BlackOut::Grad();

    //std::sort(this->trainData.begin(), this->trainData.end(), sort_pred());
  }

  //this->rnd.shuffle(miniBatch);
  this->rnd.shuffle(this->trainData);
  gettimeofday(&start, 0);

  int count = 0;

  for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it){
    std::cout << "\r"
	      << "Progress: " << ++count << "/" << miniBatch.size() << " mini batches" << std::flush;

#pragma omp parallel for num_threads(numThreads) schedule(dynamic) shared(args)
    for (int i = it->first; i <= it->second; ++i){
      int id = 0;//omp_get_thread_num();
      Real loss;
      this->train(this->trainData[i], args[id]->encState, args[id]->decState, args[id]->grad, loss);
      args[id]->loss += loss;
    }

    for (int id = 0; id < numThreads; ++id){
      grad += args[id]->grad;
      args[id]->grad.init();
      lossTrain += args[id]->loss;
      args[id]->loss = 0.0;
    }

    gradNorm = sqrt(grad.norm())/miniBatchSize;
    Utils::infNan(gradNorm);
    lr = (gradNorm > clipThreshold ? clipThreshold*learningRate/gradNorm : learningRate);
    lr /= miniBatchSize;

    this->enc.sgd(grad.lstmSrcGrad, lr);
    this->dec.sgd(grad.lstmTgtGrad, lr);

    if (!this->useBlackout){
      this->softmax.sgd(grad.softmaxGrad, lr);
    }
    else {
      this->blackout.sgd(grad.blackoutGrad, lr);
    }

    for (auto it = grad.sourceEmbed.begin(); it != grad.sourceEmbed.end(); ++it){
      this->sourceEmbed.col(it->first) -= lr*it->second;
    }
    for (auto it = grad.targetEmbed.begin(); it != grad.targetEmbed.end(); ++it){
      this->targetEmbed.col(it->first) -= lr*it->second;
    }

    grad.init();
  }

  std::cout << std::endl;
  gettimeofday(&end, 0);
  std::cout << "Training time for this epoch: " << (end.tv_sec-start.tv_sec)/60.0 << " min." << std::endl;
  std::cout << "Training Loss (/sentence):    " << lossTrain/this->trainData.size() << std::endl;
  gettimeofday(&start, 0);

#pragma omp parallel for num_threads(numThreads) schedule(dynamic) shared(perpDev, denom)
  for (int i = 0; i < (int)this->devData.size(); ++i){
    Real perp = this->calcLoss(this->devData[i], this->encStateDev[i], this->decStateDev[i]);
    
    for (auto it = this->encStateDev[i].begin(); it != this->encStateDev[i].end(); ++it){
      (*it)->clear();
    }
    for (auto it = this->decStateDev[i].begin(); it != this->decStateDev[i].end(); ++it){
      (*it)->clear();
    }

#pragma omp critical
    {
      perpDev += perp;
      denom += this->devData[i]->tgt.size();
    }
  }

  gettimeofday(&end, 0);
  std::cout << "Evaluation time for this epoch: " << (end.tv_sec-start.tv_sec)/60.0 << " min." << std::endl;
  std::cout << "Development loss (/sentence): " << perpDev/this->devData.size() << std::endl;
  std::cout << "Development perplexity (global): " << exp(perpDev/denom) << std::endl;
}

void EncDec::demo(const std::string& srcTrain, const std::string& tgtTrain, const std::string& srcDev, const std::string& tgtDev){
  const int threSource = 1;
  const int threTarget = 1;
  Vocabulary sourceVoc(srcTrain, threSource);
  Vocabulary targetVoc(tgtTrain, threTarget);
  std::vector<EncDec::Data*> trainData, devData;
  std::ifstream ifsSrcTrain(srcTrain.c_str());
  std::ifstream ifsTgtTrain(tgtTrain.c_str());
  std::ifstream ifsSrcDev(srcDev.c_str());
  std::ifstream ifsTgtDev(tgtDev.c_str());
  std::vector<std::string> tokens;
  int numLine = 0;

  //training data
  for (std::string line; std::getline(ifsSrcTrain, line); ){
    trainData.push_back(new EncDec::Data);
    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it){
      trainData.back()->src.push_back(sourceVoc.tokenIndex.count(*it) ? sourceVoc.tokenIndex.at(*it) : sourceVoc.unkIndex);
    }

    //std::reverse(trainData.back()->src.begin(), trainData.back()->src.end());
    trainData.back()->src.push_back(sourceVoc.eosIndex);
  }

  for (std::string line; std::getline(ifsTgtTrain, line); ){
    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it){
      trainData[numLine]->tgt.push_back(targetVoc.tokenIndex.count(*it) ? targetVoc.tokenIndex.at(*it) : targetVoc.unkIndex);
    }

    trainData[numLine]->tgt.push_back(targetVoc.eosIndex);
    ++numLine;
  }

  //development data
  numLine = 0;

  for (std::string line; std::getline(ifsSrcDev, line); ){
    devData.push_back(new EncDec::Data);
    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it){
      devData.back()->src.push_back(sourceVoc.tokenIndex.count(*it) ? sourceVoc.tokenIndex.at(*it) : sourceVoc.unkIndex);
    }

    //std::reverse(devData.back()->src.begin(), devData.back()->src.end());
    devData.back()->src.push_back(sourceVoc.eosIndex);
  }

  for (std::string line; std::getline(ifsTgtDev, line); ){
    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it){
      devData[numLine]->tgt.push_back(targetVoc.tokenIndex.count(*it) ? targetVoc.tokenIndex.at(*it) : targetVoc.unkIndex);
    }

    devData[numLine]->tgt.push_back(targetVoc.eosIndex);
    ++numLine;
  }

  Real learningRate = 0.5;
  const int inputDim = 200;
  const int hiddenDim = 200;
  const int miniBatchSize = 1;
  const int numThread = 1;
  const bool useBlackout = true;
  EncDec encdec(sourceVoc, targetVoc, trainData, devData, inputDim, hiddenDim, useBlackout);
  auto test = trainData[0]->src;

  std::cout << "# of training data:    " << trainData.size() << std::endl;
  std::cout << "# of development data: " << devData.size() << std::endl;
  std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
  std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;
  
  for (int i = 0; i < 30; ++i){
    if (i+1 >= 6){
      //learningRate *= 0.5;
    }

    std::cout << "\nEpoch " << i+1 << std::endl;
    encdec.trainOpenMP(learningRate, miniBatchSize, numThread);
    std::cout << "### Greedy ###" << std::endl;
    encdec.translate(test, 1, 100, 1);
    std::cout << "### Beam search ###" << std::endl;
    encdec.translate(test, 20, 100, 5);

    std::ostringstream oss;
    oss << "model." << i+1 << "itr.bin";
    //encdec.save(oss.str());
  }

  //intereactive translation
  std::cout << "Interactive translation" << std::endl;

  for (std::string line; std::getline(std::cin, line); ){
    if (line == ""){
      continue;
    }

    EncDec::Data tmp;

    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it){
      tmp.src.push_back(sourceVoc.tokenIndex.count(*it) ? sourceVoc.tokenIndex.at(*it) : sourceVoc.unkIndex);
    }

    //std::reverse(tmp.src.begin(), tmp.src.end());
    tmp.src.push_back(sourceVoc.eosIndex);

    std::cout << "### Greedy ###" << std::endl;
    encdec.translate(tmp.src, 1, 100, 1);
    std::cout << "### Beam search ###" << std::endl;
    encdec.translate(tmp.src, 12, 100, 10);
  }
  
  return;

  encdec.load("model.1itr.bin");

  struct timeval start, end;
  
  //translation
  std::vector<std::vector<int> > output(encdec.devData.size());
  gettimeofday(&start, 0);
#pragma omp parallel for num_threads(numThread) schedule(dynamic) shared(output, encdec)
  for (int i = 0; i < (int)encdec.devData.size(); ++i){
    encdec.translate(output[i], encdec.devData[i]->src, 20, 100);
  }

  gettimeofday(&end, 0);
  std::cout << "Translation time: " << (end.tv_sec-start.tv_sec)/60.0 << " min." << std::endl;
 
  std::ofstream ofs("translation.txt");
  
  for (auto it = output.begin(); it != output.end(); ++it){
    for (auto it2 = it->begin(); it2 != it->end(); ++it2){
      ofs << encdec.targetVoc.tokenList[(*it2)]->str << " ";
    }
    ofs << std::endl;
  }
}

void EncDec::save(const std::string& fileName){
  std::ofstream ofs(fileName.c_str(), std::ios::out|std::ios::binary);

  assert(ofs);

  this->enc.save(ofs);
  this->dec.save(ofs);
  Utils::save(ofs, sourceEmbed);
  Utils::save(ofs, targetEmbed);

  if (this->useBlackout){
    this->blackout.save(ofs);
  }
  else {
    this->softmax.save(ofs);
  }
}

void EncDec::load(const std::string& fileName){
  std::ifstream ifs(fileName.c_str(), std::ios::in|std::ios::binary);

  assert(ifs);

  this->enc.load(ifs);
  this->dec.load(ifs);
  Utils::load(ifs, sourceEmbed);
  Utils::load(ifs, targetEmbed);

  if (this->useBlackout){
    this->blackout.load(ifs);
  }
  else {
    this->softmax.load(ifs);
  }
}
