#include "EncDec.hpp"

int main(int argc, char** argv){
  const std::string src = "./corpus/sample.en";
  const std::string tgt = "./corpus/sample.ja";

  Eigen::initParallel();
  EncDec::demo(src, tgt, src, tgt);

  return 0;
}
