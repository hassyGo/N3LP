#include "EncDec.hpp"

int main(int argc, char** argv){
  const std::string src = "./corpus/sample.en";
  const std::string tgt = "./corpus/sample.ja";
  const std::string srcDev = "./corpus/sample.en.dev";
  const std::string tgtDev = "./corpus/sample.ja.dev";

  Eigen::initParallel();
  EncDec::demo(src, tgt, srcDev, tgtDev);

  return 0;
}
