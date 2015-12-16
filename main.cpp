#include "EncDec.hpp"

int main(int argc, char** argv){
  const std::string src = "./corpus/sample.en";
  const std::string tgt = "./corpus/sample.ja";

  EncDec::demo(src, tgt);

  return 0;
}
