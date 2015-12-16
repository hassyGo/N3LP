#pragma once

#include <string>
#include <unordered_map>
#include <vector>

class Vocabulary{
public:
  Vocabulary(const std::string& trainFile, const int tokenFreqThreshold);

  class Token;

  std::unordered_map<std::string, int> tokenIndex;
  std::vector<Vocabulary::Token*> tokenList;
  int eosIndex;
  int unkIndex;
};

class Vocabulary::Token{
public:
  Token(const std::string& str_, const int count_):
    str(str_), count(count_)
  {};

  std::string str;
  int count;
};
