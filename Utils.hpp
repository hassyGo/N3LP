#pragma once

#include "Matrix.hpp"
#include <string>
#include <vector>
#include <fstream>

namespace Utils{
  inline double max(const double& x, const double& y){
    return x > y ? x : y;
  }

  inline double min(const double& x, const double& y){
    return x > y ? y : x;
  }

  inline bool isSpace(const char& c){
    return (c == ' ' || c == '\t');
  }

  inline void split(const std::string& str, std::vector<std::string>& res){
    bool tok = false;
    int beg = 0;

    res.clear();

    for (int i = 0, len = str.length(); i < len; ++i){
      if (!tok && !Utils::isSpace(str[i])){
	beg = i;
	tok = true;
      }

      if (tok && (i == len-1 || Utils::isSpace(str[i]))){
	tok = false;
	res.push_back(isSpace(str[i]) ? str.substr(beg, i-beg) : str.substr(beg, i-beg+1));
      }
    }
  }

  inline void split(const std::string& str, std::vector<std::string>& res, const char sep){
    bool tok = false;
    int beg = 0;

    res.clear();

    for (int i = 0, len = str.length(); i < len; ++i){
      if (!tok && str[i] != sep){
	beg = i;
	tok = true;
      }

      if (tok && (i == len-1 || str[i] == sep)){
	tok = false;
	res.push_back((str[i] == sep) ? str.substr(beg, i-beg) : str.substr(beg, i-beg+1));
      }
    }
  }

  inline double cosDis(const MatD& a, const MatD& b){
    return (a.array()*b.array()).sum()/(a.norm()*b.norm());
    //return a.col(0).dot(b.col(0))/(a.norm()*b.norm());
  }

  inline void infNan(const double& x){
    assert(!isnan(x) && !isinf(x));
  }

  inline void save(std::ofstream& ofs, const MatD& params){
    double val = 0.0;
    
    for (int i = 0; i < params.cols(); ++i){
      for (int j = 0; j < params.rows(); ++j){
	val = params.coeff(j, i);
	ofs.write((char*)&val, sizeof(double));
      }
    }
  }

  inline void load(std::ifstream& ifs, MatD& params){
    double val = 0.0;
    
    for (int i = 0; i < params.cols(); ++i){
      for (int j = 0; j < params.rows(); ++j){
	ifs.read((char*)&val, sizeof(double));
	params.coeffRef(j, i) = val;
      }
    }
  }

  inline double stdDev(const Eigen::MatrixXd& input){
    return ::sqrt(((Eigen::MatrixXd)((input.array()-input.sum()/input.rows()).pow(2.0))).sum()/(input.rows()-1));
  }
}
