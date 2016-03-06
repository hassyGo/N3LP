#pragma once

#include "Matrix.hpp"
#include <string>
#include <vector>
#include <fstream>

namespace Utils{
  inline Real max(const Real& x, const Real& y){
    return x > y ? x : y;
  }

  inline Real min(const Real& x, const Real& y){
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

  template <typename T> inline void swap(std::vector<T>& vec){
    std::vector<T>().swap(vec);
  }

  inline Real cosDis(const MatD& a, const MatD& b){
    return (a.array()*b.array()).sum()/(a.norm()*b.norm());
    //return a.col(0).dot(b.col(0))/(a.norm()*b.norm());
  }

  inline void infNan(const Real& x){
    assert(!isnan(x) && !isinf(x));
  }

  inline void save(std::ofstream& ofs, const MatD& params){
    Real val = 0.0;
    
    for (int i = 0; i < params.cols(); ++i){
      for (int j = 0; j < params.rows(); ++j){
	val = params.coeff(j, i);
	ofs.write((char*)&val, sizeof(Real));
      }
    }
  }
  inline void save(std::ofstream& ofs, const VecD& params){
    Real val = 0.0;
    
    for (int i = 0; i < params.rows(); ++i){
      val = params.coeff(i, 0);
      ofs.write((char*)&val, sizeof(Real));
    }
  }

  inline void load(std::ifstream& ifs, MatD& params){
    Real val = 0.0;
    
    for (int i = 0; i < params.cols(); ++i){
      for (int j = 0; j < params.rows(); ++j){
	ifs.read((char*)&val, sizeof(Real));
	params.coeffRef(j, i) = val;
      }
    }
  }
  inline void load(std::ifstream& ifs, VecD& params){
    Real val = 0.0;
    
    for (int i = 0; i < params.rows(); ++i){
      ifs.read((char*)&val, sizeof(Real));
      params.coeffRef(i, 0) = val;
    }
  }

  inline Real stdDev(const Eigen::MatrixXd& input){
    return ::sqrt(((Eigen::MatrixXd)((input.array()-input.sum()/input.rows()).pow(2.0))).sum()/(input.rows()-1));
  }
}
