#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <random>

using namespace std;

class Instance
{
public:
  Instance();
  Instance(Instance &TS);

  void addEl(string name);	// add column
  void add(string s);		// add new string of data
  void sortByArg(int narg);	// sort for vector by narg
  void print();

  double MeanVal();
  double Sum();
  double ComputeSumWeights();
  double RMSE(vector<double> res);

  void SetWeights(vector<double> weights);

  ~Instance();

  vector<double*> data; // all arguments here
  vector<string> attr;
  int nfun;	// number of y arg column

  int size;     // number of arguments
  
  

};

