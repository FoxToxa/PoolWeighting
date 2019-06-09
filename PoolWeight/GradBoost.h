#include "DecisionTree.h"
#pragma once
class GradBoost
{
public:
  GradBoost(int max_depth = 2, int min_size = 1, int n_estimators = 10, double learn_rate = 1);
  ~GradBoost();

  int max_depth;
  int min_size;
  int n_estimators;
  double learn_rate;

  vector<DecisionTree> Trees;
  vector<double> Y;

  void fit(Instance TS);

  vector<double> predict(Instance Inst);


};

