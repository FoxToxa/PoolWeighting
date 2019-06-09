#pragma once
#include "Instance.h"
class DecisionTree
{
public:
  Instance TS;

  DecisionTree *left;
  DecisionTree *right;
  int max_depth;
  int min_arg;

  int numArg = -1;
  int idRoot = -1;
  double root;
  double value = 0;

  void fit();

  void splitByArg(Instance &leftInst, Instance &rightInst);

  double _predict(double *inst);
  vector<double> predict(Instance inst);

  double get_base_error();

  DecisionTree(Instance &TrainSamp, int depth = 2, int min_arg = 1);
  ~DecisionTree();
};

