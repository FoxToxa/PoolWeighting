#pragma once
#include "Instance.h"

class LinearApprox
{
public:
  LinearApprox(Instance& Inst);

  void CreateSimpleApproximation(int argindex = 0);
  void ShowApprFunc();
  vector<double> TuneWeights(Instance TestInst, int nTunes = 1);
  double CheckDeviation(Instance& TI);
  vector<double> predict(Instance Inst);

  ~LinearApprox();

//private:
  Instance TS;
  double a;
  double b;
  int narg;
};

