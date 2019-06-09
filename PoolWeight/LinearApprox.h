#pragma once
#include "Instance.h"

class LinearApprox
{
public:
  LinearApprox(Instance& Inst);

  void CreateSimpleApproximation(int argindex = 0);
  void CreateSimpleApproximation2(int argindex = 0);
  void ShowApprFunc();
  double CheckDeviation(Instance& TI);
  vector<double> predict(Instance Inst);

  ~LinearApprox();

private:
  Instance TS;
  double a;
  double b;
  int narg;
};

