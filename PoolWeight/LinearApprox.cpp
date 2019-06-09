#include "LinearApprox.h"


LinearApprox::LinearApprox(Instance& Inst)
{
  TS = Inst;
  a = 0;
  b = 0;
}

void LinearApprox::CreateSimpleApproximation(int argindex)
{
  double sa = 0;
  double sb = 0;
  double currA;
  double currB;
  int count = 0;
  narg = argindex;

  for (int i = 0; i < TS.data.size() - 1; ++i)
  {
    for (int j = i + 1; j < TS.data.size(); ++j)
    {
      if (TS.data[i][narg] == TS.data[j][narg])
        continue;
      currA = (TS.data[i][TS.nfun] - TS.data[j][TS.nfun]) / (TS.data[i][narg] - TS.data[j][narg]);
      currB = TS.data[i][TS.nfun] - TS.data[i][narg] * currA;
      sa += currA;
      sb += currB;
      count++;
    }
  }
  a = sa / count;
  b = sb / count;

  return;
}

void LinearApprox::CreateSimpleApproximation2(int argindex)
{
  narg = argindex;
  double sumx = 0;
  double sumxx = 0;
  double sumy = 0;
  double sumxy = 0;
  int n = TS.data.size();


  for (int i = 0; i < n; i++)
  {
    sumx += TS.data[i][narg];
    sumxx += TS.data[i][narg] * TS.data[i][narg];
    sumy += TS.data[i][TS.nfun];
    sumxy += TS.data[i][narg] * TS.data[i][TS.nfun];
  }

  a = (n * sumxy - sumx * sumy) / (n * sumxx - sumx * sumx);
  b = (sumy - a * sumx) / n;
}

void LinearApprox::ShowApprFunc()
{
  cout << TS.attr[TS.nfun] << " = " << a << " * " << TS.attr[narg] << " + " << b << endl;
  return;
}

double LinearApprox::CheckDeviation(Instance& TI)
{
  double sum = 0;
  for (int i = 0; i < TI.data.size(); ++i)
  {
    sum += TI.data[i][TI.nfun] - (a * TI.data[i][narg] + b);
  }
  return abs(sum / TI.data.size());
}

vector<double> LinearApprox::predict(Instance Inst)
{
  vector<double> res;
  for (int i = 0; i < Inst.data.size(); ++i)
  {
    res.push_back(a * Inst.data[i][narg] + b);
  }
  return res;
}

LinearApprox::~LinearApprox()
{
}
