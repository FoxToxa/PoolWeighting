#include "LinearApprox.h"


LinearApprox::LinearApprox(Instance& Inst)
{
  TS = Inst;
  a = 0;
  b = 0;
}

void LinearApprox::CreateSimpleApproximation(int argindex)
{
  narg = argindex;
  double sumx = 0;
  double sumxx = 0;
  double sumy = 0;
  double sumxy = 0;
  int n = TS.ComputeSumWeights();//TS.data.size();


  for (int i = 0; i < TS.data.size(); i++)
  {
    sumx += TS.data[i][narg] * TS.data[i][TS.nfun + 1];
    sumxx += TS.data[i][narg] * TS.data[i][narg] * TS.data[i][TS.nfun + 1];
    sumy += TS.data[i][TS.nfun] * TS.data[i][TS.nfun + 1];
    sumxy += TS.data[i][narg] * TS.data[i][TS.nfun] * TS.data[i][TS.nfun + 1];
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

vector<double> LinearApprox::TuneWeights(Instance TestInst, int nTunes)
{
  vector<double> weights;
  vector<double> result;
  const int nrolls = 50;  // number of experiments
  double alpha = pow(2, -15);
  double grad;
  double eps;
  double x;
  for (int i = 0; i < TS.data.size(); ++i)
  {
    weights.push_back(double(rand() % 12) / 10);
  }

  default_random_engine generator;
  for (int iter = 0; iter < nTunes; ++iter)
  {
    for (int i = 0; i < weights.size(); ++i)
    {
      grad = 0;
      x = weights[i];
      normal_distribution<double> distribution(0.0, x / 5);
      for (int j = 0; j < nrolls; ++j)
      {
        eps = distribution(generator);
        weights[i] = x + eps;
        TS.SetWeights(weights);
        CreateSimpleApproximation(0);
        result = predict(TestInst);

        grad += eps * TestInst.RMSE(result) * alpha;
      }
      weights[i] = x + grad;

    }
  }
  return weights;
}

LinearApprox::~LinearApprox()
{
}
