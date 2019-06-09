#include "GradBoost.h"


GradBoost::GradBoost(int max_depth, int min_size, int n_estimators, double learn_rate)
{
  this->max_depth = max_depth;
  this->min_size = min_size;
  this->n_estimators = n_estimators;
  this->learn_rate = learn_rate;
}

void GradBoost::fit(Instance TS)
{
  DecisionTree *DT;
  for (double *x : TS.data)
  {
    Y.push_back(x[TS.nfun]);
  }
  vector<double> prediction;
  double mean = TS.MeanVal();
  for (int i = 0; i < TS.data.size(); ++i)
  {
    prediction.push_back(mean);
  }
  vector<double> resid;

  for (int i = 0; i < n_estimators; ++i)
  {
    if (i)
    {
      for (int j = 0; j < TS.data.size(); ++j)
      {
        TS.data[j][TS.nfun] = Y[j] - prediction[j];
      }
    }
    Trees.push_back(DecisionTree(TS, max_depth, min_size));
    //DT = new DecisionTree(TS, max_depth, min_size);
    //DT->fit();
    Trees[i].fit();
    resid = Trees[i].predict(TS);
    for (int j = 0; j < resid.size(); ++j)
    {
      prediction[j] += learn_rate * resid[j];
    }
  }

  for (int i = 0; i < Y.size(); ++i)
    TS.data[i][TS.nfun] = Y[i];

}

vector<double> GradBoost::predict(Instance Inst)
{
  vector<double> res;
  double mean = Inst.MeanVal();
  for (int i = 0; i < Inst.data.size(); ++i)
    res.push_back(mean);
  vector<double> pred;
  for (int i = 0; i < n_estimators; ++i)
  {
    pred = Trees[i].predict(Inst);
    for (int j = 0; j < pred.size(); ++j)
    {
      res[j] += learn_rate * pred[j];
    }
  }
  return res;
}

GradBoost::~GradBoost()
{
}
