#include "DecisionTree.h"



DecisionTree::DecisionTree(Instance &TrainSamp, int depth, int min_arg)
{
  this->TS = TrainSamp;
  this->max_depth = depth;
  this->min_arg = min_arg;
}

void DecisionTree::fit()
{
  value = TS.MeanVal();

  double base_error = get_base_error();
  double error = base_error;
  bool flag = false;

  if (max_depth <= 1)
    return;

  double left_val = 0;
  double right_val = 0;

  double prev_error1, prev_error2;
  double mean1, mean2;
  double sm1, sm2;
  int N1, N2;
  int N = TS.data.size();
  int thres;
  double delta1, delta2;
  double meanval = value;
  double sumval = TS.Sum();

  for (int arg = 0; arg < TS.size - 1; arg++)
  {
    prev_error1 = base_error;
    prev_error2 = 0;

    TS.sortByArg(arg);

    mean1 = meanval;
    mean2 = 0;

    sm1 = sumval;
    sm2 = 0;

    N1 = N;
    N2 = 0;
    thres = 1;

    while (thres < N - 1)
    {
      N1--;
      N2++;

      delta1 = (sm1 - TS.data[thres][TS.nfun]) / N1 - mean1;
      delta2 = (sm2 + TS.data[thres][TS.nfun]) / N2 - mean2;

      sm1 -= TS.data[thres][TS.nfun];
      sm2 += TS.data[thres][TS.nfun];

      prev_error1 += pow(delta1, 2) * N1;
      prev_error1 -= pow(TS.data[thres][TS.nfun] - mean1, 2);
      prev_error1 -= 2 * delta1 * (sm1 - mean1 * N1);
      mean1 = sm1 / N1;

      prev_error2 += pow(delta2, 2) * N2;
      prev_error2 += pow(TS.data[thres][TS.nfun] - mean2, 2);
      prev_error2 -= 2 * delta2 * (sm2 - mean2 * N1);
      mean2 = sm2 / N2;

      if (thres < N - 1 && abs(TS.data[thres][arg] - TS.data[thres + 1][arg]) < exp(-7))
      {
        thres++;
        continue;
      }

      if (prev_error1 + prev_error2 < error)
      {
        if (min(N1, N2) > min_arg)
        {
          numArg = arg;
          idRoot = thres;
          root = TS.data[thres][arg];

          left_val = mean1;
          right_val = mean2;

          flag = true;
          error = prev_error1 + prev_error2;
        }
      }

      thres++;
    }
  }

  if (numArg == -1)
    return;

  Instance leftInst;
  Instance rightInst(TS);
  splitByArg(leftInst, rightInst);

  left = new DecisionTree(leftInst, max_depth - 1, min_arg);
  left->fit();
  right = new DecisionTree(rightInst, max_depth - 1, min_arg);
  right->fit();
}

void DecisionTree::fitWeight()
{
  value = TS.MeanVal();

  double base_error = get_base_error();
  double error = base_error;
  bool flag = false;

  if (max_depth <= 1)
    return;

  double left_val = 0;
  double right_val = 0;

  double prev_error1, prev_error2;
  double mean1, mean2;
  double sm1, sm2;
  int N1, N2;
  int N = TS.data.size();
  int thres;
  double delta1, delta2;
  double meanval = value;
  double sumval = TS.Sum();

  for (int arg = 0; arg < TS.size - 1; arg++)
  {
    prev_error1 = base_error;
    prev_error2 = 0;

    TS.sortByArg(arg);

    mean1 = meanval;
    mean2 = 0;

    sm1 = sumval;
    sm2 = 0;

    N1 = TS.ComputeSumWeights();
    N2 = 0;
    thres = 1;

    while (thres < N - 1)
    {
      N1 -= TS.data[thres][TS.nfun + 1];
      N2 += TS.data[thres][TS.nfun + 1];

      delta1 = (sm1 - TS.data[thres][TS.nfun] * TS.data[thres][TS.nfun + 1]) / N1 - mean1;
      delta2 = (sm2 + TS.data[thres][TS.nfun] * TS.data[thres][TS.nfun + 1]) / N2 - mean2;

      sm1 -= TS.data[thres][TS.nfun] * TS.data[thres][TS.nfun + 1];
      sm2 += TS.data[thres][TS.nfun] * TS.data[thres][TS.nfun + 1];

      prev_error1 += pow(delta1, 2) * N1;
      prev_error1 -= pow(TS.data[thres][TS.nfun] * TS.data[thres][TS.nfun + 1] - mean1, 2);
      prev_error1 -= 2 * delta1 * (sm1 - mean1 * N1);
      mean1 = sm1 / N1;

      prev_error2 += pow(delta2, 2) * N2;
      prev_error2 += pow(TS.data[thres][TS.nfun] * TS.data[thres][TS.nfun + 1] - mean2, 2);
      prev_error2 -= 2 * delta2 * (sm2 - mean2 * N1);
      mean2 = sm2 / N2;

      if (thres < N - 1 && abs(TS.data[thres][arg] * TS.data[thres][TS.nfun + 1] - TS.data[thres + 1][arg] * TS.data[thres + 1][TS.nfun + 1]) < exp(-7))
      {
        thres++;
        continue;
      }

      if (prev_error1 + prev_error2 < error)
      {
        if (min(N1, N2) > min_arg)
        {
          numArg = arg;
          idRoot = thres;
          root = TS.data[thres][arg];

          left_val = mean1;
          right_val = mean2;

          flag = true;
          error = prev_error1 + prev_error2;
        }
      }

      thres++;
    }
  }

  if (numArg == -1)
    return;

  Instance leftInst;
  Instance rightInst(TS);
  splitByArg(leftInst, rightInst);

  left = new DecisionTree(leftInst, max_depth - 1, min_arg);
  left->fitWeight();
  right = new DecisionTree(rightInst, max_depth - 1, min_arg);
  right->fitWeight();
}

double DecisionTree::get_base_error()
{
  double Sum = 0;
  for (double *x : TS.data)
  {
    Sum += pow((x[TS.nfun] - value) * x[TS.nfun + 1], 2);
  }
  return Sum;
}

void DecisionTree::splitByArg(Instance &leftInst, Instance &rightInst)
{
  leftInst.nfun = rightInst.nfun;
  leftInst.attr = rightInst.attr;
  leftInst.size = rightInst.size;
  rightInst.sortByArg(numArg);
  for (int i = idRoot + 1; i < rightInst.data.size(); ++i)
  {
    leftInst.data.push_back(rightInst.data[i]);
  }

  rightInst.data.erase(rightInst.data.begin() + idRoot + 1, rightInst.data.end());
}

double DecisionTree::_predict(double *inst)
{
  if (numArg == -1)
    return value;

  if (inst[numArg] > root)
    return left->_predict(inst);
  else
    return right->_predict(inst);
}

vector<double> DecisionTree::predict(Instance inst)
{
  vector<double> res;
  for (double* x : inst.data)
  {
    res.push_back(_predict(x));
  }
  return res;
}

vector<double> DecisionTree::TuneWeights(Instance TestInst, int nTunes)
{

  vector<double> weights;
  vector<double> result;
  const int nrolls = 10;  // number of experiments
  double alpha = pow(2, -10);
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
        fitWeight();
        result = predict(TestInst);

        grad += eps * TestInst.RMSE(result) * alpha;
      }
      weights[i] = x + grad;

    }
  }
  return weights;
}

DecisionTree::~DecisionTree()
{
}
