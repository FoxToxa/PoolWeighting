#include "Instance.h"


Instance::Instance()
{
  size = 0;
  nfun = -1;
  data.clear();
}

Instance::Instance(Instance &TS)
{
  this->size = TS.size;
  this->nfun = TS.nfun;
  this->attr = TS.attr;
  this->data = TS.data;
}

Instance::~Instance()
{
  size = 0;
  nfun = 0;
  data.clear();
}

void Instance::addEl(string name)
{
  attr.push_back(name);
  size++;
  nfun++;
  return;
}

void Instance::add(string s)
{
  double *arr;
  arr = new double[size];
  int pos;
  string buf;
  for (int i = 0; i < size - 1; ++i)
  {
    pos = s.find(",");
    buf = s.substr(0, pos);
    arr[i] = stof(buf);
    s = s.substr(pos + 1);
  }
  arr[size - 1] = stoi(s, nullptr, 10);
  arr[size - 1] = stof(s);
  data.push_back(arr);
  return;
}

void Instance::sortByArg(int narg)
{
  sort(data.begin(), data.end(),
    [narg](const double* a, const double* b) -> bool
      {
        return a[narg] < b[narg];
      }
    );
  return;
}


void Instance::print()
{
  for (int j = 0; j < size; ++j)
    cout << attr[j] << " | ";
  cout << endl << endl;

  for (int i = 0; i < data.size(); ++i)
  {
    for (int j = 0; j < size; ++j)
      cout << data[i][j] << " ";
    cout << endl;
  }
  return;
}

double Instance::MeanVal()
{
  double S = 0;
  for (double *x : data)
  {
    S += x[nfun];
  }
  return S / data.size();
}

double Instance::Sum()
{
  double S = 0;
  for (double *x : data)
  {
    S += x[nfun];
  }
  return S;
}
