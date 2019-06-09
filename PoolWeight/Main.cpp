#include <fstream>
#include <ctime>
#include <random>

#include "LinearApprox.h"
#include "GradBoost.h"

using namespace std;

void SplitInstance2(Instance& TrainSample, Instance& ControlSample, double div = 0.9);
double mse(Instance Inst, vector<double> pred);

int main()
{
  int start_time = clock();
  ifstream inptFile;
  string inptStr;
  string atr, name;
  int pos;
  double meanSE;

  Instance Inst;

  inptFile.open("puma32.arff");
  //inptFile.open("cpu.arff");
  //inptFile.open("Test.arff");

  while (getline(inptFile, inptStr))
  {
    //cout << inptStr << endl;
    if (inptStr[0] == '%') continue; // comment skip it
    if (inptStr[0] == '@')	// parametr
    {
      atr = inptStr.substr(1, 9);

      if (atr == "attribute")
      {
        pos = inptStr.find(" ");
        name = inptStr.substr(pos + 1);
        pos = name.find(" ");
        name = name.substr(0, pos);
        Inst.addEl(name);
      }

      continue;
    }
    if (inptStr != "")
      Inst.add(inptStr);
  } 
  inptFile.close();
  //Inst.print();

  Instance TestInst;
  Instance ContInst;
  
  SplitInstance2(Inst, ContInst, 0.9);
  
  //ContInst.print();
  cout << endl;

  //SplitInstance2(ContInst, TestInst, 0.5);

  GradBoost GB(10, 1, 20, 0.1);
  GB.fit(Inst);

  vector<double> GBres = GB.predict(ContInst);
  meanSE = mse(ContInst, GBres);
  cout << "Gradient Boosting Mean Square Error: " << meanSE << endl;
  /*
  for (double x : GBres)
  {
    cout << x << endl;
  }
  */
  LinearApprox LA(Inst);
  LA.CreateSimpleApproximation2(0);
  int minIdx = 0;
  double minDev = LA.CheckDeviation(ContInst);
  double chkDev;
  for (int i = 1; i < Inst.nfun; ++i)
  {
    LA.CreateSimpleApproximation2(i);
    chkDev = LA.CheckDeviation(ContInst);
    if (chkDev < minDev)
    {
      minIdx = i;
      minDev = chkDev;
    }
  }

  LA.CreateSimpleApproximation2(minIdx);
  cout << endl;
  LA.ShowApprFunc();
  vector<double> linRes = LA.predict(ContInst);
  cout << endl;

  meanSE = mse(ContInst, linRes);
  cout << "Linear Approximation Mean Square Error: " << meanSE << endl;
  /*
  for (double x : linRes)
  {
    cout << x << endl;
  }
  */
  int end_time = clock();
  cout << endl << "Time: " << end_time - start_time << " ms" << endl;
  
  return 0;
}

void SplitInstance2(Instance& TrainSample, Instance& ControlSample, double div)
{
  random_device rd;
  mt19937 g(rd());
  shuffle(TrainSample.data.begin(), TrainSample.data.end(), g);
  int controlBegin = TrainSample.data.size() * div;

  ControlSample.data.clear();
  ControlSample.size = TrainSample.size;
  ControlSample.nfun = TrainSample.nfun;
  ControlSample.attr = TrainSample.attr;
  for (int i = controlBegin; i < TrainSample.data.size(); ++i)
  {
    ControlSample.data.push_back(TrainSample.data[i]);
  }

  TrainSample.data.erase(TrainSample.data.begin() + controlBegin, TrainSample.data.end());
}

double mse(Instance Inst, vector<double> pred)
{
  double res = 0;
  for (int i = 0; i < Inst.data.size(); ++i)
  {
    res += pow(Inst.data[i][Inst.nfun] - pred[i], 2);
  }
  return res / pred.size();
}