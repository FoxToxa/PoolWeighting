#include <fstream>
#include <ctime>
#include <random>
#include <Windows.h>

#include "LinearApprox.h"
#include "GradBoost.h"

using namespace std;

void SplitInstance2(Instance& TrainSample, Instance& ControlSample, double div = 0.9);
double mse(Instance Inst, vector<double> pred);

double test();

int main()
{
  //test();
  //return 0;

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

  SplitInstance2(Inst, ContInst, 0.8);
  vector<double> pred;
  
  GradBoost GB(4, 2, 5, 0.1);
  GB.fit(Inst);
  pred = GB.predict(ContInst);
  cout << "GB no_weights = " << ContInst.RMSE(pred) << endl;


  DecisionTree DT(Inst, 4, 2);
  vector<double> w = DT.TuneWeights(ContInst, 5);
  Inst.SetWeights(w);
  GB.fit(Inst);
  pred = GB.predict(ContInst);
  cout << "GB with weights = " << ContInst.RMSE(pred) << endl;

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

double test()
{
  double x = 15;
  double y = 138;


  const int nrolls = 100;  // number of experiments
  
  default_random_engine generator;
  normal_distribution<double> distribution(0.0, x/5);
  double F;
  double eps;
  double delta =  0;
  for (int i = 0; i<nrolls; ++i) {
    eps = distribution(generator); 

    F = pow((2 * (x +  eps) + 10 - y), 2);

    delta += eps * F * pow(2, -8);
    //cout << eps << "  " << delta << endl;
    //Sleep(1000);
  }

  cout << delta << endl;
  return 0;
}
