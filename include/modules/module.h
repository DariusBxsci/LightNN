#ifndef LIGHTNET_MODULE_H
#define LIGHTNET_MODULE_H

#include <vector>
#include "core/neuron.h"
#include "functions/function.h"
using namespace std;

class Module {

  private:
  public:

    virtual void connect(Module* next) =0;
    virtual void process(vector<double> input) =0;
    virtual void process() =0;
    virtual void backPropagate(vector<double> delta) =0;
    virtual void backPropagate() =0;
    virtual void gradientDescent(double learningRate) =0;
    virtual vector<double> getOutput() =0;
    virtual int getInputSize() =0;
    virtual int getOutputSize() =0;

};

#endif
