#ifndef LIGHTNET_FUNCTION_MODULE_H
#define LIGHTNET_FUNCTION_MODULE_H

#include "module.h"
using namespace std;

class FunctionModule : public Module {

  private:

    Function* function;
    Module *next;

  public:

    FunctionModule(Function*);
    void connect(Module& next);
    void process(vector<double> input);
    void process();
    void backPropagate(vector<double> delta);
    void backPropagate();
    void gradientDescent(double learningRate);
    vector<double> getOutput();
    int getInputSize();
    int getOutputSize();

};

#endif
