#ifndef LIGHTNET_MODULE_H
#define LIGHTNET_MODULE_H

#include <vector>
#include "../core/neuron.h"
using namespace std;

class Module {

  protected:

    vector<Neuron*> neurons;

  public:

    virtual void connect(Module* prev) =0;
    virtual void gradientDescent(double learningRate);
    virtual void backPropagate(vector<double> &delta);
    virtual void backPropagate();

    void process(vector<double> &input);
    void process();
    void setOptimizer(Optimizer*);
    void clearDelta();
    vector<Neuron*>& getNeurons();
    vector<double> getValue();
    int getSize();
    ~Module();

};

#endif
