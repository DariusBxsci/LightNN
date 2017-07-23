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

    void process(vector<double> input);
    void process();
    void backPropagate(vector<double> delta);
    void backPropagate();
    void gradientDescent(double learningRate);
    void clearDelta();
    vector<Neuron*>& getNeurons();
    vector<double> getValue();
    int getSize();
    ~Module();

};

#endif
