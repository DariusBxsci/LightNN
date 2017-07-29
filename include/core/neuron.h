#ifndef LIGHTNET_NEURON_H
#define LIGHTNET_NEURON_H

#include "weight.h"
#include "../functions/function.h"

class Neuron {

  protected:

    vector<Weight*> weights;
    vector<Weight*> nextWeights;
    double value;
    double delta;
    double bias;
    double upperWeightLimit, lowerWeightLimit;
    Function* function;
    bool isfunction;

  public:

    Neuron(double,double);
    Neuron(Function* function);
    void connect(Neuron* prev, Weight*); //connect neuron behind this one via a weight
    void forwardConnect(Weight*); //add a pointer to a weight in front of this neuron
    void process();
    void process(double);
    void backPropagate();
    void backPropagate(double);
    void gradientDescent(double, Optimizer*);
    void clearDelta();
    double getValue();
    double getDelta();
    ~Neuron();

};

#endif
