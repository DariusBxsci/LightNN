#ifndef LIGHTNET_NEURON_H
#define LIGHTNET_NEURON_H

#include "weight.h"

class Neuron {

  private:

    vector<Weight*> weights;
    vector<Weight*> nextWeights;
    double value;
    double delta;
    double upperWeightLimit, lowerWeightLimit;

  public:

    Neuron(double,double); //set weight limits for weights
    void connect(Neuron* prev); //connect neuron behind this one via a weight
    void forwardConnect(Weight*); //add a pointer to a weight in front of this neuron
    void process();
    void process(double);
    void backPropagate();
    void backPropagate(double);
    void gradientDescent(double);
    void clearDelta();
    double getValue();
    double getDelta();
    ~Neuron();

};

#endif
