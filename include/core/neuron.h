#ifndef LIGHTNET_NEURON_H
#define LIGHTNET_NEURON_H

#include "neuron.h"

class Neuron {

  private:

    vector<Weight*> weights;
    double value;
    double delta;

  public:

    void connect(Neuron* prev);
    void process();
    void process(double);
    void backPropagate();
    void backPropagate(double);
    void gradientDescent(double);
    double getValue();
    double getDelta();
    ~Neuron();

};

#endif
