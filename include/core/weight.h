#ifndef LIGHTNET_WEIGHT_H
#define LIGHTNET_WEIGHT_H

#include <cmath>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <deque>
#include <iostream>
#include "../optimizers/optimizer.h"
using namespace std;

class Neuron;

class Weight {

  protected:

    Neuron* previous;
    double weight;
    double delta;
    double fullDelta; //stores sum of many deltas (for batch gradientDescent)
    int batch_size;
    double lastInput;
    Optimizer* optimizer;

  public:

    Weight(); //bounds for randomly generated weight values
    Weight(double, double); //bounds for randomly generated weight values
    virtual void init(double, double); //bounds for randomly generated weight values
    virtual void connect(Neuron*);
    virtual double process(); //process value from previous neuron
    virtual void backPropagate(double); //back propagate delta from next neuron
    virtual void gradientDescent(double); //perform gradient descent on weight based on delta
    void setOptimizer(Optimizer*);
    int getBatchSize();
    double getDelta();
    double getFullDelta();
    void setFullDelta(double);
    void setDelta(double);
    virtual double getWeight();
    virtual void setWeight(double);
    void clearDelta(); //set delta to 0;

};

#endif
