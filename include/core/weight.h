#ifndef LIGHTNET_WEIGHT_H
#define LIGHTNET_WEIGHT_H

#include <cmath>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <deque>
#include <iostream>
using namespace std;

class Neuron;

class Weight {

  private:

    Neuron* previous;
    double weight;
    double delta;
    double fullDelta; //stores sum of many deltas (for batch gradientDescent)
    double lastInput;

  public:

    Weight(double, double); //bounds for randomly generated weight values
    void connect(Neuron*);
    double process(); //process value from previous neuron
    void backPropagate(double); //back propagate delta from next neuron
    void gradientDescent(double); //perform gradient descent on weight based on delta
    double getDelta();
    void clearDelta(); //set delta to 0;

};

#endif
