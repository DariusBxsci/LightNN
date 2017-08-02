#ifndef LIGHTNET_CONVOLUTION_MODULE_H
#define LIGHTNET_CONVOLUTION_MODULE_H

#include "module.h"
#include "../core/SharedWeight.h"
using namespace std;

class Kernel {

  private:

    vector<vector<SharedWeight>> units; //represents the full kernel

  public:

    void connect(vector<Neuron*> input, vector<Neuron*> output, double upperWeightLimit, double lowerWeightLimit);
    void gradientDescent(double learningRate, Optimizer* optimizer);

};

class ConvolutionModule : public Module {

  private:

    int inputSize;
    double lowerWeightLimit,upperWeightLimit;

  public:

    ConvolutionModule(int images, int sizex, int sizey, double,double);
    void connect(Module* prev);

};

#endif
