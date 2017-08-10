#ifndef LIGHTNET_CONVOLUTION_MODULE_H
#define LIGHTNET_CONVOLUTION_MODULE_H

#include "module.h"
#include "../core/SharedWeight.h"
using namespace std;

using FeatureMap = vector<vector<Neuron*>>;

class Kernel {

  private:

    vector<vector<SharedWeight>> units; //represents the full kernel

  public:

    Kernel(int sizex, int sizey);
    void connect(FeatureMap*& input, FeatureMap*& output);
    void cloneWeightDeltas();
    void gradientDescent(double learningRate, Optimizer* optimizer);

};

class ConvolutionModule : public Module {

  private:

    int inputSize;
    double lowerWeightLimit,upperWeightLimit;
    int kernelsPerFeatureMap, sizex, sizey, featureSize,
     numFeatures, ksizex, ksizey;
    vector<Kernel*> kernels;
    vector<FeatureMap*> featureMaps;

  public:

    ConvolutionModule(int, int, int, int, int, int, double, double);
    void connect(Module* prev);
    void gradientDescent(double learningRate, Optimizer* optimizer);
    void backPropagate(vector<double>& delta);
    void backPropagate();
    ~ConvolutionModule();

};

#endif
