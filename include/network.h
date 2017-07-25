#ifndef LIGHTNET_NETWORK_H
#define LIGHTNET_NETWORK_H

#include <vector>
#include "modules/module.h"
#include "classifiers/classifier.h"
namespace ln {

struct Example {

  vector<double> input;
  vector<double> output;

};

class Network {

  private:

    vector<Module*> modules;
    Classifier* classifier;
    vector<double> logit; //last input to classifier
    vector<double> output; //final output of processing

  public:

    Network();
    void addModule(Module*);
    void linkModules();
    void addClassifier(Classifier*);
    vector<double> process(vector<double>);
    double getError(Example);
    double getError(vector<Example>);
    void backPropagate(Example);
    void gradientDescent(double learningRate, Optimizer*);
    double train(vector<Example>, Optimizer*, int, int, double);
    void clearDelta();
    ~Network();

};

}
#endif
