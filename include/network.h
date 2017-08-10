#ifndef LIGHTNET_NETWORK_H
#define LIGHTNET_NETWORK_H

#include <vector>
#include <ctime>
#include <fstream>
#include <typeinfo>
#include <sstream>
#include "modules/module.h"
#include "classifiers/classifier.h"
namespace ln {

struct Example {

  vector<double> input;
  vector<double> output;

};

struct TrainingSet {

  vector<Example> examples;

  void add(vector<double> in, vector<double> out) {
    Example ex;
    ex.input = in;
    ex.output = out;
    examples.push_back(ex);
  }

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
    vector<double> process(vector<double>&);
    vector<double> q_process(vector<double>);
    void printOutput();
    double getError(Example&);
    double getError(TrainingSet&);
    double getClassError(Example&);
    double getClassError(TrainingSet&);
    void setOptimizer(Optimizer*);
    void backPropagate(Example&);
    void gradientDescent(double learningRate);
    double train(TrainingSet&, int, int, double);
    void clearDelta();
    void save(string directory);
    void load(string directory);
    ~Network();

};

}
#endif
