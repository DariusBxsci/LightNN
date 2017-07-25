#include "network.h"
using namespace ln;

Network::Network() {
  srand(time(0));
}

void Network::addModule(Module* mod) {
  modules.push_back(mod);
}

void Network::linkModules() {
  for(unsigned int x = 1; x < modules.size(); x++) {
    modules[x]->connect(modules[x-1]);
  }
}

void Network::addClassifier(Classifier* c) {
  classifier = c;
}

vector<double> Network::process(vector<double> input) {
  modules[0]->process(input);
  for (unsigned int x = 1; x < modules.size(); x++) {
    modules[x]->process();
  }
  logit = modules[modules.size()-1]->getValue();
  output = classifier->classify(logit);
  return output;
}

double Network::getError(Example ex) {
  process(ex.input);
  return classifier->getError(logit,ex.output);
}

double Network::getError(TrainingSet ex) {
  double err = 0;
  for(unsigned int x = 0; x < ex.examples.size(); x++) {
    err += getError(ex.examples[x]);
  }
  return err/ex.examples.size();
}

void Network::backPropagate(Example ex) {
  process(ex.input);
  vector<double> delta = classifier->getDelta(logit,ex.output);
  modules[modules.size()-1]->backPropagate(delta);
  for (int x = modules.size()-2; x >= 0; x--) {
    modules[x]->backPropagate();
  }
}

void Network::gradientDescent(double learningRate, Optimizer* optimizer) {
  for (unsigned int x = 0; x < modules.size(); x++) {
    modules[x]->gradientDescent(learningRate, optimizer);
  }
}

double Network::train(TrainingSet trainingset, Optimizer* optimizer, int iterations, int batch_size, double learningRate) {
  for (int i = 0; i < iterations; i++) {
    for(int b = 0; b < batch_size; b++) {
      backPropagate(trainingset.examples[rand()%trainingset.examples.size()]);
    }
    gradientDescent(learningRate,optimizer);
    clearDelta();
  }
}

void Network::clearDelta() {
  for(unsigned int x = 0; x < modules.size(); x++) {
    modules[x]->clearDelta();
  }
}

Network::~Network() {
  for (auto it = modules.begin(); it != modules.end(); ++it){
      delete *it;
  }
  modules.clear();
}
