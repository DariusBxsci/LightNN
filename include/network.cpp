#include "network.h"
using namespace ln;

int classify(vector<double> v) {
  int highest = 0;
  for(unsigned int x = 0; x < v.size(); x++) {
    if (v[x] > v[highest]) highest = x;
  }
  return highest;
}

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

vector<double> Network::q_process(vector<double> input) {
  modules[0]->process(input);
  for (unsigned int x = 1; x < modules.size(); x++) {
    modules[x]->process();
  }
  logit = modules[modules.size()-1]->getValue();
  output = classifier->classify(logit);
  return output;
}

vector<double> Network::process(vector<double>& input) {
  modules[0]->process(input);
  for (unsigned int x = 1; x < modules.size(); x++) {
    //cout << "mod " << x << endl;
    modules[x]->process();
  }
  logit = modules[modules.size()-1]->getValue();
  output = classifier->classify(logit);
  return output;
}

void Network::printOutput() {
  for (unsigned int x = 0; x < output.size(); x++) {
    cout << "OUTPUT NEURON " << x << ": " << output[x] << endl;
  }
}

double Network::getError(Example& ex) {
  process(ex.input);
  double err = classifier->getError(logit,ex.output);
  if (isnan(err)) return 0;
  return err;
}

double Network::getError(TrainingSet& ex) {
  double err = 0;
  for(unsigned int x = 0; x < ex.examples.size(); x++) {
    err += getError(ex.examples[x]);
  }
  return err/ex.examples.size();
}

double Network::getClassError(Example& ex) {
  process(ex.input);
  //cout << "out " << classify(output) << " class " << classify(ex.output) << endl;
  if (classify(output) == classify(ex.output)) return 0;
  else return 1;
}

double Network::getClassError(TrainingSet& ex) {
  double err = 0;
  for(unsigned int x = 0; x < ex.examples.size(); x++) {
    err += getClassError(ex.examples[x]);
  }
  return err/ex.examples.size();
}

void Network::setOptimizer(Optimizer *o) {
  for(unsigned int x = 0; x < modules.size(); x++) {
    modules[x]->setOptimizer(o);
  }
}

void Network::backPropagate(Example &ex) {
  process(ex.input);
  vector<double> delta = classifier->getDelta(logit,ex.output);
  modules[modules.size()-1]->backPropagate(delta);
  for (int x = modules.size()-2; x >= 0; x--) {
    modules[x]->backPropagate();
  }
}

void Network::gradientDescent(double learningRate) {
  for (unsigned int x = 0; x < modules.size(); x++) {
    modules[x]->gradientDescent(learningRate);
  }
}

double Network::train(TrainingSet &trainingset, int iterations, int batch_size, double learningRate) {
  std::clock_t start;
  start = std::clock();
  for (int i = 0; i < iterations; i++) {
    for(int b = 0; b < batch_size; b++) {
      int ex = rand()%trainingset.examples.size();
      //cout << "ex " << i << endl;
      backPropagate(trainingset.examples[ex]);
    }
    gradientDescent(learningRate);
    clearDelta();
  }
  return ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
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
