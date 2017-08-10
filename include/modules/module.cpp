#include "module.h"

void Module::process(vector<double>& input) {
  for(unsigned int x = 0; x < neurons.size(); x++) {
    neurons[x]->process(input[x]);
  }
}

void Module::process() {
  for(unsigned int x = 0; x < neurons.size(); x++) {
    neurons[x]->process();
  }
  //cout << " " << neurons.size() << endl;
}

void Module::backPropagate(vector<double>& delta) {
  for(unsigned int x = 0; x < neurons.size(); x++) {
    neurons[x]->backPropagate(delta[x]);
  }
}

void Module::backPropagate() {
  for(unsigned int x = 0; x < neurons.size(); x++) {
    neurons[x]->backPropagate();
  }
}

void Module::gradientDescent(double learningRate) {
  for(unsigned int x = 0; x < neurons.size(); x++) {
    neurons[x]->gradientDescent(learningRate);
  }
}

void Module::setOptimizer(Optimizer* o) {
  for(unsigned int x = 0; x < neurons.size(); x++) {
    neurons[x]->setOptimizer(o);
  }
}

void Module::clearDelta() {
  for(unsigned int x = 0; x < neurons.size(); x++) {
    neurons[x]->clearDelta();
  }
}

vector<Neuron*>& Module::getNeurons() {
  return neurons;
}

vector<double> Module::getValue() {
  vector<double> o;
  for(unsigned int x = 0; x < neurons.size(); x++) {
    o.push_back(neurons[x]->getValue());
  }
  return o;
}

int Module::getSize() {
  return neurons.size();
}

vector<double> Module::getWeightVector() {
  vector<double> wv;
  for(unsigned int x = 0; x < neurons.size(); x++) {
    vector<double> v = neurons[x]->getWeightVector();
    wv.insert(wv.end(), v.begin(), v.end());
  }
  return wv;
}

void Module::load(vector<double> wv) {
  int b = 0;
  for(unsigned int x = 0; x < neurons.size(); x++) {
    vector<double> vals(wv.begin() + b,wv.begin() + b+neurons[x]->getNumWeights());
    neurons[x]->load(vals);
    b += neurons[x]->getNumWeights();
  }
}

Module::~Module() {
  for (auto it = neurons.begin(); it != neurons.end(); ++it){
      delete *it;
  }
  neurons.clear();
}
