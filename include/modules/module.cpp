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

void Module::gradientDescent(double learningRate, Optimizer* optimizer) {
  for(unsigned int x = 0; x < neurons.size(); x++) {
    neurons[x]->gradientDescent(learningRate,optimizer);
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

Module::~Module() {
  for (auto it = neurons.begin(); it != neurons.end(); ++it){
      delete *it;
  }
  neurons.clear();
}
