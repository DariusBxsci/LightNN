#include "neuron.h"

Neuron::Neuron(double lb, double ub) {
  delta = 0;
  lowerWeightLimit = lb;
  upperWeightLimit = ub;
}

void Neuron::connect(Neuron* prev) {
  Weight *w = new Weight(lowerWeightLimit, upperWeightLimit);
  w->connect(prev);
  prev->forwardConnect(w);
  weights.push_back(w);
}

void Neuron::forwardConnect(Weight* nextWeight) {
  nextWeights.push_back(nextWeight);
}

void Neuron::process() {
  value = 0;
  for (unsigned int x = 0; x < weights.size(); x++) {
    value += weights[x]->process();
  }
}

void Neuron::process(double input) {
  value = input;
}

void Neuron::backPropagate() {
  delta = 0;
  for (unsigned int x = 0; x < nextWeights.size(); x++) {
    delta += nextWeights[x]->getDelta();
  }
  for (unsigned int x = 0; x < weights.size(); x++) {
    weights[x]->backPropagate(delta);
  }
}

void Neuron::backPropagate(double inDelta) {
  delta = inDelta;
  for (unsigned int x = 0; x < weights.size(); x++) {
    weights[x]->backPropagate(delta);
  }
}

void Neuron::gradientDescent(double learningRate) {
  for (unsigned int x = 0; x < weights.size(); x++) {
    weights[x]->gradientDescent(learningRate);
  }
}

void Neuron::clearDelta() {
  for (unsigned int x = 0; x < weights.size(); x++) {
    weights[x]->clearDelta();
  }
}

double Neuron::getValue() {
  return value;
}

double Neuron::getDelta() {
  return delta;
}

Neuron::~Neuron() {
  for (auto it = weights.begin(); it != weights.end(); ++it){
      delete *it;
  }
  weights.clear();
}
