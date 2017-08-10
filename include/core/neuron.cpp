#include "neuron.h"

Neuron::Neuron(double lb, double ub) {
  isfunction = false;
  delta = 0;
  lowerWeightLimit = lb;
  upperWeightLimit = ub;
}

Neuron::Neuron(Function* function) {
  isfunction = true;
  this->function = function;
  delta = 0;
  lowerWeightLimit = 1;
  upperWeightLimit = 1;
}

void Neuron::connect(Neuron* prev, Weight *w) {
  w->init(lowerWeightLimit, upperWeightLimit);
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
    //cout << x << ": " << weights[x]->process() << endl;
    value += weights[x]->process();
  }
  this->input = value;
  if (isfunction == true) value = function->process(value);
  //cout << "     " << value << endl;
}

void Neuron::process(double input) {
  value = input;
  this->input = input;
  if (isfunction == true) value = function->process(value);
}

void Neuron::backPropagate() {
  delta = 0;
  for (unsigned int x = 0; x < nextWeights.size(); x++) {
    delta += nextWeights[x]->getDelta();
  }
  if(isfunction == true) delta = delta * function->derive(input);
  for (unsigned int x = 0; x < weights.size(); x++) {
    weights[x]->backPropagate(delta);
  }
}

void Neuron::backPropagate(double inDelta) {
  delta = inDelta;
  if(isfunction == true) delta = delta * function->derive(input);
  for (unsigned int x = 0; x < weights.size(); x++) {
    weights[x]->backPropagate(delta);
  }
}

void Neuron::setOptimizer(Optimizer* o) {
  for (unsigned int x = 0; x < weights.size(); x++) {
    weights[x]->setOptimizer(o);
  }
}

void Neuron::gradientDescent(double learningRate) {
  if (isfunction == false) {
    for (unsigned int x = 0; x < weights.size(); x++) {
      weights[x]->gradientDescent(learningRate);
    }
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
