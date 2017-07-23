#include "weight.h"
#include "neuron.h"

Weight::Weight(double lb, double ub) {
  delta = 0;
  fullDelta = 0;
  weight = (rand() % (int)(ub*100000*2) + lb*100000.0)/100000.0;
  //weight = 1;
  //generate weight in range to the hundred thousandths place
}

void Weight::connect(Neuron* p) {
  previous = p;
}

double Weight::process() {
  lastInput = previous->getValue();
  return lastInput*weight;
}

void Weight::backPropagate(double d) {
  delta = d*lastInput;
  fullDelta += delta;
}

void Weight::gradientDescent(double learningRate) {
  //cout << fullDelta << endl;
  weight -= learningRate*fullDelta; //use full delta because of batch gradients
}

double Weight::getDelta() {
  return delta;
}

void Weight::clearDelta() {
  delta = 0;
  fullDelta = 0;
}
