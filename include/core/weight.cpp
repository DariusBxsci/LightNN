#include "weight.h"
#include "neuron.h"

Weight::Weight(double lb, double ub) {
  delta = 0;
  fullDelta = 0;
  batch_size = 0;
  if (abs(lb-ub) == 0) weight = lb;
  else weight = (rand() % (int)(abs(lb-ub)*100000) + lb*100000.0)/100000.0;
  //cout << weight << endl;
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
  delta = weight*d; //pass back
  //cout << lastInput << endl;
  fullDelta += d*lastInput;
  batch_size++;
}

void Weight::gradientDescent(double learningRate, Optimizer* optimizer) {
  //cout << fullDelta << endl;
  weight = optimizer->optimize(weight,fullDelta/batch_size,learningRate); //use full delta because of batch gradients
}

double Weight::getDelta() {
  return delta;
}

void Weight::clearDelta() {
  delta = 0;
  fullDelta = 0;
  batch_size = 0;
}
