#include "weight.h"
#include "neuron.h"
#include <unistd.h>


Weight::Weight() {
}

Weight::Weight(double lb, double ub) {
  init(lb,ub);
}

void Weight::init(double lb, double ub) {
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
  //cout << "Input: " << lastInput << "  Out: " << weight*lastInput << " from weight " << weight << endl;
  if (isnan(lastInput*weight)) {
    return 0;
  }
  return lastInput*weight;
}

void Weight::backPropagate(double d) {
  delta = weight*d; //pass back
  //cout << lastInput << endl;
  fullDelta += d*lastInput;
  /*if (fullDelta > 5) {
    cout << " with d " << d << endl;
    cout << "   and lastin " << lastInput << endl;
    usleep(10000000);
    //fullDelta = 0;
  }*/
  //if(weight == 1) cout << "delta " << fullDelta << endl;
  batch_size++;
}

void Weight::gradientDescent(double learningRate) {
  //if (isnan(fullDelta)) usleep(10000000);
  //cout << fullDelta/batch_size << endl;
  weight = optimizer->optimize(weight,fullDelta/batch_size,learningRate); //use full delta because of batch gradients
}

void Weight::setOptimizer(Optimizer* o) {
  optimizer = o;
}

int Weight::getBatchSize() {
  return batch_size;
}

double Weight::getDelta() {
  return delta;
}

double Weight::getFullDelta() {
  return fullDelta;
}

void Weight::setFullDelta(double fd) {
  fullDelta = fd;
}

double Weight::getWeight() {
  return weight;
}

void Weight::setWeight(double w) {
  weight = w;
}

void Weight::clearDelta() {
  delta = 0;
  fullDelta = 0;
  batch_size = 0;
}
