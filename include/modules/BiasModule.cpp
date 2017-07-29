#include "BiasModule.h"

BiasModule::BiasModule(int size, double lb, double ub) {
  inputSize = size;
  lowerWeightLimit = lb;
  upperWeightLimit = ub;
  for(int x = 0; x < size; x++) {
    neurons.push_back(new Neuron(lowerWeightLimit,upperWeightLimit));
  }
}

void BiasModule::connect(Module* prev) {
  for(unsigned int x = 0; x < prev->getNeurons().size(); x++) {
    neurons[x]->connect(prev->getNeurons()[x], new BiasWeight());
  }
}
