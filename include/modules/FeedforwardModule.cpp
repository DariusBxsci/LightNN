#include "FeedforwardModule.h"

FeedforwardModule::FeedforwardModule(int size, double lb, double ub) {
  inputSize = size;
  lowerWeightLimit = lb;
  upperWeightLimit = ub;
  for(int x = 0; x < size; x++) {
    neurons.push_back(new Neuron(lowerWeightLimit,upperWeightLimit));
  }
}

FeedforwardModule::FeedforwardModule(int size) {
  inputSize = size;
  lowerWeightLimit = 0;
  upperWeightLimit = 0;
  for(int x = 0; x < size; x++) {
    neurons.push_back(new Neuron(lowerWeightLimit,upperWeightLimit));
  }
}

void FeedforwardModule::connect(Module* prev) {
  for(unsigned int x = 0; x < neurons.size(); x++) {
    for(unsigned int y = 0; y < prev->getNeurons().size(); y++) {
      neurons[x]->connect(prev->getNeurons()[y]);
    }
  }
}
