#include "FunctionModule.h"

FunctionModule::FunctionModule(Function* function) {
  this->function = function;
}

void FunctionModule::connect(Module* prev) {
  for(unsigned int x = 0; x < prev->getNeurons().size(); x++) {
    neurons.push_back(new Neuron(function));
  }
  for(unsigned int x = 0; x < prev->getNeurons().size(); x++) {
    neurons[x]->connect(prev->getNeurons()[x]);
  }
}

FunctionModule::~FunctionModule() {
  delete function;
  for (auto it = neurons.begin(); it != neurons.end(); ++it){
      delete *it;
  }
  neurons.clear();
}
