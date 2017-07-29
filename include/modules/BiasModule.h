#ifndef LIGHTNET_BIAS_MODULE_H
#define LIGHTNET_BIAS_MODULE_H

#include "module.h"
#include "../core/BiasWeight.h"
using namespace std;

class BiasModule : public Module {

  private:

    int inputSize;
    double lowerWeightLimit,upperWeightLimit;

  public:

    BiasModule(int,double,double);
    BiasModule(int);
    void connect(Module* prev);

};

#endif
