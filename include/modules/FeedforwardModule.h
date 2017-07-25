#ifndef LIGHTNET_FEEDFORWARD_MODULE_H
#define LIGHTNET_FEEDFORWARD_MODULE_H

#include "module.h"
using namespace std;

class FeedforwardModule : public Module {

  private:

    int inputSize;
    double lowerWeightLimit,upperWeightLimit;

  public:

    FeedforwardModule(int,double,double);
    FeedforwardModule(int);
    void connect(Module* prev);

};

#endif
