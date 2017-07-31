#ifndef LIGHTNET_SHARED_WEIGH_H
#define LIGHTNET_SHARED_WEIGH_H

#include "weight.h"

class SharedWeight {

  private:

    vector<Weight*> weights;

  public:

    void addWeight(Weight*);
    void cloneWeights();

};

#endif
