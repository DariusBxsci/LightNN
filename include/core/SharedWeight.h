#ifndef LIGHTNET_SHARED_WEIGH_H
#define LIGHTNET_SHARED_WEIGH_H

#include "weight.h"

class SharedWeight {

  private:

    vector<Weight*> weights; //should all be the same

  public:

    void addWeight(Weight* w) {
      weights.push_back(w);
    }

    void cloneWeights() { //make all weight vals and deltas the same
        double cw = weights[0]->getWeight();
        for(unsigned int x; x < weights.size(); x++) {
          weights[x]->setWeight(cw);
        }
    }

    void cloneWeightDeltas() { //make all weight vals and deltas the same
        double sumDelt = 0;
        for(unsigned int x; x < weights.size(); x++) {
          sumDelt += weights[x]->getFullDelta() / weights[x]->getBatchSize();
        }
        for(unsigned int x; x < weights.size(); x++) {
          weights[x]->setFullDelta(sumDelt);
        }
    }

};

#endif
