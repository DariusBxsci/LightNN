#ifndef LIGHTNET_SHARED_WEIGH_H
#define LIGHTNET_SHARED_WEIGH_H

#include "weight.h"
#include "neuron.h"

class SharedWeight {

  private:

    vector<Weight*> weights; //should all be the same

  public:

    void addWeight(Weight* w) {
      weights.push_back(w);
    }

    void connect(Neuron*& input, Neuron*& output, Weight* w) {
      addWeight(w);
      output->connect(input, w);
      cloneWeights();
    }

    void cloneWeights() { //make all weight vals the same
        double cw = weights[0]->getWeight();
        for(unsigned int x = 0; x < weights.size(); x++) {
          weights[x]->setWeight(cw);
        }
    }

    void cloneWeightDeltas() { //make all deltas the same
        double sumDelt = 0;
        //cout << "SIZE " << weights.size() << endl;
        for(unsigned int x = 0; x < weights.size(); x++) {
          sumDelt += weights[x]->getFullDelta();
        }
        //cout << sumDelt << endl;
        for(unsigned int x = 0; x < weights.size(); x++) {
          weights[x]->setFullDelta(sumDelt/weights.size());
          //cout << "w " << weights[x]->getFullDelta() << endl;
        }
    }

    void gradientDescent(double learningRate) {
      cloneWeightDeltas();
      //cout << "size " << weights.size() << endl;
      for (unsigned int x = 0; x < weights.size(); x++) {
        //cout << "weight " << x << ": " << weights[x]->getWeight() << endl;
        weights[x]->gradientDescent(learningRate);
      }
    }

};

#endif
