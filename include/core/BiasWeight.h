#ifndef LIGHTNET_BIAS_WEIGHT_H
#define LIGHTNET_BIAS_WEIGHT_H

#include "weight.h"

class BiasWeight : public Weight {

  private:

    double bias;

  public:

    void init(double lb, double ub) {
      delta = 0;
      fullDelta = 0;
      batch_size = 0;
      if (abs(lb-ub) == 0) bias = lb;
      else bias = (rand() % (int)(abs(lb-ub)*100000) + lb*100000.0)/100000.0;
    }

    BiasWeight() {
    }

    BiasWeight(double lb, double ub) {
      init(lb,ub);
    }

    double process() {
      lastInput = previous->getValue();
      if (isnan(lastInput+bias)) {
        return 0;
      }
      return lastInput+bias;
    }

    void backPropagate(double d) {
      delta = d; //pass back
      fullDelta += d;
      batch_size++;
    }

    void gradientDescent(double learningRate) {
      bias = optimizer->optimize(bias,fullDelta/batch_size,learningRate);
      //cout << bias << endl;
    }

    double getWeight() {
      return bias;
    }

    void setWeight(double w) {
      bias = w;
    }

};

#endif
