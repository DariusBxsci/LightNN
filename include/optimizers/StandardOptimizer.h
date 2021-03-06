#ifndef LIGHTNET_STANDARD_OPTIMIZER_H
#define LIGHTNET_STANDARD_OPTIMIZER_H

#include "optimizer.h"

class StandardOptimizer : public Optimizer {

  private:

    double gradClip;

  public:

    StandardOptimizer() {
      gradClip = 0;
    }
    StandardOptimizer(double clip) {
      gradClip = clip;
    }
    double optimize(double weightVal, double weightDelta, double learningRate) {
      if (gradClip == 0) {
        return weightVal - (weightDelta*learningRate);
      }
      else {
        //cout << "clipping" << endl;
        if (weightDelta < -gradClip) {
          //cout << "opt " << weightVal - (-gradClip*learningRate) << endl;
          return weightVal - (-gradClip*learningRate);
        };
        if (weightDelta > gradClip) {
          //cout << "opt " << weightVal - (gradClip*learningRate) << endl;
          return weightVal - (gradClip*learningRate);
        };
      }
    }

};

#endif
