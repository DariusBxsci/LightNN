#ifndef LIGHTNET_OPTIMIZER_H
#define LIGHTNET_OPTIMIZER_H

class Optimizer {

  private:

  public:

    virtual double optimize(double weightVal, double weightDelta, double learningRate) =0;

};

#endif
