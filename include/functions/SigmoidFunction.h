#ifndef LIGHTNET_SIGMOID_FUNCTION_H
#define LIGHTNET_SIGMOID_FUNCTION_H

#include "function.h"

class SigmoidFunction : public Function {

  private:
  public:

    double process(double in) {
      return pow(2.718,in)/(1+pow(2.718,in));
    }
    double derive(double in) {
      return (pow(2.718,in)/(1+pow(2.718,in))) * (1/(1+pow(2.718,in)));
    }

};

#endif
