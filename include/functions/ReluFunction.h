#ifndef LIGHTNET_RELU_FUNCTION_H
#ifndef LIGHTNET_RELU_FUNCTION_H

#include "function.h"

class Relu : public Function {

  private:
  public:

    double process(double in) {
      if (in < 0) return 0;
      return in;
    }
    double derive(double in) {
      if (in > 0) return 1;
      return 0;      
    }

};

#endif
