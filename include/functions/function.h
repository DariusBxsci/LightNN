#ifndef LIGHTNET_FUNCTION_H
#ifndef LIGHTNET_FUNCTION_H

class Function {

  private:
  public:

    virtual double process(double) =0;
    virtual double derive(double) =0;

};

#endif
