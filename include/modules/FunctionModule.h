#ifndef LIGHTNET_FUNCTION_MODULE_H
#define LIGHTNET_FUNCTION_MODULE_H

#include "module.h"
#include "../functions/function.h"
using namespace std;

class FunctionModule : public Module {

  private:

    Function* function;

  public:

    FunctionModule(Function*);
    void connect(Module* prev);
    ~FunctionModule();

};

#endif
