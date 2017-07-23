#ifndef LIGHTNET_NETWORK_H
#define LIGHTNET_NETWORK_H
namespace ln {

#include <vector>
#include "modules/module.h"

struct Example {

  vector<double> input;
  vector<double> output;

};

class Network {

  private:

    vector<Module*> modules;

  public:

    void addLayer(Module*);
    void linkLayers();
    void process(vector<double>);
    void backPropagate(Example);
    void gradientDescent(double learningRate);
    ~Network();

};

}
#endif
