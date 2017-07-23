#ifndef LIGHTNET_CLASSIFIER_H
#define LIGHTNET_CLASSIFIER_H

#include <vector>
#include <math.h>
using namespace std;

class Classifier {

  private:

  public:

    virtual vector<double> classify(vector<double> input) =0;
    //return output by processing input;
    virtual double getError(vector<double> input, vector<double> desiredOutput) =0;
    //get error by processing input and comparing to desired output by using
      //an appropriate error function
    virtual vector<double> getDelta(vector<double> input, vector<double> desiredOutput) =0;
    //get a vector of delta values by backPropagation through the classifier

};

#endif
