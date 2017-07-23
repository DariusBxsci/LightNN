#ifndef LIGHTNET_STANDARD_CLASSIFIER_H
#define LIGHTNET_STANDARD_CLASSIFIER_H

#include "classifier.h"

class StandardClassifier : public Classifier {

  private:

  public:

    vector<double> classify(vector<double> input) {
      return input;
    }

    double getError(vector<double> input, vector<double> desiredOutput) {
      double err = 0;
      for (unsigned int x = 0; x < input.size(); x++) {
        err += pow(desiredOutput[x]-input[x],2)/2;
      }
      err /= input.size();
      return err;
    }

    vector<double> getDelta(vector<double> input, vector<double> desiredOutput) {
      vector<double> delta;
      delta.resize(input.size());
      for(unsigned int x = 0; x < delta.size(); x++) {
        delta[x] = input[x] - desiredOutput[x];
      }
      return delta;
    }

};

#endif
