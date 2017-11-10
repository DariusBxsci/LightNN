#ifndef LIGHTNET_SOFTMAX_CLASSIFIER_H
#define LIGHTNET_SOFTMAX_CLASSIFIER_H

#include "classifier.h"
#include <math.h>

class SoftmaxClassifier : public Classifier {

  private:

  public:

    vector<double> classify(vector<double> input) {
      double sum = 0;
      for (unsigned int x = 0; x < input.size(); x++) {
        //cout << input[x] << endl;
        sum += exp(input[x]);
      }
      for (unsigned int x = 0; x < input.size(); x++) {
        input[x] = exp(input[x]) / sum;
      }
      return input;
    }

    double getError(vector<double> input, vector<double> desiredOutput) {
      vector<double> output = classify(input);
      double err = 0;
      for (unsigned int x = 0; x < input.size(); x++) {
        err += desiredOutput[x] * log(output[x]);
      }
      err /= input.size();
      return -err;
    }

    vector<double> getDelta(vector<double> input, vector<double> desiredOutput) {
      vector<double> delta;
      vector<double> output = classify(input);
      delta.resize(input.size());
      for(unsigned int x = 0; x < delta.size(); x++) {
        delta[x] = output[x] - desiredOutput[x];
      }
      return delta;
    }

};

#endif
