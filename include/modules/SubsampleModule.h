#ifndef LIGHTNET_SUBSAMPLE_MODULE_H
#define LIGHTNET_SUBSAMPLE_MODULE_H

#include "module.h"
using namespace std;

using FeatureMap = vector<vector<Neuron*>>;

FeatureMap* matrifyNeurons(vector<Neuron*>& v, int sizex, int sizey) {
  FeatureMap *fmap = new FeatureMap();
  fmap->resize(sizex);
  int z = 0;
  for(unsigned int x = 0; x < fmap->size(); x++) {
    fmap->at(x).resize(sizey);
    for(unsigned int y = 0; y < fmap->at(x).size(); y++) {
      fmap->at(x).at(y) = v.at(z);
      z++;
    }
  }
  return fmap;
}

class SubsampleModule : public Module {

  private:

    int inputSize;
    double lowerWeightLimit,upperWeightLimit;
    int sizex, sizey, numFeatures, featureSize;
    vector<FeatureMap*> featureMaps;

  public:

    SubsampleModule(int numFeatures, int sizex, int sizey) {
      this->sizex = sizex;
      this->sizey = sizey;
      this->featureSize = sizex*sizey;
      this->numFeatures = numFeatures;

      for(int n = 0; n < featureSize*numFeatures/pow(2,2); n++) {
        neurons.push_back(new Neuron(0.25, 0.25));
      }
      for(int f = 0; f < numFeatures; f++) {
        vector<Neuron*> v(neurons.begin()+(f*floor(sizex/2.0)*floor(sizey/2.0)),neurons.begin()+((f+1)*floor(sizex/2.0)*floor(sizey/2.0)));
        featureMaps.push_back(matrifyNeurons(v,floor(sizex/2.0),floor(sizey/2.0)));
      }
    }

    void connect(Module* prev) {
      int z = 0;
      for(int f = 0; f < numFeatures; f++) {
        vector<Neuron*> v(prev->getNeurons().begin()+(f*sizex*sizey),prev->getNeurons().begin()+((f+1)*sizex*sizey));
        FeatureMap* fmap = matrifyNeurons(v,sizex,sizey);

        for(int fx = 0; fx < featureMaps[f]->size(); fx++) {
          for(int fy = 0; fy < featureMaps[f]->at(0).size(); fy++) {

            for(int ix = fx*2; ix < fx*2 + 2; ix++) {
              for(int iy = fy*2; iy < fy*2 + 2; iy++) {

                //cout << f << " connecting " << fx << "," << fy << " to " << ix << "," << iy << endl;
                featureMaps[f]->at(fx).at(fy)->connect(fmap->at(ix).at(iy),new Weight());

              }
            }

          }
        }

      }

      //cout << "size " << neurons.size() << endl;
    }

    void gradientDescent(double learningRate) {}


};

#endif
