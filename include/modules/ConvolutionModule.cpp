#include "ConvolutionModule.h"

Kernel::Kernel(int sizex, int sizey) {
  units.resize(sizex);
  for(unsigned int x = 0; x < units.size(); x++) {
    units[x].resize(sizey);
  }
}

void Kernel::connect(FeatureMap*& input, FeatureMap*& output) {
  for(unsigned int ix = 0; ix < input->size(); ix++) {
    for(unsigned int iy = 0; iy < input->at(0).size(); iy++) {

      for(unsigned int kx = 0; kx < units.size(); kx++) {
        for(unsigned int ky = 0; ky < units[0].size(); ky++) {
            if (ix+kx < input->size() && iy+ky < input->at(0).size()) {
              //cout << "Connecting kernel " << kx << "," << ky << " to " << ix+kx << "," << iy+ky << endl;
              units[kx][ky].connect(input->at(ix+kx).at(iy+ky),output->at(ix).at(iy),new Weight());
            }
        }
      }

    }
  }
}

void Kernel::cloneWeightDeltas() {
  for(unsigned int x = 0; x < units.size(); x++) {
    for(unsigned int y = 0; y < units[0].size(); y++) {
      units[x][y].cloneWeightDeltas();
    }
  }
}

void Kernel::gradientDescent(double learningRate) {
  for(int kx = 0; kx < units.size(); kx++) {
    for(int ky = 0; ky < units[0].size(); ky++) {
      units[kx][ky].gradientDescent(learningRate);
    }
  }
}

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

ConvolutionModule::ConvolutionModule(int numFeatures, int kernelsPerFeatureMap, int ksizex, int ksizey, int sizex, int sizey, double lowerWeightLimit, double upperWeightLimit) {
  this->kernelsPerFeatureMap = kernelsPerFeatureMap;
  this->upperWeightLimit = upperWeightLimit;
  this->lowerWeightLimit = lowerWeightLimit;
  this->sizex = sizex;
  this->sizey = sizey;
  this->ksizex = ksizex;
  this->ksizey = ksizey;
  this->featureSize = sizex*sizey;
  this->numFeatures = numFeatures;
  for(int n = 0; n < featureSize*numFeatures*kernelsPerFeatureMap; n++) {
    neurons.push_back(new Neuron(lowerWeightLimit, upperWeightLimit));
  }
  //cout << "Matrifying Neurons " << neurons.size() << endl;
  for(int f = 0; f < numFeatures*kernelsPerFeatureMap; f++) {
    //cout << "Doing feature " << f << endl;
    vector<Neuron*> v(neurons.begin()+(f*sizex*sizey),neurons.begin()+((f+1)*sizex*sizey));
    featureMaps.push_back(matrifyNeurons(v,sizex,sizey));
    kernels.push_back(new Kernel(ksizex,ksizey));
  }
  //cout << neurons.size() << endl;
}

void ConvolutionModule::connect(Module* prev) {
  int z = 0;
  for(int f = 0; f < numFeatures; f++) {
    vector<Neuron*> v(prev->getNeurons().begin()+(f*sizex*sizey),prev->getNeurons().begin()+((f+1)*sizex*sizey));
    FeatureMap* fmap = matrifyNeurons(v,sizex,sizey);
    for(int k = 0; k < kernelsPerFeatureMap; k++) {
      kernels[z]->connect(fmap,featureMaps[z]);
      z++;
    }
  }
}

void ConvolutionModule::gradientDescent(double learningRate) {
  for(unsigned int x = 0; x < kernels.size(); x++) {
    kernels[x]->gradientDescent(learningRate);
  }
}

void ConvolutionModule::backPropagate(vector<double>& delta) {
  for(unsigned int x = 0; x < neurons.size(); x++) {
    neurons[x]->backPropagate(delta[x]);
  }
  /*for(unsigned int x = 0; x < kernels.size(); x++) {
    kernels[x]->cloneWeightDeltas();
  }*/
}

void ConvolutionModule::backPropagate() {
  for(unsigned int x = 0; x < neurons.size(); x++) {
    neurons[x]->backPropagate();
  }
  /*for(unsigned int x = 0; x < kernels.size(); x++) {
    kernels[x]->cloneWeightDeltas();
  }*/
}

ConvolutionModule::~ConvolutionModule() {
  for (auto it = kernels.begin(); it != kernels.end(); ++it){
      delete *it;
  }
  kernels.clear();
}
