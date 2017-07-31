#ifndef LIGHTNET_DATA_IMAGE_H
#define LIGHTNET_DATA_IMAGE_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include "../network.h"
using namespace cv;
using namespace std;

namespace ln {

  vector<double> loadImage(string filename, int sx, int sy) {
      Mat img = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
      vector<double> image;
      for (int x = 0; x < sx; x++) {
          for (int y = 0; y < sy; y++) {
              //cout << (double)img.at<uchar>(x,y) << endl;
              image.push_back((double)img.at<uchar>(x,y)/255.0);
          }
      }
      return image;
  }

  void addToTrainingSet(ln::TrainingSet *tset, vector<double> out, string dirname, string ftype, int sx, int sy) {
      vector<cv::String> fn;
      glob(dirname + "/*." + ftype, fn, false);
      //cout << "HI" << endl;
      size_t c = fn.size(); //number of files in images folder
      for (size_t i=0; i < c; i++) {
        tset->add(loadImage(fn[i], sx, sy), out);
      }
  }

}

#endif
