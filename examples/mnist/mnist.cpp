#include <iostream>
#include <LightNet/standard.h>
using namespace std;

int main() {

  ln::Network test_net;
  test_net.addModule(new FeedforwardModule(1));
  test_net.addModule(new FeedforwardModule(4,1,1));
  test_net.addModule(new FeedforwardModule(50,1,1));
  test_net.addModule(new FunctionModule(new ReluFunction()));
  test_net.addModule(new FunctionModule(new ReluFunction()));
  test_net.addModule(new FunctionModule(new ReluFunction()));
  test_net.addModule(new FunctionModule(new ReluFunction()));
  test_net.addModule(new FeedforwardModule(50,1,1));
  test_net.addModule(new FeedforwardModule(4,1,1));
  test_net.addModule(new FeedforwardModule(1,1,1));
  test_net.addClassifier(new StandardClassifier);
  test_net.linkModules();

  vector<ln::Example> trainingSet;
  for (int x = 0; x < 100; x++) {
    ln::Example ex;
    ex.input = {x/100.0};
    ex.output = {x*6/100.0};
    trainingSet.push_back(ex);
  }

  /*ln::Example ex;
  ex.input = {0.5};
  ex.output = {3};
  trainingSet.push_back(ex);*/

  cout << test_net.process({0.5})[0] << endl;
  //cout << "ERROR: " << test_net.getError(trainingSet) << endl;
  //test_net.train(trainingSet, new StandardOptimizer(), 50, 1, 0.005);
  //cout << "ERROR: " << test_net.getError(trainingSet) << endl;
  //cout << test_net.process({0.5})[0] << endl;

  cout << "done" << endl;
  return 0;
}
