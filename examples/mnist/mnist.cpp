#include <iostream>
#include <LightNet/standard.h>
using namespace std;

int main() {

  ln::Network test_net;
  test_net.addModule(new FeedforwardModule(2));
  test_net.addModule(new FeedforwardModule(1,-1,2));
  test_net.addModule(new FunctionModule(new ReluFunction()));
  test_net.addModule(new FeedforwardModule(15,-1,2));
  test_net.addModule(new FunctionModule(new ReluFunction()));
  test_net.addModule(new FeedforwardModule(10,-1,2));
  test_net.addModule(new FunctionModule(new ReluFunction()));
  test_net.addModule(new FeedforwardModule(1,-1,2));
  test_net.addClassifier(new StandardClassifier);
  test_net.linkModules();

  ln::TrainingSet trainingSet;
  trainingSet.add({0,0},{0});
  trainingSet.add({1,0},{1});
  trainingSet.add({0,1},{1});
  trainingSet.add({1,1},{0});

  cout << "ERROR: " << test_net.getError(trainingSet) << endl;
  test_net.train(trainingSet, new StandardOptimizer(), 5000, 1, 0.002);
  cout << "ERROR: " << test_net.getError(trainingSet) << endl;
  cout << test_net.process({0,0})[0] << endl;
  cout << test_net.process({1,0})[0] << endl;
  cout << test_net.process({0,1})[0] << endl;
  cout << test_net.process({1,1})[0] << endl;

  cout << "done" << endl;
  return 0;
}
