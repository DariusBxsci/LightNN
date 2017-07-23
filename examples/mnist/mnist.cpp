#include <iostream>
#include <LightNet/standard.h>
using namespace std;

int main() {

  ln::Network test_net;
  test_net.addModule(new FeedforwardModule(1,-0.1,0.1));
  test_net.addModule(new FeedforwardModule(5,-0.1,0.1));
  test_net.addModule(new FeedforwardModule(1,-0.1,0.1));
  test_net.addClassifier(new StandardClassifier);
  test_net.linkModules();

  vector<ln::Example> trainingSet;
  for (int x = 0; x < 100; x++) {
    ln::Example ex;
    ex.input = {x/100.0};
    ex.output = {x*2/100.0};
    trainingSet.push_back(ex);
  }

  cout << "ERROR: " << test_net.getError(trainingSet) << endl;
  test_net.train(trainingSet, 30, 1, 0.5);
  cout << "ERROR: " << test_net.getError(trainingSet) << endl;
  cout << test_net.process({0.23})[0] << endl;

  return 0;
}
