#include <iostream>
#include <LightNet/standard.h>
#include <LightNet/data.h>
using namespace std;

int main() {

  ln::Network test_net;
  test_net.addModule(new FeedforwardModule(28*28));
  test_net.addModule(new FeedforwardModule(14*14,0,0.01));
  test_net.addModule(new BiasModule(14*14,0,0.01));
  test_net.addModule(new FunctionModule(new ReluFunction()));
  test_net.addModule(new FeedforwardModule(7*7,0,0.01));
  test_net.addModule(new BiasModule(7*7,0,0.01));
  test_net.addModule(new FunctionModule(new ReluFunction()));
  test_net.addModule(new FeedforwardModule(10,-0.1,0.1));
  test_net.addModule(new BiasModule(10,0,0.01));
  test_net.addModule(new FunctionModule(new ReluFunction()));
  test_net.addClassifier(new SoftmaxClassifier());
  test_net.linkModules();

  ln::TrainingSet trainingSet;
  ln::addToTrainingSet(&trainingSet, {1,0,0,0,0,0,0,0,0,0}, "data/mnist/training/0/", "png", 28,28);
  ln::addToTrainingSet(&trainingSet, {0,1,0,0,0,0,0,0,0,0}, "data/mnist/training/1/", "png", 28,28);
  ln::addToTrainingSet(&trainingSet, {0,0,1,0,0,0,0,0,0,0}, "data/mnist/training/2/", "png", 28,28);
  ln::addToTrainingSet(&trainingSet, {0,0,0,1,0,0,0,0,0,0}, "data/mnist/training/3/", "png", 28,28);
  ln::addToTrainingSet(&trainingSet, {0,0,0,0,1,0,0,0,0,0}, "data/mnist/training/4/", "png", 28,28);
  ln::addToTrainingSet(&trainingSet, {0,0,0,0,0,1,0,0,0,0}, "data/mnist/training/5/", "png", 28,28);
  ln::addToTrainingSet(&trainingSet, {0,0,0,0,0,0,1,0,0,0}, "data/mnist/training/6/", "png", 28,28);
  ln::addToTrainingSet(&trainingSet, {0,0,0,0,0,0,0,1,0,0}, "data/mnist/training/7/", "png", 28,28);
  ln::addToTrainingSet(&trainingSet, {0,0,0,0,0,0,0,0,1,0}, "data/mnist/training/8/", "png", 28,28);
  ln::addToTrainingSet(&trainingSet, {0,0,0,0,0,0,0,0,0,1}, "data/mnist/training/9/", "png", 28,28);

  ln::TrainingSet testingSet;
  ln::addToTrainingSet(&testingSet, {1,0,0,0,0,0,0,0,0,0}, "data/mnist/testing/0/", "png", 28,28);
  ln::addToTrainingSet(&testingSet, {0,1,0,0,0,0,0,0,0,0}, "data/mnist/testing/1/", "png", 28,28);
  ln::addToTrainingSet(&testingSet, {0,0,1,0,0,0,0,0,0,0}, "data/mnist/testing/2/", "png", 28,28);
  ln::addToTrainingSet(&testingSet, {0,0,0,1,0,0,0,0,0,0}, "data/mnist/testing/3/", "png", 28,28);
  ln::addToTrainingSet(&testingSet, {0,0,0,0,1,0,0,0,0,0}, "data/mnist/testing/4/", "png", 28,28);
  ln::addToTrainingSet(&testingSet, {0,0,0,0,0,1,0,0,0,0}, "data/mnist/testing/5/", "png", 28,28);
  ln::addToTrainingSet(&testingSet, {0,0,0,0,0,0,1,0,0,0}, "data/mnist/testing/6/", "png", 28,28);
  ln::addToTrainingSet(&testingSet, {0,0,0,0,0,0,0,1,0,0}, "data/mnist/testing/7/", "png", 28,28);
  ln::addToTrainingSet(&testingSet, {0,0,0,0,0,0,0,0,1,0}, "data/mnist/testing/8/", "png", 28,28);
  ln::addToTrainingSet(&testingSet, {0,0,0,0,0,0,0,0,0,1}, "data/mnist/testing/9/", "png", 28,28);

  ln::Example testEx;
  testEx.input = ln::loadImage("data/mnist/testing/7/383.png", 28, 28);

  cout << "Dataset loaded successfully" << endl;

  for (int x = 0; x < 10; x++) {
    test_net.process(testEx.input);
    test_net.printOutput();
    cout << "ERROR: " << test_net.getClassError(testingSet) << endl;
    cout << "ERROR: " << test_net.getError(testingSet) << endl;
    test_net.train(trainingSet, new StandardOptimizer(), 60000, 1, 0.000002);
  }

  cout << "done" << endl;
  return 0;
}
