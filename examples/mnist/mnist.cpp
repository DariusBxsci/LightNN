#include <iostream>
#include <LightNet/standard.h>
#include <LightNet/data.h>
using namespace std;

int main() {

  double w = 1.0/(28*28);
  double kw = 1.0/(3*3);

  ln::Network test_net;

  /*test_net.addModule(new FeedforwardModule(28*28));
  test_net.addModule(new ConvolutionModule(1,32, 3,3, 28, 28, 0,kw/2));
  test_net.addModule(new SubsampleModule(32,28,28));
  test_net.addModule(new BiasModule(32*14*14,0,0));
  test_net.addModule(new FunctionModule(new ReluFunction()));
  test_net.addModule(new ConvolutionModule(32,2, 3,3, 14, 14, 0,kw/2));
  test_net.addModule(new SubsampleModule(64,14,14));
  test_net.addModule(new BiasModule(64*7*7,0,0));
  test_net.addModule(new FunctionModule(new ReluFunction()));
  test_net.addModule(new FeedforwardModule(10,0,w/4));*/

  test_net.addModule(new FeedforwardModule(28*28));
  test_net.addModule(new ConvolutionModule(1,5, 3,3, 28, 28, 0,kw/2));
  test_net.addModule(new SubsampleModule(5,28,28));
  test_net.addModule(new BiasModule(5*14*14,0,0));
  test_net.addModule(new FunctionModule(new ReluFunction()));
  test_net.addModule(new FeedforwardModule(10,0,w/2));

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

  test_net.setOptimizer(new StandardOptimizer());

  cout << "ESTIMATED TIME TO TRAIN " << test_net.train(trainingSet, 1, 100, 0)*100 << " seconds." << endl;

  test_net.process(testEx.input);
  test_net.printOutput();
  for (int x = 0; x < 50; x++) {
    int it = 6;
    for (int i = 0; i < it; i++) {
      cout << "Finished epoch " << i << " out of " << it << " (" << test_net.train(trainingSet, 100, 100, 0.5) << "s)" << endl;
    }

    test_net.process(testEx.input);
    test_net.printOutput();

    cout << "ERROR: " << test_net.getClassError(testingSet) << endl;
    cout << "ERROR: " << test_net.getError(testingSet) << endl;
  }

  //test_net.process(testEx.input);
  //test_net.printOutput();
  //test_net.train(trainingSet, 100, 1, 0.5);
  //test_net.process(testEx.input);
  //test_net.printOutput();


  /*ln::Network test_net;
  test_net.addModule(new FeedforwardModule(4));
  test_net.addModule(new FeedforwardModule(4,1,1));
  test_net.addModule(new FunctionModule(new SigmoidFunction()));
  test_net.addModule(new FeedforwardModule(4,1,1));
  test_net.addModule(new FunctionModule(new SigmoidFunction()));
  test_net.addModule(new FeedforwardModule(4,1,1));
  test_net.addModule(new FunctionModule(new SigmoidFunction()));
  test_net.addModule(new FeedforwardModule(4,1,1));
  test_net.addClassifier(new SoftmaxClassifier());
  test_net.linkModules();

  test_net.q_process({1,2,3,4});*/

  cout << "done" << endl;
  return 0;
}
