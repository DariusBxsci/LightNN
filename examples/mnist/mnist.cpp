  #include <iostream>
#include <LightNet/standard.h>
#include <LightNet/data.h>
#include <SFML/Graphics.hpp>
using namespace std;

vector<double> drawDigit() { //uses the SFML library to create a simple painting interface to test digit recognition

    vector<vector<double>> digit2d;
    digit2d.resize(28);
    for(unsigned int x = 0; x < digit2d.size(); x++) {
      digit2d[x].resize(28);
    }

    vector<double> digit;
    sf::RenderWindow window(sf::VideoMode(560, 560), "Draw a digit!");
    sf::RectangleShape shape(sf::Vector2f(28, 28));
    shape.setFillColor(sf::Color(255, 255, 255));
    window.clear();
    while (window.isOpen())
        {
            sf::Event event;
            while (window.pollEvent(event))
            {
                if (event.type == sf::Event::Closed) {
                    window.close();
                }
            }

            shape.setPosition(sf::Mouse::getPosition(window).x, sf::Mouse::getPosition(window).y);
            if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
                window.draw(shape);
                int px = sf::Mouse::getPosition(window).x;
                int py = sf::Mouse::getPosition(window).y;
                //cout << floor(px/20.0) << "," << floor(py/20.0) << endl;
                if(px < 560 && px > 50 && py < 560 && py > 50) {
                  digit2d[floor(px/20.0)][floor(py/20.0)] = 1;

                  /*if(digit2d[floor(px/20.0)+1][floor(py/20.0)] != 1) digit2d[floor(px/20.0)+1][floor(py/20.0)] = 0.5;
                  if(digit2d[floor(px/20.0)-1][floor(py/20.0)] != 1) digit2d[floor(px/20.0)-1][floor(py/20.0)] = 0.5;
                  if(digit2d[floor(px/20.0)][floor(py/20.0)+1] != 1) digit2d[floor(px/20.0)][floor(py/20.0)+1] = 0.5;
                  if(digit2d[floor(px/20.0)][floor(py/20.0)-1] != 1) digit2d[floor(px/20.0)][floor(py/20.0)-1] = 0.5;*/

                }
            }

            window.display();
        }

        for(unsigned int x = 0; x < digit2d.size(); x++) {
          for(unsigned int y = 0; y < digit2d[0].size(); y++) {
            digit.push_back(digit2d[y][x]);
          }
        }

    return digit;
}

int main() {

  ln::Network test_net;

  test_net.addModule(new FeedforwardModule(28*28));
  test_net.addModule(new ConvolutionModule(1,32, 5,5, 28,28, 0, 1.0/(5*5)));
  test_net.addModule(new BiasModule(32*28*28,0,0));
  test_net.addModule(new FunctionModule(new ReluFunction()));
  test_net.addModule(new FeedforwardModule(10,0,1.0/(784*8)));

  /*test_net.addModule(new FeedforwardModule(28*28));
  test_net.addModule(new ConvolutionModule(1,4, 3,3, 28,28, 0, 1.0/(3*3)));
  test_net.addModule(new BiasModule(4*28*28,0,0));
  test_net.addModule(new FunctionModule(new ReluFunction()));
  test_net.addModule(new FeedforwardModule(10,0,1.0/(784*4)));*/

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

  cout << "ESTIMATED TIME TO TRAIN " << test_net.train(trainingSet, 1, 4, 0)*2500 << " seconds." << endl;

  test_net.load("./models/test_model");

  double cerr = 1;
  test_net.process(testEx.input);
  test_net.printOutput();

  cout << "ERROR: " << test_net.getClassError(testingSet) << endl;

  for (int x = 0; cerr > 0.01; x++) {

    int it = 6;
    for (int i = 0; i < it; i++) {
      cout << "Finished epoch " << i << " out of " << it << " (" << test_net.train(trainingSet, 2500, 4, 0.3) << "s)" << endl;
    }

    test_net.process(testEx.input);
    test_net.printOutput();

    double lastcerr = cerr;
    cerr = test_net.getClassError(testingSet);
    cout << "ERROR: " << cerr << endl;
    cout << "ERROR: " << test_net.getError(testingSet) << endl;

    test_net.save("./models/test_model");
  }

  while (true) {
    vector<double> digit = drawDigit();

    /*for(int x = 0; x < 28; x++) {
      cout << endl;
      for (int y = 0; y < 28; y++) {
        cout << digit[x*28+y] << " ";
      }
    }
    cout << endl;

    for(int x = 0; x < 28; x++) {
      cout << endl;
      for (int y = 0; y < 28; y++) {
        cout << testingSet.examples[rand()%10000].input[x*28+y] << " ";
      }
    }*/
    cout << endl;

    test_net.process(digit);
    cout << "THIS IS A " << test_net.getClass() << endl;
  }

  cout << "done" << endl;
  return 0;
}
