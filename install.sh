cd build; cmake ..; make; sudo make install;
echo "FINISHED INSTALLING LIGHTNET"

cd ../examples;
echo "BUILDING EXAMPLES..."
cd mnist; cmake .; make;
echo "FINISHED BUILDING MNIST EXAMPLE"


