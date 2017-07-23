cmake_minimum_required(VERSION 2.4.0)
set(CMAKE_CXX_STANDARD 11)

project(example_mnist)

add_executable(mnist mnist.cpp)
target_link_libraries(mnist LightNet)

