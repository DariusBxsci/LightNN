
cmake_minimum_required(VERSION 2.4.0)
set(CMAKE_CXX_STANDARD 11)

project(example_mnist)
find_package( OpenCV REQUIRED )

    if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)
    endif(COMMAND cmake_policy)

add_executable(mnist mnist.cpp)
target_link_libraries(mnist LightNet ${OpenCV_LIBS})


