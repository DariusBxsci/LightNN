cmake_minimum_required(VERSION 2.4.0)
set(CMAKE_CXX_STANDARD 11)

project(LightNet)

# Find source files
file(GLOB SOURCES include/*.cpp include/*/*.cpp)

# Include header files
include_directories(include)
find_package( OpenCV REQUIRED )

# Create shared library
add_library(${PROJECT_NAME} SHARED ${SOURCES})
# Install library
install(TARGETS LightNet
  LIBRARY DESTINATION lib
)
install(DIRECTORY include/ DESTINATION /usr/local/include/LightNet)
