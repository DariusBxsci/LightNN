cmake_minimum_required(VERSION 2.4.0)

project(LightNet)

# Find source files
file(GLOB SOURCES include/*.cpp)

# Include header files
include_directories(include)

# Create shared library
add_library(${PROJECT_NAME} SHARED ${SOURCES})

# Install library
install(TARGETS ${PROJECT_NAME} DESTINATION lib/${PROJECT_NAME})

# Install library headers
file(GLOB HEADERS include/*.h)
install(TARGETS LightNet
  EXPORT LightNet
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

