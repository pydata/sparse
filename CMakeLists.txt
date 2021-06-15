cmake_minimum_required(VERSION 3.4.0)

project(sparse)

list(APPEND CMAKE_MODULE_PATH "/home/sayandip/anaconda3/lib/python3.7/site-packages/skbuild/resources/cmake")

find_package(PythonExtensions)



include(FetchContent)

FetchContent_Declare(
  taco
  GIT_REPOSITORY https://github.com/tensor-compiler/taco.git
  GIT_TAG        master
)

FetchContent_MakeAvailable(taco)

if(NOT taco_POPULATED)
  FetchContent_Populate(taco)
  add_subdirectory(${taco_SOURCE_DIR} ${taco_BINARY_DIR})
endif()
