cmake_minimum_required(VERSION 3.4.0)

project(sparse)

add_subdirectory(sparse)

list(APPEND CMAKE_MODULE_PATH "/home/sayandip/anaconda3/lib/python3.7/site-packages/skbuild/resources/cmake")

find_package(PythonExtensions)



set(PROJECT_SOURCE_DIR "sparse")

include(ExternalProject)

 ExternalProject_Add(
  taco
  GIT_REPOSITORY https://github.com/tensor-compiler/taco.git
  GIT_TAG        origin/master
)
