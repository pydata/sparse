cmake_minimum_required(VERSION 3.4.0)

find_package(PythonExtensions)

include( ExternalProject)

 ExternalProject_Add(
  taco
  GIT_REPOSITORY https://github.com/tensor-compiler/taco.git
  GIT_TAG        origin/master
)
