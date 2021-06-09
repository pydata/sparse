cmake_minimum_required(VERSION 3.4.0)

project(sparse)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}")

find_package (Python COMPONENTS Interpreter Development)





include(FetchContent)

FetchContent_Declare(
  taco
  GIT_REPOSITORY https://github.com/tensor-compiler/taco.git
  GIT_TAG        master
)

FetchContent_MakeAvailable(taco)

add_executable(sparse sparse)