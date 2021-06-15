cmake_minimum_required(VERSION 3.11.0)
project(sparse)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 CONFIG)

pybind11_add_module(_cxx sparse/_cxx.cxx)
install(TARGETS _cxx LIBRARY DESTINATION sparse)