cmake_minimum_required(VERSION 3.11.0)
project(sparse)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 CONFIG)

# Satisfy pybind11 dep from source if not found on system
if(NOT TARGET pybind11::pybind11)
  include(FetchContent)

  # To prevent download of the pybind11 source you can overwrite the following
  # behavior by setting the cmake variable FETCHCONTENT_SOURCE_DIR_PYBIND11 to a
  # local source directory
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.6.2
  )
  FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(_cxx sparse/_cxx.cxx)
install(TARGETS _cxx LIBRARY DESTINATION sparse)