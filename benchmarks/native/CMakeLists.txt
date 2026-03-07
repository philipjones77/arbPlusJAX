cmake_minimum_required(VERSION 3.16)
project(boost_ref_adapter LANGUAGES CXX)

add_executable(boost_ref_adapter boost_ref_adapter.cpp)
target_compile_features(boost_ref_adapter PRIVATE cxx_std_17)

if (MSVC)
  target_compile_options(boost_ref_adapter PRIVATE /W4)
else()
  target_compile_options(boost_ref_adapter PRIVATE -Wall -Wextra -Wpedantic)
endif()
