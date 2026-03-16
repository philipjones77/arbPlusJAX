cmake_minimum_required(VERSION 3.16)
project(boost_ref_adapter LANGUAGES CXX)

add_executable(boost_ref_adapter boost_ref_adapter.cpp)
target_compile_features(boost_ref_adapter PRIVATE cxx_std_17)

set(BOOST_ROOT "$ENV{BOOST_ROOT}" CACHE PATH "Boost root directory")
set(BOOST_INCLUDEDIR "$ENV{BOOST_INCLUDEDIR}" CACHE PATH "Boost include directory")
set(BOOST_LIBRARYDIR "$ENV{BOOST_LIBRARYDIR}" CACHE PATH "Boost library directory")

if(BOOST_INCLUDEDIR)
  target_include_directories(boost_ref_adapter PRIVATE "${BOOST_INCLUDEDIR}")
elseif(BOOST_ROOT)
  target_include_directories(boost_ref_adapter PRIVATE "${BOOST_ROOT}/include")
endif()

if(BOOST_LIBRARYDIR)
  target_link_directories(boost_ref_adapter PRIVATE "${BOOST_LIBRARYDIR}")
endif()

if (MSVC)
  target_compile_options(boost_ref_adapter PRIVATE /W4)
else()
  target_compile_options(boost_ref_adapter PRIVATE -Wall -Wextra -Wpedantic)
endif()
