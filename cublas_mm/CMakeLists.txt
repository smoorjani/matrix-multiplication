cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cublas_mm)
set(CMAKE_CXX_STANDARD 11)

set(Caffe2_DIR "/home/user/src/pytorch/torch/share/cmake/Caffe2")

find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_executable(cublas_mm cublas_mm.cpp)
target_link_libraries(cublas_mm "${PYTHON_LIBRARIES}")
target_link_libraries(cublas_mm "${TORCH_LIBRARIES}")
set_property(TARGET cublas_mm PROPERTY CXX_STANDARD 14)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET cublas_mm
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:cublas_mm>)
endif (MSVC)

