cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

# Rembember to download di abicxx11
set(CMAKE_PREFIX_PATH /home/pietro/Downloads/libtorch)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(Torch REQUIRED)
find_package(OpenCV 4 REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
