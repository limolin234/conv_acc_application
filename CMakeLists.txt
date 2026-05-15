cmake_minimum_required(VERSION 3.16)

project(DmaApp LANGUAGES C CXX)

option(BUILD_HOST_YOLO_CPP "Build a host-only yolo C++ checker" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(BUILD_HOST_YOLO_CPP)
    add_executable(YoloCppCheck
        src/yolo_cpp_check.cpp
        src/yolo_conv3x3.cpp
    )

    target_include_directories(YoloCppCheck PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/inc"
    )
else()
    add_executable(${PROJECT_NAME}
        src/main.cpp
        src/yolo_conv3x3.cpp
        src/tlsf.c
    )

    target_include_directories(${PROJECT_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/inc"
    )

    add_executable(AccUnitTest
        src/acc_unit_test.cpp
        src/yolo_conv3x3.cpp
        src/tlsf.c
    )

    target_include_directories(AccUnitTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/inc"
    )
endif()
