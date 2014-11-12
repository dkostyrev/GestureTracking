cmake_minimum_required(VERSION 2.8)
project(GestureTracking)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
cmake_policy(SET CMP0004 OLD)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SRC_LIST
    main.cpp
    src/vibe.h
    src/vibe.cpp
    src/timedispersion.h
    src/timedispersion.cpp
    src/blobgetter.h
    src/blobgetter.cpp
    src/blobprocessor.h
    src/blobprocessor.cpp
    src/integralorientationhistogram.h
    src/integralorientationhistogram.cpp
    src/colorfinder.h
    src/colorfinder.cpp
    src/classifier.h
    src/classifier.cpp
    src/controller.h
    src/controller.cpp
    src/motionestimator_serial.h
    src/motionestimator_serial.cpp
    src/blobintegralhistogram.h
    src/blobintegralhistogram.cpp
)

add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
