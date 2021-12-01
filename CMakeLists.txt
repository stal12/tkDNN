cmake_minimum_required(VERSION 3.18...3.19)

project(tkDNN VERSION 1.0
              DESCRIPTION "tkDNN project"
              LANGUAGES CXX CUDA)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32)
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "/O2 /FS /EHsc")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif(WIN32)

if(DEBUG)
    add_definitions(-DDEBUG)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tkConf.cmake OPTIONAL)

#-------------------------------------------------------------------------------
# External Libraries
#-------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
find_package(CUDNN REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenCV COMPONENTS dnn QUIET)
find_package(yaml-cpp REQUIRED)

#-------------------------------------------------------------------------------
# CUDA DNN
#-------------------------------------------------------------------------------
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "--maxrregcount=32")
file(GLOB tkdnn_CUSRC "src/kernels/*.cu" "src/sorting.cu")
add_library(dnnCUDA ${tkdnn_CUSRC})
target_include_directories(dnnCUDA PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include> 
    ${CUDNN_INCLUDE_DIRS}
)
target_link_libraries(dnnCUDA 
    CUDA::cudart 
    CUDA::cublas
    ${CUDNN_LIBRARIES}
)
#set_target_properties(dnnCUDA PROPERTIES CUDA_ARCHITECTURES "61")
set_target_properties(dnnCUDA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_library(${PROJECT_NAME}::dnnCUDA ALIAS dnnCUDA)

#-------------------------------------------------------------------------------
# tkDNN Library
#-------------------------------------------------------------------------------
file(GLOB tkdnn_SRC "src/*.cpp")
add_library(dnn SHARED ${tkdnn_SRC})
target_link_libraries(dnn 
    ${PROJECT_NAME}::dnnCUDA 
    Eigen3::Eigen 
    ${YAML_CPP_LIBRARIES}
    ${OpenCV_LIBS}
)
target_compile_options(dnn PUBLIC -DTKDNN_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(dnn PUBLIC -DOPENCV)
if(OpenCV_dnn_FOUND)
    target_compile_options(dnn PUBLIC -DOPENCV_HAS_DNN)
endif()
if(OpenCV_CUDA_VERSION)
target_compile_options(dnn PUBLIC -DOPENCV_CUDACONTRIB)
endif()
add_library(${PROJECT_NAME}::dnn ALIAS dnn)

#-------------------------------------------------------------------------------
# Small test
#-------------------------------------------------------------------------------
add_executable(test_simple tests/simple/test_simple.cpp)
target_link_libraries(test_simple ${PROJECT_NAME}::dnn)

add_executable(test_mnist tests/mnist/test_mnist.cpp)
target_link_libraries(test_mnist ${PROJECT_NAME}::dnn)

add_executable(test_mnistRT tests/mnist/test_mnistRT.cpp)
target_link_libraries(test_mnistRT ${PROJECT_NAME}::dnn)

add_executable(test_imuodom tests/imuodom/imuodom.cpp)
target_link_libraries(test_imuodom ${PROJECT_NAME}::dnn)

#-------------------------------------------------------------------------------
# Darknet
#-------------------------------------------------------------------------------
file(GLOB darknet_SRC "tests/darknet/*.cpp")
foreach(test_SRC ${darknet_SRC})
    get_filename_component(test_NAME "${test_SRC}" NAME_WE)
    set(test_NAME test_${test_NAME})
    add_executable(${test_NAME} ${test_SRC})
    target_link_libraries(${test_NAME} ${PROJECT_NAME}::dnn)
endforeach()

#-------------------------------------------------------------------------------
# Mobilenet
#-------------------------------------------------------------------------------
add_executable(test_mobilenetv2ssd tests/mobilenet/mobilenetv2ssd/mobilenetv2ssd.cpp)
target_link_libraries(test_mobilenetv2ssd ${PROJECT_NAME}::dnn)

add_executable(test_bdd-mobilenetv2ssd tests/mobilenet/bdd-mobilenetv2ssd/bdd-mobilenetv2ssd.cpp)
target_link_libraries(test_bdd-mobilenetv2ssd ${PROJECT_NAME}::dnn)

add_executable(test_mobilenetv2ssd512 tests/mobilenet/mobilenetv2ssd512/mobilenetv2ssd512.cpp)
target_link_libraries(test_mobilenetv2ssd512 ${PROJECT_NAME}::dnn)

#-------------------------------------------------------------------------------
# Backbones
#-------------------------------------------------------------------------------
add_executable(test_resnet101 tests/backbones/resnet101/resnet101.cpp)
target_link_libraries(test_resnet101 ${PROJECT_NAME}::dnn)

add_executable(test_dla34 tests/backbones/dla34/dla34.cpp)
target_link_libraries(test_dla34 ${PROJECT_NAME}::dnn)

#-------------------------------------------------------------------------------
# Centernet
#-------------------------------------------------------------------------------
add_executable(test_resnet101_cnet tests/centernet/resnet101_cnet/resnet101_cnet.cpp)
target_link_libraries(test_resnet101_cnet ${PROJECT_NAME}::dnn)

add_executable(test_dla34_cnet tests/centernet/dla34_cnet/dla34_cnet.cpp)
target_link_libraries(test_dla34_cnet ${PROJECT_NAME}::dnn)

add_executable(test_resnet101_cnet3d tests/centernet/resnet101_cnet3d/resnet101_cnet3d.cpp)
target_link_libraries(test_resnet101_cnet3d ${PROJECT_NAME}::dnn)

add_executable(test_dla34_cnet3d tests/centernet/dla34_cnet3d/dla34_cnet3d.cpp)
target_link_libraries(test_dla34_cnet3d ${PROJECT_NAME}::dnn)

# CENTERTRACK

add_executable(test_dla34_ctrack tests/centertrack/dla34_ctrack/dla34_ctrack.cpp)
target_link_libraries(test_dla34_ctrack ${PROJECT_NAME}::dnn)

# SHELFNET
add_executable(test_shelfnet tests/shelfnet/shelfnet.cpp)
target_link_libraries(test_shelfnet ${PROJECT_NAME}::dnn)

add_executable(test_shelfnet_berkeley tests/shelfnet/shelfnet_berkeley.cpp)
target_link_libraries(test_shelfnet_berkeley ${PROJECT_NAME}::dnn)

add_executable(test_shelfnet_mapillary tests/shelfnet/shelfnet_mapillary.cpp)
target_link_libraries(test_shelfnet_mapillary ${PROJECT_NAME}::dnn)

add_executable(test_shelfnet_coco tests/shelfnet/shelfnet_coco.cpp)
target_link_libraries(test_shelfnet_coco ${PROJECT_NAME}::dnn)

#-------------------------------------------------------------------------------
# Demos
#-------------------------------------------------------------------------------
add_executable(test_rtinference tests/test_rtinference/rtinference.cpp)
target_link_libraries(test_rtinference ${PROJECT_NAME}::dnn)

add_executable(map_demo demo/demo/map.cpp)
target_link_libraries(map_demo ${PROJECT_NAME}::dnn)

add_executable(demo demo/demo/demo.cpp)
target_link_libraries(demo ${PROJECT_NAME}::dnn)

add_executable(demo3D demo/demo/demo3D.cpp)
target_link_libraries(demo3D ${PROJECT_NAME}::dnn)

add_executable(demoTracker demo/demo/demoTracker.cpp)
target_link_libraries(demoTracker ${PROJECT_NAME}::dnn)

add_executable(seg_demo demo/demo/seg_demo.cpp)
target_link_libraries(seg_demo ${PROJECT_NAME}::dnn)

add_executable(reID_demo demo/demo/reIDdemo.cpp)
target_link_libraries(reID_demo ${PROJECT_NAME}::dnn)

#-------------------------------------------------------------------------------
# Install
#-------------------------------------------------------------------------------
install(TARGETS dnnCUDA dnn
    DESTINATION "lib/"
    EXPORT ${PROJECT_NAME}Targets
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" 
    DESTINATION "include/"
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/" 
    DESTINATION "share/${PROJECT_NAME}/cmake/" 
)

#-------------------------------------------------------------------------------
# Export Configuration
#-------------------------------------------------------------------------------
install(EXPORT ${PROJECT_NAME}Targets NAMESPACE ${PROJECT_NAME}::
    FILE ${PROJECT_NAME}Targets.cmake
    DESTINATION "share/${PROJECT_NAME}/cmake/"
)

include(CMakePackageConfigHelpers)

# generate the config file that is includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION "share/${PROJECT_NAME}/cmake/"
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# generate the version file for the config file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION "1.0"
    COMPATIBILITY AnyNewerVersion
)

# export targets for build tree
export(EXPORT ${PROJECT_NAME}Targets NAMESPACE ${PROJECT_NAME}::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
)

# install the configuration file
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION "share/${PROJECT_NAME}/cmake/"
)
