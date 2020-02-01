cmake_minimum_required(VERSION 3.5)
project(box_tree)

find_package(embree 3 REQUIRED)
find_package(TBB REQUIRED)
find_package(glm REQUIRED)

if (NOT WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
endif()


add_executable(box_tree box_tree.cpp)

set_target_properties(box_tree PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

target_compile_options(box_tree PUBLIC
    -DGLM_ENABLE_EXPERIMENTAL)

target_include_directories(box_tree PUBLIC
    $<BUILD_INTERFACE:${EMBREE_INCLUDE_DIRS}>)

target_link_libraries(box_tree PUBLIC
    glm::glm TBB::tbb ${EMBREE_LIBRARIES})


