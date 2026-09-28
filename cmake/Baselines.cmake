if(NOT BLITZ_TOOLS)
  message(FATAL_ERROR "Research baselines require BLITZ_TOOLS")
endif()
include(FetchContent)
FetchContent_Declare(meshoptimizer
  GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
  GIT_TAG 9e1f07b159d3cb777f1c67ed31fc11fd117986f4)
FetchContent_Declare(fast_quadric
  GIT_REPOSITORY https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification.git
  GIT_TAG 65df07dc54766e3ee480482f1c881a62767831cc)
set(MESHOPT_BUILD_DEMO OFF CACHE BOOL "" FORCE)
set(MESHOPT_BUILD_GLTFPACK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(meshoptimizer fast_quadric)
add_executable(blitz-baseline-meshopt tools/baseline_meshopt.cpp)
target_link_libraries(blitz-baseline-meshopt PRIVATE blitz_io meshoptimizer)
add_executable(blitz-vegetation-meshopt tools/vegetation_meshopt.cpp)
target_link_libraries(blitz-vegetation-meshopt PRIVATE blitz_io meshoptimizer)
add_executable(blitz-baseline-fastquadric tools/baseline_fastquadric.cpp)
target_include_directories(blitz-baseline-fastquadric PRIVATE "${fast_quadric_SOURCE_DIR}/src.cmd")
target_link_libraries(blitz-baseline-fastquadric PRIVATE blitz_io)
find_package(CGAL 6.1 REQUIRED)
find_package(Eigen3 REQUIRED)
add_executable(blitz-baseline-cgal tools/baseline_cgal.cpp)
target_link_libraries(blitz-baseline-cgal PRIVATE blitz_io CGAL::CGAL Eigen3::Eigen)
