
add_executable(01_helloOptix main.cpp)

target_link_libraries(01_helloOptix PUBLIC CUDA::cudart OptiX::OptiX)