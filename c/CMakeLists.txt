cmake_minimum_required(VERSION 3.10)
project(ECG C)
set(CMAKE_C_STANDARD 11)

# Enable compile_commands.json generation
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C compiler to clang
set(CMAKE_C_COMPILER clang)

# Set compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -g")

# Set directories
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")

# Find all source files
file(GLOB SOURCES "${SRC_DIR}/*.c")

# Add executable
add_executable(main ${SOURCES})

# Set include directories
target_include_directories(main PRIVATE
    ${INCLUDE_DIR}
    ${LIB_DIR}/sml
)

# Set library directories and link libraries
target_link_directories(main PRIVATE ${LIB_DIR})
target_link_libraries(main PRIVATE
    ${LIB_DIR}/sml/sml.lib
    m
)

# Set output directory
set_target_properties(main
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    OUTPUT_NAME "main.out"
)

# Add custom target for running the executable
add_custom_target(run
    COMMAND ./main.out
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Create build directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR})
