# cmake version
cmake_minimum_required(VERSION 3.22.1)
# project name
project(seedcup-cppclient)

# set c++ version
set(CMAKE_CXX_STANDARD 17)

# include dir add,split by<space>
include_directories(./util)

# link dir add
# link_directories()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

# add source
aux_source_directory(. DIR_SRCS)
aux_source_directory(./util DIR_SRCS)
add_executable(client ${DIR_SRCS})
