############################################################################################
# <summary> CMakeLists.txt for ECE408 								</summary>
#CUDA_INCLUDE_DIRS <date>    2021-06-07          											</date>
# <author>  yuanxiulong 			</author>
# <email>   yuanxl19@mails.tsinghua.edu.cn	</email>
############################################################################################
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(GPU VERSION 0.0.1
        DESCRIPTION "reference implementation of CUDA kernel"
        LANGUAGES CXX CUDA)  # CUDA is enable later after the check

# need to specify if not in default path
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.2/bin/nvcc")

# arch
set(CMAKE_CUDA_ARCHITECTURES 80)  # A100


######################################################################################################################
# DEPENDENCIES
######################################################################################################################

# add cuda & hello-world
# include(CheckLanguage)
# check_language(CUDA)
# if(CMAKE_CUDA_COMPILER)
#         message(STATUS "[CUDA]pass cuda check")
#         enable_language(CUDA)
# else()
#         message(FATAL_ERROR "[CUDA]No CUDA support")
# endif()

add_executable(hello hello_world.cu)
# set_target_properties(hello PROPERTIES CUDA_STANDARD 11)  # set property to target

# find clang-format & clang-tidy BIN, but still need to exec these bins in custom scripts
# clang-format
# find_program(CLANG_FORMAT_BIN
#         NAMES clang-format clang-format-8
#         HINTS ${BUSTUB_CLANG_SEARCH_PATH})
# if ("${CLANG_FORMAT_BIN}" STREQUAL "CLANG_FORMAT_BIN-NOTFOUND")
#     message(FATAL_ERROR "cannot find clang-format")
# else()
#     message(STATUS "found clang-format at ${CLANG_FORMAT_BIN}")
# endif()
# find_program(CLANG_TIDY_BIN
#         NAMES clang-tidy clang-tidy-8
#         HINTS ${BUSTUB_CLANG_SEARCH_PATH})
# if ("${CLANG_TIDY_BIN}" STREQUAL "CLANG_TIDY_BIN-NOTFOUND")
#     message(FATAL_ERROR "cannot find clang-tidy")
# else()
#     # Output compile_commands.json
#     set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
#     message(STATUS "found clang-tidy at ${CLANG_TIDY_BIN}")
# endif()


# # add lib; NOTE: GLOB is deprecated, CONFIGURE_DEPENDS is preferred
# file( GLOB  wbhdr lib/*.hpp lib/*.h )
# file( GLOB  wbsrc lib/*.cpp lib/*.c )
# message(STATUS "CMAKE SOURCE DIR: ${CMAKE_SOURCE_DIR} ")
# include_directories("${CMAKE_SOURCE_DIR}")
# include_directories("lib/")
#
# ######################################################################################################################
# # MAKE TARGETS
# ######################################################################################################################
#
# ##########################################
# # "make format"
# # "make check-format"
# ##########################################
#
# string(CONCAT FORMAT_DIRS
#         "${CMAKE_CURRENT_SOURCE_DIR},"
#         "${CMAKE_CURRENT_SOURCE_DIR}/my-impl,"
#         )
# # runs clang format and updates files in place.
# set(BUILD_SUPPORT_DIR "${CMAKE_SOURCE_DIR}/build_support")
# add_custom_target(format ${BUILD_SUPPORT_DIR}/run_clang_format.py
#         ${CLANG_FORMAT_BIN}
#         ${BUILD_SUPPORT_DIR}/clang_format_exclusions.txt
#         --source_dirs
#         ${FORMAT_DIRS}
#         --fix
#         --quiet
#         )
#
# # runs clang format and exits with a non-zero exit code if any files need to be reformatted
# add_custom_target(check-format ${BUILD_SUPPORT_DIR}/run_clang_format.py
#         ${CLANG_FORMAT_BIN}
#         ${BUILD_SUPPORT_DIR}/clang_format_exclusions.txt
#         --source_dirs
#         ${FORMAT_DIRS}
#         --quiet
#         )
#
#
# # implementation code
#
# # mp0 as an example
# ## e.g. compile a library and link
# add_library(wblib SHARED ${wbhdr} ${wbsrc})  # STATIC or SHARED determines how wblib is linked
# add_executable(mp0 my-impl/mp0.cu)
# target_link_libraries(mp0 PUBLIC wblib)
# ## vs directly compile
# #add_executable(mp0 my-impl/mp0.cu ${wbhdr} ${wbsrc})
# ## finish e.g.
#
# add_executable(mp1 my-impl/mp1.cu)
# target_link_libraries(mp1 PUBLIC wblib)
# #add_executable(ref-mp1 ref-impl/mp1_implement.cu)
# #target_link_libraries(ref-mp1 PUBLIC wblib)
#
# add_executable(mp2 my-impl/mp2.cu)
# target_link_libraries(mp2 PUBLIC wblib)
#
# add_executable(mp3 my-impl/mp3.cu)
# target_link_libraries(mp3 PUBLIC wblib)
#
# add_executable(mp4 my-impl/mp4.cu)
# target_link_libraries(mp4 PUBLIC wblib)
#
# add_executable(mp5 my-impl/mp5.cu)
# target_link_libraries(mp5 PUBLIC wblib)
#
# add_executable(mp6 my-impl/mp6.cu)
# target_link_libraries(mp6 PUBLIC wblib)
#
# add_executable(ref-mp6 ref-impl/mp6_implement.cu)
# target_link_libraries(ref-mp6 PUBLIC wblib)
#
# add_executable(mp7 my-impl/mp7.cu)
# target_link_libraries(mp7 PUBLIC wblib)
# add_executable(ref-mp7 ref-impl/mp7_implement.cu)
# target_link_libraries(ref-mp7 PUBLIC wblib)
#
# add_executable(mp8 my-impl/mp8.cu)
# target_link_libraries(mp8 PUBLIC wblib)
# add_executable(ref-mp8 ref-impl/mp8_implement.cu)
# target_link_libraries(ref-mp8 PUBLIC wblib)
#
# add_executable(mp11 my-impl/mp11.cu)
# target_link_libraries(mp11 PUBLIC wblib)
# #Add Tools
# #add_executable(MP1_Dataset tools/mp1_dataset.cpp ${wbhdr} ${wbsrc})
# #add_executable(MP2_Dataset tools/mp2_dataset.cpp ${wbhdr} ${wbsrc})
# #add_executable(MP3_Dataset tools/mp3_dataset.cpp ${wbhdr} ${wbsrc})
# #add_executable(MP4_Dataset tools/mp4_dataset.cpp ${wbhdr} ${wbsrc})
# #add_executable(MP5_Dataset tools/mp5_dataset.cpp ${wbhdr} ${wbsrc})
# #add_executable(MP6_Dataset tools/mp6_dataset.cpp ${wbhdr} ${wbsrc})
# #add_executable(MP7_Dataset tools/mp7_dataset.cpp ${wbhdr} ${wbsrc})
# #add_executable(Bank_Conflict tools/bank_conflict.cu ${wbhdr} ${wbsrc})
# #add_executable(Memory_Transfer tools/memory_transfer.cu ${wbhdr} ${wbsrc})
#
