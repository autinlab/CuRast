cmake_minimum_required(VERSION 3.5)

set(TARGET_NAME CuRast) 
project(${TARGET_NAME} LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 23)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(cmake/common.cmake)

# Paths to TurboJPEG
# The bundled libs/jpg/turbojpeg.h is from libjpeg-turbo 3.1.1; the prebuilt .lib next to it
# is Windows-only. On other platforms point TURBOJPEG_LIBRARY at a matching 3.1.x build.
if (WIN32)
	set(TURBOJPEG_LIBRARY "${CMAKE_SOURCE_DIR}/libs/jpg/turbojpeg-static.lib")
else()
	find_library(TURBOJPEG_LIBRARY NAMES turbojpeg
		HINTS ENV TURBOJPEG_ROOT
		PATH_SUFFIXES lib lib64)
	if (NOT TURBOJPEG_LIBRARY)
		message(FATAL_ERROR "libturbojpeg not found. Build libjpeg-turbo 3.1.x and set TURBOJPEG_ROOT.")
	endif()
	message(STATUS "TURBOJPEG_LIBRARY:            ${TURBOJPEG_LIBRARY}")
endif()

# --------------------------------------------------------
# Source files (C++ and CUDA)
# --------------------------------------------------------
file(GLOB PROJECT_SOURCES CONFIGURE_DEPENDS 
	src/*.[ch]pp
	src/*.cu
	src/kernels/textureTools.cu
	src/kernels/lines.cu
	#src/kernels/rasterization_utils.cu
)

add_executable(${TARGET_NAME} ${PROJECT_SOURCES})
if (MSVC)
	target_compile_options(${TARGET_NAME} PRIVATE
		$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
		$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /Zc:preprocessor>
	)
endif()
target_compile_definitions(${TARGET_NAME} PUBLIC
	IMGUI_DEFINE_MATH_OPERATORS
	_HAS_STD_BYTE=0
)
if (WIN32)
	target_compile_definitions(${TARGET_NAME} PUBLIC VK_USE_PLATFORM_WIN32_KHR)
	set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(SolutionDir)..")
endif()

# Include paths
target_include_directories(${TARGET_NAME} PRIVATE
	./src
	./libs
	${TURBOJPEG_INCLUDE_DIR}
)

add_subdirectory(libs/laszip)
target_link_libraries(${TARGET_NAME} PRIVATE
	${TURBOJPEG_LIBRARY}
	laszip
)

if (NOT WIN32)
	# libstdc++ needs these spelled out where MSVC does not:
	#  - <execution> parallel algorithms dispatch to TBB
	#  - <stacktrace> (used by unsuck.hpp) lives in libstdc++exp
	find_package(TBB QUIET)
	if (TBB_FOUND)
		target_link_libraries(${TARGET_NAME} PRIVATE TBB::tbb)
	else()
		target_link_libraries(${TARGET_NAME} PRIVATE tbb)
	endif()
	target_link_libraries(${TARGET_NAME} PRIVATE stdc++exp)
endif()

# Add other third-party libs
ADD_CUDA(${TARGET_NAME})
ADD_VULKAN(${TARGET_NAME})
ADD_GLM(${TARGET_NAME})
ADD_IMGUI(${TARGET_NAME})

# --------------------------------------------------------
# CUDA configuration
# --------------------------------------------------------
# Choose GPU architectures (adjust for your machine).
# 120 = Blackwell (RTX PRO 6000, CC 12.0) and needs CUDA >= 12.9.
# Override with -D CURAST_CUDA_ARCH=120 to build a single arch; the full list is slow
# under separable compilation. Set on the target too: the bare set() below runs after
# add_executable(), so it would otherwise not reach the already-created target.
if (NOT CURAST_CUDA_ARCH)
	set(CURAST_CUDA_ARCH 75 86 89 90 120)
endif()
set(CMAKE_CUDA_ARCHITECTURES ${CURAST_CUDA_ARCH})
set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES ${CURAST_CUDA_ARCH})
message(STATUS "CURAST_CUDA_ARCH:             ${CURAST_CUDA_ARCH}")

# Enable separable compilation (for multi-file CUDA builds)
set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Treat warnings as errors (optional)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror all-warnings")

# set proper startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${TARGET_NAME})







# Second project
add_executable(PlyToGlb 
	src/tools/PlyToGlb.cpp
	src/unsuck_platform_specific.cpp
)
target_include_directories(PlyToGlb PRIVATE libs/glm)
target_include_directories(PlyToGlb PRIVATE
	./src
	./libs
)
if (NOT WIN32)
	# Same <execution>/<stacktrace> link requirements as the main target.
	find_package(TBB QUIET)
	if (TBB_FOUND)
		target_link_libraries(PlyToGlb PRIVATE TBB::tbb)
	else()
		target_link_libraries(PlyToGlb PRIVATE tbb)
	endif()
	target_link_libraries(PlyToGlb PRIVATE stdc++exp)
endif()