function(ADD_IMGUI TARGET_NAME)
	target_include_directories(${TARGET_NAME} PRIVATE
		libs/imgui
		libs/imgui/backends)

	target_sources(${TARGET_NAME} PRIVATE
		libs/imgui/imgui.cpp
		libs/imgui/imgui_demo.cpp
		libs/imgui/imgui_draw.cpp
		libs/imgui/imgui_tables.cpp
		libs/imgui/imgui_widgets.cpp
		libs/imgui/backends/imgui_impl_glfw.cpp
		libs/imgui/backends/imgui_impl_vulkan.cpp)
endfunction()

function(ADD_IMPLOT TARGET_NAME)
	target_include_directories(${TARGET_NAME} PRIVATE
		libs/implot)
	target_sources(${TARGET_NAME} PRIVATE
		libs/implot/implot_items.cpp
		libs/implot/implot.cpp)
endfunction()

function(ADD_IMGUIZMO TARGET_NAME)
	target_include_directories(${TARGET_NAME} PRIVATE
		libs/ImGuizmo-1.83)
	target_sources(${TARGET_NAME} PRIVATE
		libs/ImGuizmo-1.83/ImGuizmo.cpp)
endfunction()



function(ADD_GLM TARGET_NAME)
	target_include_directories(${TARGET_NAME} PRIVATE libs/glm)
endfunction()

function(ADD_CUDA TARGET_NAME)
	# 12.9 is the floor, not 13.1: it is the first widely deployed toolkit with
	# sm_120 (Blackwell) plus the nvrtc/nvJitLink LTO path this project uses, and
	# it is what the Garibaldi cluster actually has installed.
	if (NOT CURAST_CUDA_MIN_VERSION)
		set(CURAST_CUDA_MIN_VERSION 12.9)
	endif()
	find_package(CUDAToolkit ${CURAST_CUDA_MIN_VERSION} REQUIRED)
	find_library(CUDA_DEVRTLIB NAMES cudadevrt libcudadevrt PATHS "${CUDAToolkit_LIBRARY_DIR}")

	MESSAGE(STATUS "CUDAToolkit_INCLUDE_DIRS:     " ${CUDAToolkit_INCLUDE_DIRS})
	MESSAGE(STATUS "CUDAToolkit_BIN_DIR:          " ${CUDAToolkit_BIN_DIR})
	MESSAGE(STATUS "CUDAToolkit_LIBRARY_DIR:      " ${CUDAToolkit_LIBRARY_DIR})
	MESSAGE(STATUS "CUDAToolkit_LIBRARY_ROOT:     " ${CUDAToolkit_LIBRARY_ROOT})
	MESSAGE(STATUS "CUDAToolkit_NVCC_EXECUTABLE:  " ${CUDAToolkit_NVCC_EXECUTABLE})
	MESSAGE(STATUS "CUDA_DEVRTLIB:                " ${CUDA_DEVRTLIB})

	target_include_directories(${TARGET_NAME} PRIVATE CUDAToolkit_INCLUDE_DIRS)
	target_link_libraries(${TARGET_NAME} PRIVATE
		CUDA::cuda_driver
		CUDA::nvrtc
		CUDA::nvJitLink
	)

	target_compile_definitions(${TARGET_NAME} PRIVATE CUDA_DEVRTLIB="${CUDA_DEVRTLIB}")

	# NVRTC needs the toolkit include dir at runtime. Pass what CMake resolved rather than
	# letting the app derive it from CUDA_DEVRTLIB: the lib layout differs between
	# Windows (lib/x64) and Linux (targets/<arch>/lib via a lib64 symlink).
	list(GET CUDAToolkit_INCLUDE_DIRS 0 CUDA_INCLUDE_DIR_FIRST)
	message(STATUS "CUDA_INCLUDE_DIR:             " ${CUDA_INCLUDE_DIR_FIRST})
	target_compile_definitions(${TARGET_NAME} PRIVATE CUDA_INCLUDE_DIR="${CUDA_INCLUDE_DIR_FIRST}")
endfunction()

function(ADD_VULKAN TARGET_NAME)
	target_include_directories(${TARGET_NAME} PRIVATE
		libs/vulkan
		libs/vk_video)

	add_subdirectory(libs/glfw)
	target_include_directories(${TARGET_NAME} PRIVATE ${glfw_SOURCE_DIR}/include)
	target_link_libraries(${TARGET_NAME} PRIVATE glfw)

	# Link the Vulkan loader library so core Vulkan functions are available without VK_NO_PROTOTYPES
	if (WIN32)
		find_library(VULKAN_LIB vulkan-1 HINTS "$ENV{VULKAN_SDK}/Lib")
		if (VULKAN_LIB)
			target_link_libraries(${TARGET_NAME} PRIVATE ${VULKAN_LIB})
		else()
			message(FATAL_ERROR "vulkan-1.lib not found. Set VULKAN_SDK environment variable.")
		endif()
	else()
		find_library(VULKAN_LIB vulkan HINTS "$ENV{VULKAN_SDK}/lib" /usr/lib /usr/local/lib)
		if (VULKAN_LIB)
			target_link_libraries(${TARGET_NAME} PRIVATE ${VULKAN_LIB})
		else()
			target_link_libraries(${TARGET_NAME} PRIVATE vulkan)
		endif()
	endif()
endfunction()
