
#pragma once

#include <string>
#include <unordered_map>
#include <map>
#include <vector>
#if !defined(__linux__)
	// Not on Linux: gcc-toolset-14 has no libstdc++exp, so std::stacktrace cannot link and
	// unsuck.hpp substitutes its own. Including this header would make `stacktrace`
	// ambiguous against the substitute wherever `using namespace std` is in effect.
	#include <stacktrace>
#endif

#include "OrbitControls.h"
#include "unsuck.hpp"

#include "glm/common.hpp"

#include "cuda.h"
#include "cuda_runtime.h"

#ifdef _WIN32
	#define NOMINMAX
	#include "windows.h"
#endif

using namespace std;

struct CURuntime{

	inline static CUdevice device;

	CURuntime(){

	}

	static int getNumSMs(){
		CUdevice device;
		int numSMs;
		cuCtxGetDevice(&device);
		cuDeviceGetAttribute(&numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);

		return numSMs;
	}

	// Free / total GPU bytes via cuMemGetInfo. Sets both to 0 on failure.
	static void getGPUMemory(uint64_t& freeBytes, uint64_t& totalBytes){
		size_t f = 0, t = 0;
		CUresult result = cuMemGetInfo(&f, &t);
		if(result != CUDA_SUCCESS){
			freeBytes = 0;
			totalBytes = 0;
			return;
		}
		freeBytes = (uint64_t)f;
		totalBytes = (uint64_t)t;
	}

	// Available / total system RAM in bytes. Sets both to 0 on failure.
	static void getHostMemory(uint64_t& availableBytes, uint64_t& totalBytes){
	#ifdef _WIN32
		MEMORYSTATUSEX status;
		status.dwLength = sizeof(status);
		if(GlobalMemoryStatusEx(&status)){
			availableBytes = (uint64_t)status.ullAvailPhys;
			totalBytes     = (uint64_t)status.ullTotalPhys;
			return;
		}
	#endif
		availableBytes = 0;
		totalBytes = 0;
	}


	static void assertCudaSuccess(CUresult result, stacktrace trace = stacktrace::current()){

		if(result == CUDA_SUCCESS) return;

		println("ERROR: CUDA result != CUDA_SUCCESS.");

		const char* name = nullptr;
		const char* desc = nullptr;
		cuGetErrorName(result, &name);
		cuGetErrorString(result, &desc);

		println(stderr, "CUDA error {} ({}): {}\n ",
			int(result),
			name ? name : "unknown",
			desc ? desc : "unknown");

		println("{}", trace);

		fflush(stdout);
		fflush(stderr);
		__debugbreak();

		exit(6123453456);
	}

};