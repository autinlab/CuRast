
#pragma once

#include <string>
#include <unordered_map>
#include <map>
#include <vector>
#include <fstream>
#include <limits>
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
	#elif defined(__linux__)
		// Under a batch scheduler the cgroup limit is what this process may actually use,
		// and it is usually far below the machine's RAM, so prefer it when present.
		auto readValue = [](const char* path, uint64_t& out) -> bool {
			std::ifstream in(path);
			if(!in) return false;
			std::string token;
			if(!(in >> token)) return false;
			if(token == "max") return false;          // cgroup v2: no limit set
			try { out = std::stoull(token); } catch(...) { return false; }
			return true;
		};

		uint64_t limit = 0, current = 0;
		bool haveCgroup =
			(readValue("/sys/fs/cgroup/memory.max", limit) && readValue("/sys/fs/cgroup/memory.current", current))
			|| (readValue("/sys/fs/cgroup/memory/memory.limit_in_bytes", limit)
				&& readValue("/sys/fs/cgroup/memory/memory.usage_in_bytes", current));

		// MemAvailable is the kernel's own estimate of what a new allocation can get -
		// the closest analogue to ullAvailPhys. MemFree alone badly understates it.
		uint64_t memTotal = 0, memAvailable = 0;
		std::ifstream meminfo("/proc/meminfo");
		std::string key;
		uint64_t valueKB = 0;
		while(meminfo >> key >> valueKB){
			if(key == "MemTotal:")           memTotal     = valueKB * 1024;
			else if(key == "MemAvailable:")  memAvailable = valueKB * 1024;
			meminfo.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			if(memTotal && memAvailable) break;
		}

		// A sane cgroup limit wins; otherwise fall back to the machine-wide figures.
		if(haveCgroup && limit > 0 && (memTotal == 0 || limit <= memTotal)){
			totalBytes     = limit;
			availableBytes = (limit > current) ? limit - current : 0;
			// Never claim more than the machine can actually hand out.
			if(memAvailable > 0 && availableBytes > memAvailable) availableBytes = memAvailable;
			return;
		}

		if(memTotal > 0){
			totalBytes     = memTotal;
			availableBytes = memAvailable;
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