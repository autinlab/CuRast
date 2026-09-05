#pragma once

#include <string>
#include <unordered_map>
#include <print>
#if !defined(__linux__)
	// Not on Linux: gcc-toolset-14 has no libstdc++exp, so std::stacktrace cannot link and
	// unsuck.hpp substitutes its own. Including this header would make `stacktrace`
	// ambiguous against the substitute wherever `using namespace std` is in effect.
	#include <stacktrace>
#endif

#include "unsuck.hpp"

#include "nvrtc.h"
#include <nvJitLink.h>
#include <cmath>
#include "cuda.h"
#include "Timer.h"
#include "CURuntime.h"

using std::string;

using namespace std;

#define NVJITLINK_SAFE_CALL(h,x)                                  \
do {                                                              \
   nvJitLinkResult result = x;                                    \
   if (result != NVJITLINK_SUCCESS) {                             \
      std::cerr << "\nerror: " #x " failed with error "           \
                << result << '\n';                                \
      size_t lsize;                                               \
      result = nvJitLinkGetErrorLogSize(h, &lsize);               \
      if (result == NVJITLINK_SUCCESS && lsize > 0) {             \
         char *log = (char*)malloc(lsize);                        \
         result = nvJitLinkGetErrorLog(h, log);                   \
         if (result == NVJITLINK_SUCCESS) {                       \
            std::cerr << "error: " << log << '\n';                \
            free(log);                                            \
         }                                                        \
      }                                                           \
      exit(1);                                                    \
   } else {                                                       \
      size_t lsize;                                               \
      result = nvJitLinkGetInfoLogSize(h, &lsize);                \
      if (result == NVJITLINK_SUCCESS && lsize > 0) {             \
         char *log = (char*)malloc(lsize);                        \
         result = nvJitLinkGetInfoLog(h, log);                    \
         if (result == NVJITLINK_SUCCESS) {                       \
            std::cerr << "info: " << log << '\n';                 \
            free(log);                                            \
         }                                                        \
      }                                                           \
      break;                                                      \
   }                                                              \
} while(0)


struct OptionalLaunchSettings{
	uint32_t gridsize = 0;
	uint32_t blocksize = 0;
	vector<void*> args;
	bool measureDuration = false;
	CUstream stream;
};

struct CudaModule{

	static void cu_checked(CUresult result){
		if(result != CUDA_SUCCESS){
			cout << "cuda error code: " << result << endl;
		}
	};

	string path = "";
	string name = "";
	bool compiled = false;
	bool success = false;
	bool useLTO = true;
	vector<string> defines;

	size_t ptxSize = 0;
	char* ptx = nullptr;

	size_t ltoirSize = 0;
	char* ltoir = nullptr;

	size_t cubinSize = 0;
	char* cubin = nullptr;

	CudaModule(string path, string name, bool useLTO = true){
		this->path = path;
		this->name = name;
		this->useLTO = useLTO;
	}

	void compile(){
		auto tStart = now();

		cout << "================================================================================" << endl;
		cout << "=== COMPILING: " << fs::path(path).filename().string() << endl;
		cout << "================================================================================" << endl;

		success = false;

		string dir = fs::path(path).parent_path().string();

		// Prefer the CUDA toolkit that was used to build this binary (CUDA_DEVRTLIB is a
		// compile-time constant set by CMake).  The CUDA_PATH environment variable can
		// point to a different (e.g. older) version, which would cause nvrtcCompileProgram
		// to fail with incompatible headers.  Fall back to CUDA_PATH only when the build-
		// time path no longer exists on disk.
		string cuda_path_build = fs::path(CUDA_DEVRTLIB).parent_path().parent_path().parent_path().string();
		const char* cuda_path_env = std::getenv("CUDA_PATH");
		string cuda_path = fs::exists(cuda_path_build)
			? cuda_path_build
			: (cuda_path_env ? cuda_path_env : cuda_path_build);
		string optInclude = std::format("-I {}", dir).c_str();
		string cuda_include = std::format("-I {}/include", cuda_path);
		string cudastd_include = std::format("-I {}/include/cccl/cuda/std", cuda_path);
		string cccl_include = std::format("-I {}/include/cccl", cuda_path);
		
		println("cuda_path: {}", cuda_path);
		println("cudastd_include: {}", cudastd_include);
		println("cccl_include: {}", cccl_include);

		CUdevice device;
		cuCtxGetDevice(&device);

		int major = 0;
		int minor = 0;
		cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
		cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);

		// LTO requires virtual arch (compute_XX); direct cubin needs physical arch (sm_XX)
		string arch = useLTO
			? format("--gpu-architecture=compute_{}{}", major, minor)
			: format("--gpu-architecture=sm_{}{}", major, minor);

		nvrtcProgram prog;
		string source = readFile(path);
		nvrtcCreateProgram(&prog, source.c_str(), name.c_str(), 0, NULL, NULL);
		
		vector<const char*> opts = {
			arch.c_str(),
			"--use_fast_math",
			"--extra-device-vectorization",
			"-lineinfo",
			cudastd_include.c_str(),
			cuda_include.c_str(),
			cccl_include.c_str(),
			optInclude.c_str(),
			"-I ./",
			"-I ./include",
			"-I ./libs",
			"-default-device",
			"--std=c++20",
			"--disable-warnings",
			"--split-compile=0",
		};
		if(useLTO){
			opts.push_back("--relocatable-device-code=true");
			opts.push_back("--dlink-time-opt");
		}

		// need to prepend "-D " to the defines, and need to keep the variables holding the c_str alive until compilation is done
		vector<string> defineArgs;
		for(string& define : defines){
			string str = format("-D {}", define);
			defineArgs.push_back(str);
			opts.push_back(defineArgs[defineArgs.size() - 1].c_str());
		}

		println("nvrtcCompileProgram arguments: ");
		for(auto opt : opts){
			println("    {}", opt);
		}
		println("======");

		nvrtcResult res = nvrtcCompileProgram(prog, int(opts.size()), opts.data());
		
		if (res != NVRTC_SUCCESS)
		{
			size_t logSize;
			nvrtcGetProgramLogSize(prog, &logSize);
			char* log = new char[logSize];
			nvrtcGetProgramLog(prog, log);
			println("Program Log: {}", log);

			delete[] log;

			if(res != NVRTC_SUCCESS && ltoir != nullptr){
				return;
			}else if(res != NVRTC_SUCCESS && ltoir == nullptr){
				println("failed gto compile {}. {}:{}", path, __FILE__, __LINE__);
				exit(123);
			}
		}

		if(useLTO){
			nvrtcGetLTOIRSize(prog, &ltoirSize);
			ltoir = new char[ltoirSize];
			nvrtcGetLTOIR(prog, ltoir);
			cout << format("compiled ltoir. size: {} byte \n", ltoirSize);
		}else{
			// Generate cubin directly — avoids LTO overhead and PTX JIT version issues
			nvrtcGetCUBINSize(prog, &cubinSize);
			cubin = new char[cubinSize];
			nvrtcGetCUBIN(prog, cubin);
			cout << format("compiled cubin. size: {} byte \n", cubinSize);
		}

		nvrtcDestroyProgram(&prog);

		compiled = true;
		success = true;

		printElapsedTime("compile " + name, tStart);
	}

};


struct CudaModularProgram{

	struct CudaModularProgramArgs{
		vector<string> modules;
		vector<string> kernels;
		vector<string> defines;
		bool useLTO = true;
	};

	static void cu_checked(CUresult result){
		if(result != CUDA_SUCCESS){
			cout << "cuda error code: " << result << endl;
		}
	};

	vector<CudaModule*> modules;
	unordered_map<string, CUdeviceptr> cachedGlobals;

	CUmodule mod;
	// CUfunction kernel = nullptr;
	void* cubin = nullptr;
	size_t cubinSize;

	vector<std::function<void(void)>> compileCallbacks;

	vector<string> kernelNames;
	unordered_map<string, CUfunction> kernels;
	vector<string> defines;

	unordered_map<string, CUevent> events_launch_start;
	unordered_map<string, CUevent> events_launch_end;

	int MAX_LAUNCH_DURATIONS = 50;
	unordered_map<string, vector<float>> last_launch_durations;
	unordered_map<string, int> launches_per_frame;

	inline static vector<CudaModularProgram*> instances;
	mutex mtx_instances;

	CudaModularProgram(){
		lock_guard<mutex> lock(mtx_instances);
		instances.push_back(this);
	}

	CudaModularProgram(vector<string> modules){
		construct({.modules = modules,});

		lock_guard<mutex> lock(mtx_instances);
		instances.push_back(this);
	}

	CudaModularProgram(CudaModularProgramArgs args){
		this->defines = args.defines;

		construct(args);

		lock_guard<mutex> lock(mtx_instances);
		instances.push_back(this);
	}

	static CudaModularProgram* fromCubin(void* cubin, int64_t size){
		CudaModularProgram* program = new CudaModularProgram();

		program->cubin = cubin;
		program->cubinSize = size;

		cu_checked(cuModuleLoadData(&program->mod, cubin));

		{ // Retrieve Kernels
			uint32_t count = 0;
			cuModuleGetFunctionCount(&count, program->mod);

			vector<CUfunction> functions(count);
			cuModuleEnumerateFunctions(functions.data(), count, program->mod);

			program->kernelNames.clear();

			for(CUfunction function : functions){
				const char* name;

				cuFuncGetName(&name, function);

				string strName = name;

				program->kernelNames.push_back(strName);
				program->kernels[strName] = function;

				CUevent event_start;
				CUevent event_end;
				cuEventCreate(&event_start, CU_EVENT_DEFAULT);
				cuEventCreate(&event_end, CU_EVENT_DEFAULT);

				program->events_launch_start[strName] = event_start;
				program->events_launch_end[strName] = event_end;
			}
		}

		return program;
	}

	void construct(CudaModularProgramArgs args){
		vector<string> modulePaths = args.modules;

		for(auto modulePath : modulePaths){

			string moduleName = fs::path(modulePath).filename().string();
			CudaModule* module = new CudaModule(modulePath, moduleName, args.useLTO);
			module->defines = args.defines;

			module->compile();

			monitorFile(modulePath, [&, module]() {
				module->compile();
				link();
			});

			modules.push_back(module);
		}

		link();
	}

	void link(){

		cout << "================================================================================" << endl;
		cout << "=== LINKING" << endl;
		cout << "================================================================================" << endl;
		
		auto tStart = now();

		for(auto module : modules){
			if(!module->success){
				return;
			}
		}

		//float walltime;
		constexpr uint32_t v_optimization_level = 1;
		constexpr uint32_t logSize = 8192;
		//char info_log[logSize];
		//char error_log[logSize];

		//CUlinkState linkState;

		CUdevice cuDevice;
		cuDeviceGet(&cuDevice, 0);

		int major = 0;
		int minor = 0;
		cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
		cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);

		 int arch = major * 10 + minor;
		//int arch = 86;
		string strArch = std::format("-arch=sm_{}", arch);

		bool anyLTO = false;
		for(auto module : modules) if(module->useLTO) anyLTO = true;

		if(anyLTO){
			// Full LTO path via nvJitLink
			vector<const char*> lopts = {
				"-lto",
				strArch.c_str(),
				"-O3",
				"-optimize-unused-variables",
				"-split-compile=0",
			};

			nvJitLinkHandle handle;
			nvJitLinkCreate(&handle, lopts.size(), lopts.data());

			for(auto module : modules){
				NVJITLINK_SAFE_CALL(handle, nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, (void *)module->ltoir, module->ltoirSize, module->name.c_str()));
			}
			NVJITLINK_SAFE_CALL(handle, nvJitLinkAddFile(handle, NVJITLINK_INPUT_ANY, CUDA_DEVRTLIB));

			NVJITLINK_SAFE_CALL(handle, nvJitLinkComplete(handle));
			NVJITLINK_SAFE_CALL(handle, nvJitLinkGetLinkedCubinSize(handle, &cubinSize));

			if(cubin){ free(cubin); cubin = nullptr; }
			cubin = malloc(cubinSize);
			NVJITLINK_SAFE_CALL(handle, nvJitLinkGetLinkedCubin(handle, cubin));
			NVJITLINK_SAFE_CALL(handle, nvJitLinkDestroy(&handle));

			cu_checked(cuModuleLoadData(&mod, cubin));
		} else {
			// Non-LTO path: NVRTC generated cubin directly — load it straight into CUDA
			// without any JIT or LTO step. Instantaneous and avoids PTX version issues.
			for(auto module : modules){
				cu_checked(cuModuleLoadData(&mod, module->cubin));
				break; // single-module programs only
			}
		}

		{ // Retrieve Kernels
			uint32_t count = 0;
			cuModuleGetFunctionCount(&count, mod);

			vector<CUfunction> functions(count);
			cuModuleEnumerateFunctions(functions.data(), count, mod);

			kernelNames.clear();

			for(CUfunction function : functions){
				const char* name;

				cuFuncGetName(&name, function);

				string strName = name;

				kernelNames.push_back(strName);
				kernels[strName] = function;

				CUevent event_start;
				CUevent event_end;
				cuEventCreate(&event_start, CU_EVENT_DEFAULT);
				cuEventCreate(&event_end, CU_EVENT_DEFAULT);

				events_launch_start[strName] = event_start;
				events_launch_end[strName] = event_end;
			}
		}
		
		cachedGlobals.clear();

		for(auto& callback : compileCallbacks){
			callback();
		}

		printElapsedTime("link duration: ", tStart);


	}

	CUdeviceptr getGlobalsPointer(string name){

		if(!cachedGlobals.contains(name)){
			CUdeviceptr cptr = 0;
			size_t bytes = 0;
			auto result = cuModuleGetGlobal(&cptr, &bytes, mod, name.c_str());

			if(result == CUDA_SUCCESS){
				cachedGlobals[name] = cptr;
			}else{
				println("did not find global variable {}", name);
				return 0;
			}
		}

		return cachedGlobals[name];
	}

	void onCompile(std::function<void(void)> callback){
		compileCallbacks.push_back(callback);
	}

	void addLaunchDuration(string kernelName, float duration){
		last_launch_durations[kernelName].resize(MAX_LAUNCH_DURATIONS);

		last_launch_durations[kernelName][0] += duration;
		
		launches_per_frame[kernelName]++;
	}

	void launch(string kernelName, vector<void*> args, OptionalLaunchSettings launchArgs = {}){
		void** _args = &args[0];

		this->launch(kernelName, _args, launchArgs);
	}

	void launch(string kernelName, void** args, OptionalLaunchSettings launchArgs){
		auto custart = Timer::recordCudaTimestamp();

		auto res_launch = cuLaunchKernel(kernels[kernelName],
			launchArgs.gridsize, 1, 1,
			launchArgs.blocksize, 1, 1,
			0, launchArgs.stream, args, nullptr);


		if (res_launch != CUDA_SUCCESS) {
			const char* str;
			cuGetErrorString(res_launch, &str);
			printf("error: %s \n", str);
			cout << __FILE__ << " - " << __LINE__ << endl;
			println("kernel: {}", kernelName);
		}

		Timer::recordDuration(kernelName, custart, Timer::recordCudaTimestamp());
	}

	void launch(string kernelName, vector<void*> args, int count, CUstream stream = 0){
		if(count == 0) return;

		void** _args = &args[0];

		this->launch(kernelName, _args, count, stream);
	}

	void launch(string kernelName, void** args, int count, CUstream stream = 0){

		if (count == 0){
			// last_launch_durations[kernelName] += 0.0f;
			// addLaunchDuration(kernelName, duration);
			return;
		}

		CUevent event_start = events_launch_start[kernelName];
		CUevent event_end   = events_launch_end[kernelName];

		uint32_t blockSize = 256;
		uint32_t gridSize = (count + blockSize - 1) / blockSize;

		auto custart = Timer::recordCudaTimestamp();

		auto res_launch = cuLaunchKernel(kernels[kernelName],
			gridSize, 1, 1,
			blockSize, 1, 1,
			0, stream, args, nullptr);

		CURuntime::assertCudaSuccess(res_launch);

		Timer::recordDuration(kernelName, custart, Timer::recordCudaTimestamp());
	}

	void launch2D(string kernelName, void** args, int width, int height, CUstream stream = 0){

		if (width == 0 || height == 0){
			return;
		}

		CUevent event_start = events_launch_start[kernelName];
		CUevent event_end   = events_launch_end[kernelName];

		uint32_t blockSize = 8;
		uint32_t gridSizeX = (width + blockSize - 1) / blockSize;
		uint32_t gridSizeY = (height + blockSize - 1) / blockSize;

		auto custart = Timer::recordCudaTimestamp();

		auto res_launch = cuLaunchKernel(kernels[kernelName],
			gridSizeX, gridSizeY, 1,
			blockSize, blockSize, 1,
			0, stream, args, nullptr);

		if (res_launch != CUDA_SUCCESS) {
			const char* str;
			cuGetErrorString(res_launch, &str);
			printf("error %d, %s \n", int(res_launch), str);
			println("{} - {}", __FILE__, __LINE__);
			println("failed to launch kernel \"{}\". gridSize: {} x {}", kernelName, gridSizeX, gridSizeY);
			println("{}", stacktrace::current());

			exit(42415);
		}

		Timer::recordDuration(kernelName, custart, Timer::recordCudaTimestamp());
	}

	void launchCooperative(string kernelName, vector<void*> args, OptionalLaunchSettings launchArgs = {}){
		void** _args = &args[0];

		this->launchCooperative(kernelName, _args, launchArgs);
	}

	void launchCooperative(string kernelName, void** args, OptionalLaunchSettings launchArgs = {}){

		auto custart = Timer::recordCudaTimestamp();

		CUdevice device;
		int numSMs;
		cuCtxGetDevice(&device);
		cuDeviceGetAttribute(&numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);

		
		int blockSize = launchArgs.blocksize > 0 ? launchArgs.blocksize : 128;

		int numBlocks;
		CUresult resultcode = cuOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernels[kernelName], blockSize, 0);
		CURuntime::assertCudaSuccess(resultcode);

		numBlocks *= numSMs;
		
		//numGroups = 100;
		// make sure at least 10 workgroups are spawned)
		numBlocks = std::clamp(numBlocks, 10, 100'000);

		auto kernel = this->kernels[kernelName];
		auto res_launch = cuLaunchCooperativeKernel(kernel,
			numBlocks, 1, 1,
			blockSize, 1, 1,
			0, launchArgs.stream, args);
		CURuntime::assertCudaSuccess(res_launch);

		Timer::recordDuration(kernelName, custart, Timer::recordCudaTimestamp());
	}

	void clearTimings(){
		for(auto& [key, value] : last_launch_durations){
			for(size_t i = value.size() - 1; i > 0; i--){
				value[i] = value[i - 1];
			}
			value[0] = 0.0f;
		}

		for(auto& [key, value] : launches_per_frame){
			value = 0;
		}
		
	}

	float getAvgTiming(string kernelName){
		if(last_launch_durations.find(kernelName) != last_launch_durations.end()){
			float sum = 0.0f;
			for(float value : last_launch_durations[kernelName]){
				sum += value;
			}

			float avg = sum / float(last_launch_durations[kernelName].size());

			return avg;
		}else{
			return 0.0f;
		}
	}

};