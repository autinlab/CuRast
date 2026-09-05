#pragma once 

#include "cuda.h"
#include "VulkanCudaSharedMemory.h"

// Usage flags are a strong contended for dumbest things in Vulkan. Just give me device memory...
constexpr VkBufferUsageFlags DEFAULT_USAGE_FLAGS = 
	VK_BUFFER_USAGE_STORAGE_BUFFER_BIT 
	| VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT 
	| VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT 
	| VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT;

struct MemoryManager{

	struct Allocation {
		string label;
		CUdeviceptr cptr;
		int64_t size;
	};

	inline static mutex mtx;
	inline static vector<Allocation> allocations;
	inline static vector<VKBuffer*> allocations_vulkan;
	inline static vector<VulkanCudaSharedMemory*> vulkanCudaShared;
	inline static vector<CudaVirtualMemory*> cudaVirtual;

	inline static VulkanCudaSharedMemory* allocVulkanCudaShared(uint64_t virtualCapacity, string label = "none"){

		VulkanCudaSharedMemory* memory = VulkanCudaSharedMemory::create(virtualCapacity, label);
		vulkanCudaShared.push_back(memory);

		return memory;
	}

	inline static CudaVirtualMemory* allocVirtualCuda(uint64_t virtualCapacity, string label = "none"){

		CudaVirtualMemory* memory = CudaVirtualMemory::create(virtualCapacity, label);
		cudaVirtual.push_back(memory);

		return memory;
	}

	// Allocates `size` bytes of GPU memory. Returns 0 if the allocation cannot
	// safely fit in free VRAM (with ~64 MB headroom for driver overhead) or if
	// cuMemAlloc itself fails. Callers must check the returned pointer.
	inline static CUdeviceptr alloc(int64_t size, string label){
		if(size <= 0) return 0;

		size_t free = 0, total = 0;
		CUresult info = cuMemGetInfo(&free, &total);
		if(info == CUDA_SUCCESS){
			constexpr uint64_t HEADROOM_BYTES = 64ull * 1024ull * 1024ull;
			if((uint64_t)size + HEADROOM_BYTES > (uint64_t)free){
				println("MemoryManager::alloc REFUSED '{}': requested {} MB, only {} MB free",
					label, size >> 20, (uint64_t)free >> 20);
				return 0;
			}
		}

		CUdeviceptr cptr = 0;
		auto result = cuMemAlloc(&cptr, size);
		if(result != CUDA_SUCCESS){
			const char* name = nullptr;
			const char* desc = nullptr;
			cuGetErrorName(result, &name);
			cuGetErrorString(result, &desc);
			println("MemoryManager::alloc FAILED '{}' ({} MB): {} ({})",
				label, size >> 20,
				name ? name : "unknown",
				desc ? desc : "unknown");
			return 0;
		}

		lock_guard<mutex> lock(mtx);
		Allocation entry = { label, cptr, size};
		allocations.push_back(entry);

		return cptr;
	}

	

	inline static VKBuffer* allocVulkan(
		int64_t size, 
		string label,
		VkBufferUsageFlags usageFlags = DEFAULT_USAGE_FLAGS, 
		VkMemoryPropertyFlags memoryPropertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
	){
		VKBuffer* buffer = new VKBuffer(size, usageFlags, memoryPropertyFlags);
		buffer->label = label;

		allocations_vulkan.push_back(buffer);

		return buffer;
	}

	static void free(CUdeviceptr cptr) {
		if (cptr == 0) {
			println("WARNING: attempted to CURuntime::free a null ptr. Already freed?");
			return;
		}

		lock_guard<mutex> lock(mtx);

		int index = -1;
		for(int i = 0; i < allocations.size(); i++){
			if(allocations[i].cptr == cptr){
				index = i;
				cuMemFree(cptr);
			}
		}

		if(index != -1){
			allocations.erase(allocations.begin() + index);
		}
	}

	static int64_t getByteSize(CUdeviceptr cptr){
		for(int i = 0; i < allocations.size(); i++){
			if(allocations[i].cptr == cptr){
				return allocations[i].size;
			}
		}

		return 0;
	}

	static int64_t getTotalAllocatedMemory(){

		int64_t bytes = 0;

		for(int i = 0; i < allocations.size(); i++){
			bytes += allocations[i].size;
		}

		for(auto memory : allocations_vulkan){
			bytes += memory->size;
		}

		for(auto memory : vulkanCudaShared){
			bytes += memory->comitted;
		}

		for(auto memory : cudaVirtual){
			bytes += memory->comitted;
		}

		return bytes;
	}

};