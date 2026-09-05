#pragma once

#include "kernels/HostDeviceInterface.h"

struct CuRastSettings{
	static inline bool showBoundingBoxes = false;
	static inline bool enableEDL = true;
	static inline bool enableFrustumCulling = true;
	static inline bool hideGUI = false;

	static inline bool showKernelInfos = false;
	static inline bool showMemoryInfos = false;
	static inline bool showTimingInfos = false;
	static inline bool showStats = false;
	static inline bool showOverlay = true;
	static inline bool showInset = false;
	static inline bool showBenchmarking = false;
	static inline int supersamplingFactor = 1;

	static inline bool enableLinearInterpolation = true;
	static inline bool enableMipMapping = true;
	static inline float threshold = 0.0f;
	static inline bool freezeFrustum = false;
	static inline bool enableSSAO = false;
	static inline bool enableDiffuseLighting = false;
	static inline bool disableInstancing = false;
	static inline bool enableObjectPicking = false;
	static inline shared_ptr<string> requestScreenshot = nullptr; // Set to path of screenshot, or empty string for auto path
	static inline vec4 background = {1.0f, 1.0f, 1.0f, 1.0f};


	static inline DisplayAttribute displayAttribute = DisplayAttribute::TEXTURE;
	static inline bool showWireframe = false;

	// static inline uint32_t rasterizer = RASTERIZER_VULKAN_INDEXPULLING_INSTANCED;
	// static inline uint32_t rasterizer = RASTERIZER_VISBUFFER_INDEXED;
	static inline uint32_t rasterizer = RASTERIZER_VISBUFFER_INSTANCED;
	// static inline uint32_t rasterizer = RASTERIZER_VULKAN_INDEXPULLING_VISBUFFER;
	// static inline uint32_t rasterizer = RASTERIZER_VISBUFFER_CLUSTERS;

	static inline bool benchmark_load_sponza = false;
	static inline bool benchmark_load_lantern = false;

	// ----- Sphere LOD ---------------------------------------------------------
	// When enabled, sphereLodConfig is uploaded to kernel_draw_spheres / resolve.
	// Default table is built from a "scene scale" estimate at first load.
	static inline bool             enableSphereLOD = true;
	static inline float            sphereLodSceneScale = 0.0f; // 0 = uninitialised; populated on first load
	static inline SphereLodConfig  sphereLodConfig = []{
		SphereLodConfig c{};
		c.numLevels = 4;
		// Bands are normalised to scene radius (rescaled at render time).
		// Strides form a clean chain {1, 4, 16, 64} so divisibility cascades — atom i
		// divisible by 64 is also divisible by 16 and 4. This makes the partition rule
		// in the kernel a single check (skip if divisible by next-coarser active stride).
		// scale = stride^(1/3) keeps projected screen area roughly constant.
		c.levels[0] = { 0.0f,   1.5f,  0.20f, 1.000f, 1,  {} };
		c.levels[1] = { 1.0f,   4.0f,  0.50f, 1.587f, 4,  {} };
		c.levels[2] = { 3.0f,  12.0f,  1.50f, 2.520f, 16, {} };
		c.levels[3] = { 9.0f, 100.0f,  4.00f, 4.000f, 64, {} };
		return c;
	}();

	// ----- Multiscale SSAO ----------------------------------------------------
	static inline bool  enableMultiscaleSSAO = true;
	static inline int   ssaoLevels           = 4;     // 1..4
	// Per-level sample radius expressed as a fraction of the centre pixel's view-space
	// depth. The previous defaults (0.02..0.7) were tuned for object-space scenes; for
	// molecular assemblies they were 10–100× too large and sampled across the whole
	// structure. New defaults span 4 octaves from atom-cavity scale (~0.001) up to ~5%
	// of depth, which is appropriate for atomic / protein / assembly geometry alike.
	static inline float ssaoLevelRadius[4]   = { 0.0015f, 0.005f, 0.015f, 0.050f };
	static inline float ssaoLevelBias[4]     = { 1.0f,    1.0f,   1.0f,   1.0f   };
	// A single multiplier the user can dial to retune for any scene without touching
	// per-level radii (e.g. set to 0.5 for tight cavities, 4.0 for huge enclosures).
	static inline float ssaoRadiusScale      = 1.0f;
	static inline float ssaoIntensity        = 1.1f;
	// Below ~12 samples the per-pixel noise is too coarse for the 7×7 bilateral blur
	// to fully resolve. 16 is a good default; molstar's default is 32.
	static inline int   ssaoSamplesPerLevel  = 16;    // total samples = ssaoSamplesPerLevel * ssaoLevels
};

// Enabling this makes CuRast allocate memory for geometry with the Vulkan API instead of CUDA.
// - Needs to be enabled to render things in Vulkan.
// - It's still shared to CUDA because LargeGlbLoader.h uses CUDA for streaming mesh data to GPU. 
// - It's off by default because allocating in Vulkan vs. CUDA has different performance implications.
// - From observations, we assume that the Vulkan buffer implicitly enables compression. 
//   This makes some scenarios faster (e.g. uncompressed geometry) but others slower (resolve).
// - Explicitly enabling compressed CUDA buffers seems to equalize the performance.
// - For benchmarking, we enable it for Vulkan measurements and disable it for CUDA measuerements.
// #define USE_VULKAN_SHARED_MEMORY