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
	static inline bool enableSSAO = true;
	static inline bool enableDiffuseLighting = false;
	static inline bool enableTranslucency = false;
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
	// Two levels by default: a CLOSE level that resolves the cavities between
	// neighbouring atoms, and a FAR level supplying the broad enclosure / skylight
	// cue that makes a large assembly read as solid. Cost is the SUM of the active
	// levels' sample counts, so two levels is materially cheaper than four.
	static inline int   ssaoLevels           = 2;     // 1..4  (0 = close, 1 = far)
	// Per-level sample radius expressed as a fraction of the centre pixel's view-space
	// depth. The previous defaults (0.02..0.7) were tuned for object-space scenes; for
	// molecular assemblies they were 10–100× too large and sampled across the whole
	// structure. New defaults span 4 octaves from atom-cavity scale (~0.001) up to ~5%
	// of depth, which is appropriate for atomic / protein / assembly geometry alike.
	// [0] = close, [1] = far. Slots 2-3 keep the old 4-level values, so setting
	// ssaoLevels back to 4 restores the previous continuous scale ramp.
	static inline float ssaoLevelRadius[4]   = { 0.004f, 0.040f, 0.015f, 0.050f };
	static inline float ssaoLevelBias[4]     = { 1.0f,   1.0f,   1.0f,   1.0f   };
	// A single multiplier the user can dial to retune for any scene without touching
	// per-level radii (e.g. set to 0.5 for tight cavities, 4.0 for huge enclosures).
	static inline float ssaoRadiusScale      = 1.0f;
	static inline float ssaoIntensity        = 1.1f;
	// Samples are budgeted PER LEVEL. The close level carries the detail the eye reads
	// as shape, so it takes most of the budget; the far level is low-frequency and looks
	// the same with far fewer taps once the 7x7 bilateral blur has run. Below ~12 samples
	// a level goes visibly noisy (molstar uses 32 for its single level).
	// Total depth taps per pixel = sum over the active levels.
	static inline int   ssaoSamplesPerLevel[4] = { 24, 8, 16, 16 };

	// ----- GTAO ---------------------------------------------------------------
	// Ground Truth Ambient Occlusion (Jimenez et al. 2016). Horizon search plus an
	// analytic per-slice visibility integral, instead of the hemisphere point
	// sampling above. Kept side by side with the old path (aoMode) so the two can
	// be compared directly rather than swapped blind.
	//
	// The close/far LEVEL structure does not carry over: GTAO has one search radius
	// and gets its quality from slices x steps, not from combining scales.
	static inline int   aoMode        = 1;      // 0 = legacy hemisphere, 1 = GTAO

	// Search radius as a fraction of the pixel's depth (screen footprint is then
	// constant with distance). The large-scale "enclosure" cue that makes a packed
	// cell read as a sphere lives in this number, not in the slice count -- the old
	// far level probed 0.04 of depth, which is far too short for it.
	static inline float gtaoRadius    = 0.12f;
	static inline int   gtaoSlices    = 3;      // hemisphere slices; each is exact, so few are needed
	static inline int   gtaoSteps     = 8;      // horizon march steps per side
	static inline float gtaoIntensity = 1.0f;
	// Multiplies the search radius for the distance falloff. Below 1 makes occluders
	// fade before the end of the march; above 1 lets far geometry keep occluding.
	static inline float gtaoThickness = 1.0f;
	// Debug: render the AO buffer as greyscale instead of the shaded image.
	static inline bool  aoDebugView   = false;
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