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
	// Diagnostic view: 0 = off, 1 = AO buffer, 2 = normal buffer, 3 = impostor mask
	// (green = analytic ray-sphere hit, red = sub-pixel fallback).
	static inline int   debugView    = 0;

	// Flat sphere shading: albedo only, no directional term. AO and the halo still
	// apply, so shape comes from occlusion and outlines -- the illustrative look.
	static inline bool  flatSpheres  = false;
	// Flat shading drops the directional term, which carries much of the average
	// brightness, so it is lifted back to roughly the lit mode's level.
	static inline float flatBrightness = 2.2f;

	// Environment framing. Rotation spins the map about world +Z. Background widen
	// spreads the background sample angle so more of the panorama is visible, which
	// makes its features read smaller relative to the model; lighting is unaffected.
	static inline float envRotation  = 0.0f;   // degrees
	static inline float envBgWiden   = 3.0f;
	// With this on, turning env lighting off gives a uniform ambient instead of the
	// baked studio SH, so the toggle actually shows what the environment contributes.
	static inline bool  envNeutralWhenOff = true;

	// ----- Environment lighting -----------------------------------------------
	// Path to an .exr or .hdr equirectangular environment map. Empty = use the
	// studio coefficients baked into resolve.cu. Set envMapReload to have the
	// renderer pick up a new path (loading + SH projection happen on the host).
	static inline string envMapPath    = "";
	static inline bool   envMapReload  = false;
	static inline float  envExposure   = 0.85f;
	// Master switch. Off = fall back to the studio coefficients baked into resolve.cu.
	static inline bool   envEnabled    = true;
	// Lighting the scene with an environment and showing that environment are separate
	// choices; molecular figures usually want the first without the second.
	static inline bool   envShowBackground = false;

	// ----- AO response curve --------------------------------------------------
	// shade = aoFloor + (1 - aoFloor) * ao^aoPower. Replaces the old fixed
	// `ao * 0.4 + 0.6`, which compressed AO into [0.6, 1.0] and discarded most of it.
	// ----- Camera -------------------------------------------------------------
	// Orthographic removes perspective foreshortening, which is what you usually want
	// for a figure of a large assembly: the far side of the cell is drawn at the same
	// scale as the near side, so the silhouette reads as the true cross-section.
	static inline bool  orthographic = false;
	// Multiplies the orthographic half-height, which is otherwise derived from the
	// orbit distance and fov so that toggling modes preserves framing.
	static inline float orthoZoom    = 1.0f;

	// ----- QuteMol-style halo --------------------------------------------------
	// Depth-aware dark glow around silhouettes. QuteMol draws an enlarged billboard per
	// atom; this is the screen-space equivalent over the depth buffer, since per-atom
	// billboards are not an option at 158.9M atoms.
	static inline bool  haloEnabled   = false;
	static inline float haloSize      = 0.012f;  // fraction of the smaller viewport side
	static inline float haloStrength  = 0.7f;
	static inline float haloColor     = 0.0f;    // 0 = black, 1 = white
	static inline float haloDepthFull = 0.06f;   // depth gap for a fully opaque halo
	static inline int   haloDirs      = 12;
	static inline int   haloSteps     = 4;

	// ----- QuteMol-style baked per-atom AO --------------------------------------
	// Object-space, view-independent, one byte per atom. Baked once at load; the cost
	// per frame is a single byte fetch. Supplies the large-scale enclosure that a
	// screen-space method cannot see when ~90 atoms share a pixel.
	static inline bool  atomAOEnabled    = false;
	static inline int   atomAODirections = 64;
	static inline int   atomAOResolution = 2048;
	static inline float atomAOIntensity  = 1.0f;
	static inline float atomAOMaxRadius  = 2.0f;   // world units, used as the depth slack

	static inline float aoFloor = 0.25f;
	static inline float aoPower = 1.0f;
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