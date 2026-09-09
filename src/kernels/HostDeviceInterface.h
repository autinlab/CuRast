
#pragma once

#include <cmath>
#include <bit>


#ifdef __CUDA_ARCH__
	#include <math_constants.h>
	constexpr float Infinity = __builtin_bit_cast(float, 0x7f800000);
	#include "../jpeg/HashMap.cuh"
	#include "../jpeg/JptInterface.cuh"

	// === required by GLM ===
	#define GLM_FORCE_CUDA
	#define CUDA_VERSION 12000
	namespace std {
		using size_t = ::size_t;
	};
	// =======================
	#include "./glm/glm/glm.hpp"

#else
	constexpr float Infinity = __builtin_bit_cast(float, 0x7f800000);
	#include "./jpeg/HashMap.cuh"
#endif


 using glm::vec2;
 using glm::vec3;
 using glm::vec4;
 using glm::ivec2;
 using glm::ivec3;
 using glm::ivec4;
 using glm::mat4;

// constexpr uint32_t BACKGROUND_COLOR = 0xff887766;
constexpr uint32_t BACKGROUND_COLOR = 0xffffffff;
constexpr uint64_t DEFAULT_PIXEL = (uint64_t(0x7f800000) << 32) | BACKGROUND_COLOR;

constexpr uint32_t RASTERIZER_BASIC = 0;
constexpr uint32_t RASTERIZER_VISBUFFER = 1;
constexpr uint32_t RASTERIZER_OPENGL = 2;
constexpr uint32_t RASTERIZER_VISBUFFER2 = 3;
constexpr uint32_t RASTERIZER_VISBUFFER_SERIALIZED = 4;
constexpr uint32_t RASTERIZER_VISBUFFER_SCANLINE = 5;
constexpr uint32_t RASTERIZER_VISBUFFER_16BYTE_ALIGNED = 6;
constexpr uint32_t RASTERIZER_FORWARD = 7;
constexpr uint32_t RASTERIZER_VISBUFFER_INDEXED = 8;
//constexpr uint32_t RASTERIZER_VISBUFFER_CLUSTERS = 9;
constexpr uint32_t RASTERIZER_VISBUFFER_INSTANCED = 10;
constexpr uint32_t RASTERIZER_VULKAN = 11;
constexpr uint32_t RASTERIZER_VULKAN_INDEXPULLING = 12;
constexpr uint32_t RASTERIZER_VULKAN_INDEXED_DRAW = 13;
constexpr uint32_t RASTERIZER_VULKAN_INDEXPULLING_VISBUFFER = 14;
constexpr uint32_t RASTERIZER_VULKAN_INDEXPULLING_INSTANCED = 15;

struct Box3 {
	vec3 min = { Infinity, Infinity, Infinity };
	vec3 max = { -Infinity, -Infinity, -Infinity };

	bool isDefault() {
		return min.x == Infinity && min.y == Infinity && min.z == Infinity && max.x == -Infinity && max.y == -Infinity && max.z == -Infinity;
	}

	bool isEqual(Box3 box, float epsilon) {
		float diff_min = length(box.min - min);
		float diff_max = length(box.max - max);

		if (diff_min >= epsilon) return false;
		if (diff_max >= epsilon) return false;

		return true;
	}

	void extend(vec3 v){
		this->min.x = ::min(this->min.x, v.x);
		this->min.y = ::min(this->min.y, v.y);
		this->min.z = ::min(this->min.z, v.z);
		this->max.x = ::max(this->max.x, v.x);
		this->max.y = ::max(this->max.y, v.y);
		this->max.z = ::max(this->max.z, v.z);
	}

	Box3 transform(mat4 matrix){

		Box3 result;

		vec3 corners[8] = {
			{min.x, min.y, min.z},
			{max.x, min.y, min.z},
			{min.x, max.y, min.z},
			{max.x, max.y, min.z},
			{min.x, min.y, max.z},
			{max.x, min.y, max.z},
			{min.x, max.y, max.z},
			{max.x, max.y, max.z},
		};

		for(auto& c : corners){
			result.extend(vec3(matrix * vec4(c, 1.0f)));
		}

		return result;
	}
};

struct DeviceState{
	int counter;
	uint32_t numSmall;
	uint32_t numLarge;
	uint32_t numMassive;
	// uint32_t numNontrivial;
	uint64_t nanotime_start;
	uint64_t nanotime_stage_1;
	uint64_t nanotime_stage_2;
	uint64_t nanotime_stage_3;

	int32_t hovered_meshId;
	int32_t hovered_triangleIndex;

	uint32_t dbg_hovered_textureHandle;
	uint32_t dbg_hovered_mipLevel;
	uint32_t dbg_hovered_tx;
	uint32_t dbg_hovered_ty;
	uint32_t dbg_hovered_mcu_x;
	uint32_t dbg_hovered_mcu_y;
	uint32_t dbg_hovered_mcu;
	uint32_t dbg_hovered_decoded_color;
	uint64_t dbg_fragcount;
};

struct RenderTarget{
	uint64_t* framebuffer;
	uint64_t* colorbuffer;
	int width;
	int height;
	mat4 view;
	mat4 viewI;
	mat4 proj;
	vec3 cameraPos;
	float f;
	float aspect;
	bool debug;

	// Projection mode. In BOTH modes proj[0][0] and proj[1][1] are the view-space ->
	// NDC scale factors; the only difference is whether the result is divided by
	// depth. Perspective: ndc = proj[i][i] * v / depth. Orthographic: ndc = proj[i][i]
	// * v, with proj[i][i] = 1 / half-extent. Keeping that shared shape means every
	// projection site needs one branch rather than a separate code path.
	int   projMode;    // 0 = perspective, 1 = orthographic
	float orthoHalfH;  // orthographic half-height in world units (0 in perspective)
};

// True when the render target is orthographic. Free function so the branch reads the
// same on host and device.
#define CURAST_ORTHO(target) ((target).projMode == 1)

struct Uniforms{
	mat4 world;
	mat4 camWorld;
	mat4 transform;
	float time;
	float pad;
	uint32_t frameCount;

	struct {
		bool show;
		ivec2 start;
		ivec2 size;
	} inset;

};

struct CommonLaunchArgs{
	Uniforms uniforms;
	DeviceState* state;
};

struct HuffmanTable {
	int num_codes_per_bit_length[16];
	int huffman_values[256];
	// packed[i] = (codelength << 16) | huffman_key � one load covers both per ballot lane
	uint32_t packed[256];
};

struct QuantizationTable {
	int values[64];
};

struct Texture{
	int width;
	int height;
	uint32_t* data;
	HuffmanTable* huffmanTables;
	QuantizationTable* quanttables;
	uint32_t* mcuPositions;
	uint32_t handle;
	bool isTranslucent;
};

struct JpegPipeline{
	uint32_t* toDecode;
	uint32_t* toDecodeCounter;
	uint32_t* decoded;
	uint32_t* TBSlots;
	uint32_t* TBSlotsCounter;
	HashMap decodedMcuMap;
};

struct CMesh{
	uint32_t numTriangles;
	uint32_t* indices;
	vec3* positions;
	uint64_t cummulativeTriangleCount; // sum of all triangles in prior CMesh instances
	Box3 aabb;
	vec3 compressionFactor;
	uint32_t index_min;
	uint32_t bitsPerIndex;

	struct{
		int offset;
		int count;
	} instances;

	mat4 world;
	int id;
	vec2* uvs;
	vec3* normals;
	uint32_t* colors;
	uint32_t firstTriangle;
	uint32_t numVertices;
	uint32_t index_max;
	uint64_t address;


	Texture texture;

	bool isLoaded;
	bool flipTriangles;
	bool compressed;


	struct{
		vec3* positions;
		uint32_t* colors;
		uint32_t numPoints;
	} impostor;
};

// struct InstanceData{
// 	mat4 transform;
// 	bool flip;
// };

struct CPointcloud{
	mat4 world;
	vec3* positions;
	uint32_t* colors;
	uint32_t numPoints;
};

// struct Instances{
// 	uint32_t meshIndex;
// 	uint32_t numInstances;
// 	mat4* transforms;
// };

struct HugeTriangle{
	int meshIndex;
	int triangleIndex;
	int tile_x;
	int tile_y;
};

struct TranslucentTriangle{
	int meshIndex;
	int triangleIndex;
	int tile_x;
	int tile_y;
};

constexpr int TILE_SIZE = 64;
constexpr int TILE_SIZE_TRANSLUCENT = 16;
constexpr uint32_t TRIANGLES_PER_SWEEP = 256;
constexpr uint32_t MAX_HUGE_TRIANGLES = 5'000'000;
constexpr uint32_t MAX_TRANSLUCENT_TRIANGLES = 5'000'000;
constexpr uint32_t MAX_NONTRIVIAL_TRIANGLES = 5'000'000;
constexpr uint32_t THRESHOLD_SMALL = 128;
constexpr uint32_t THRESHOLD_LARGE = 4096;
constexpr uint32_t TRIANGLES_PER_CHUNK = 128;

//constexpr float NEAR = 0.01f;
constexpr uint64_t PACKMASK_MESHINDEX = 0b111'1111'1111'1111; // 15 bit
constexpr uint64_t PACKMASK_TRIANGLEINDEX = 0b1'1111'1111'1111'1111'1111'1111; // 25 bit
// constexpr float INFINITY_F32 = INFINITY; // 1.0f / 0.0f;
// constexpr float INFINITY_F32 = 0x7F800000u; // 1.0f / 0.0f;
// constexpr float NEAR = 0.01f;
// constexpr float INFINITY_F32 = __builtin_huge_valf();

enum class DisplayAttribute : int{
	NONE,
	TEXTURE,
	UV,
	NORMAL,
	VERTEX_COLORS,
	TRIANGLE_ID,
	MESH_ID,
	STAGE,
};

struct RasterizationSettings{
	bool showWireframe;
	bool enableDiffuseLighting;
	bool enableObjectPicking;
	DisplayAttribute displayAttribute;
};

struct IndexbufferCompressInfo{
	uint32_t* uncompressedIndices;
	void* compressedIndices;
	uint32_t numIndices;
	uint32_t minIndex;
	uint32_t maxIndex;
};

enum class IndexFetch{DIRECT, INDEXBUFFER};
enum class Compression{UNCOMPRESSED, IX_PU16};
enum class Instancing{NO, YES};



struct RasterArgs{
	CMesh* meshes;
	uint32_t numMeshes;
	CMesh* instances;
	uint32_t numInstances;
	mat4* transforms;
	uint32_t* numProcessedBatches;
	uint32_t* numProcessedBatches_nontrivial;
	HugeTriangle* hugeTriangles;
	uint32_t* hugeTrianglesCounter;
	uint32_t* numProcessedHugeTriangles;
	uint32_t* nontrivialTrianglesCounter;
	uint64_t* nontrivialTrianglesList;
	RenderTarget target;
	DeviceState* state;
};

extern __constant__ RenderTarget c_target;

// Environment lighting, uploaded to the resolve module's `c_env` constant.
//
// Diffuse irradiance is held as a 9-coefficient spherical-harmonic projection of an
// HDR environment map. SH9 is essentially exact for Lambertian diffuse (Ramamoorthi &
// Hanrahan 2001), so lighting needs no cubemap, no prefilter chain and no texture
// fetch -- just these constants and ~30 flops per shaded pixel.
//
// vec4 rows rather than vec3 on purpose: __constant__ reads want 16-byte alignment,
// and a vec3 array would leave the host and device disagreeing about padding.
struct EnvLighting {
	vec4 sh[9];      // xyz = A_l * L_lm (RGB), w unused
	vec4 keyDir;     // xyz = unit direction toward the brightest region of the map
	vec4 keyColor;   // xyz = that region's colour, normalised to peak 1
	float exposure;  // scales the SH irradiance
	int   enabled;   // 0 = use the built-in studio coefficients compiled into resolve.cu

	// Equirectangular radiance for drawing the map as a background. Kept separate from
	// `enabled` on purpose: lighting a scene with an environment and *showing* that
	// environment are independent choices, and for molecular figures you usually want
	// the former without the latter.
	vec4* pixels;         // device RGBA32F, row-major, null = nothing to draw
	int   texWidth;
	int   texHeight;
	int   showBackground; // 0 = keep the lighting, draw the solid background colour
	int   _pad;
};

extern __constant__ EnvLighting c_env;

// How the AO buffer becomes a shading multiplier: shade = floor + (1-floor)*ao^power.
// A __constant__ rather than kernel parameters because the curve is applied in three
// separate composite kernels, and threading two more floats through each of them adds
// signature churn for no benefit.
struct AoParams {
	float floorValue;  // darkest a fully occluded pixel may get
	float power;       // >1 deepens contact shadows without darkening open surfaces
	float _pad[2];
};

extern __constant__ AoParams c_ao;

// QuteMol-style depth-aware halo (Tarini et al. 2006).
//
// QuteMol draws an enlarged billboard per atom into a half-size halo texture and, per
// fragment, samples the scene depth behind it:
//
//     tmp.z = saturate((sceneDepth - fragDepth) * 1/P_depth_full)
//     tmp.z *= radialFalloff; tmp.z *= radialFalloff;   // squared, "for smoother edges"
//     color  = black * tmp.z + haloColour
//
// so the halo is strongest where a silhouette stands in front of something far behind
// it. Reproduced here as a screen-space pass over the depth buffer instead of extra
// billboard geometry -- at 158.9M atoms, per-atom billboards are not an option, and
// the depth buffer already holds everything the effect needs.
struct HaloParams {
	int   enabled;
	float size;       // search radius as a fraction of the smaller viewport dimension
	float strength;   // overall opacity                          (QuteMol P_halo_str)
	float color;      // 0 = black halo .. 1 = white halo         (QuteMol P_halo_col)
	float depthFull;  // depth gap giving a fully opaque halo, as a fraction of depth
	                  //                                          (QuteMol P_depth_full)
	int   dirs;       // sparse angular samples
	int   steps;      // radial samples per direction
	float _pad;
};

extern __constant__ HaloParams c_halo;

// Shading / diagnostic switches that several kernels need. A __constant__ rather than
// kernel parameters so adding a debug mode does not churn three kernel signatures.
struct ShadeParams {
	int   flatSpheres;  // 1 = albedo only, no directional lighting (illustrative style)
	int   debugView;    // see CURAST_DEBUG_* below
	float envRotation;  // radians, about world +Z, applied to environment lookups
	float envBgWiden;   // >1 widens the background's effective FOV so the map reads smaller
	// 1 = env lighting OFF gives a uniform ambient rather than the baked studio SH.
	// Without this the toggle is nearly invisible whenever the loaded map resembles
	// the map the baked coefficients came from.
	int   envNeutralWhenOff;
	float _pad2[3];
};

#define CURAST_DEBUG_OFF        0
#define CURAST_DEBUG_AO         1
#define CURAST_DEBUG_NORMAL     2
// Green = analytic ray-sphere hit, red = sub-pixel fallback (normal faked as -rayDir).
// The proportion of red is the thing to look at: where it dominates, shading and AO
// are both running on a constant normal and the image cannot help but look flat.
#define CURAST_DEBUG_IMPOSTOR   3

extern __constant__ ShadeParams c_shade;

// One sweep direction for the QuteMol-style AO bake. Passed through device memory as a
// single struct rather than as loose vec3 kernel parameters: a 12-byte glm::vec3 by
// value depends on host and device agreeing about struct layout inside the kernel
// parameter block, and a mismatch there corrupts every parameter after it silently.
struct AoSweep {
	vec3  origin;   float _p0;
	vec3  right;    float _p1;
	vec3  up;       float _p2;
	vec3  dir;      float _p3;
	float halfExtent;
	float depthBias;
	float tolerance;
	int   res;
};

// Sphere LOD configuration. Each level defines a camera-distance band, an atom-skip
// stride (atom is in level k iff sphereIdx % level[k].stride == 0), and a radius scale
// (typically stride^(1/scaleBias) so projected screen area stays ~constant). At render
// time each sphere picks the level whose [minDist..maxDist] band contains its distance
// to the camera AND whose stride filter accepts its index.
constexpr int MAX_SPHERE_LOD_LEVELS = 4;

struct SphereLodLevel {
	float minDist;     // sphere is invisible when camera distance < minDist (after fade-in band)
	float maxDist;     // sphere is invisible when camera distance > maxDist (after fade-out band)
	float overlap;     // width of the smoothstep fade band at each end of [minDist..maxDist]
	float scale;       // radius multiplier when this level is fully active
	int   stride;      // atom kept iff sphereIdx % stride == 0
	int   _pad0;
	int   _pad1;
	int   _pad2;
};

struct SphereLodConfig {
	int            numLevels;       // 0 = LOD disabled
	int            _pad[3];
	SphereLodLevel levels[MAX_SPHERE_LOD_LEVELS];
};

struct SphereRasterArgs {
	vec3*     positions;          // GPU device ptr, world space
	float*    radii;              // GPU device ptr
	uint32_t* colors;             // GPU device ptr, RGBA8 — null when atomTypes/palette is used
	uint8_t*  atomTypes;          // GPU device ptr, per-atom palette index (preferred path)
	uint32_t* colorPalette;       // GPU device ptr, 256-entry RGBA8 LUT for atomTypes
	// Baked per-atom AO (QuteMol style), 0..255. null = not baked, treat as fully open.
	uint8_t*  atomAO;
	uint32_t  numSpheres;
	uint64_t* sphere_framebuffer; // separate from triangle framebuffer
	SphereLodConfig lod;          // numLevels=0 → LOD off
	// LOD per-launch state. The kernel maps thread tid to atom (tid * dispatchStride),
	// so the host launches only numSpheres/dispatchStride threads per active level —
	// avoiding the pay-then-cull pattern that was bandwidth-bound at high atom counts.
	int activeLevel;              // -1 = no LOD active for this launch
	int dispatchStride;           // 1 = no stride mapping (one thread per sphere)
	// Partition rule: when multiple levels are active, this thread skips atoms that a
	// coarser active level will render — preventing the doughnut artifact (same atom
	// drawn twice at different scales). skipStride=0 disables the check.
	int   skipStride;
	float skipMinDist;            // coarser active level's distance band — only skip if
	float skipMaxDist;            // the coarser level ACTUALLY covers this atom
};

#ifdef __CUDA_ARCH__
// Per-sphere LOD selection. Returns the effective rasterization radius (0 = cull).
// Picks the level with the largest effective radius (scale * fade) among levels whose
// distance band contains `dist` AND whose stride accepts `sphereIdx`.
__device__ inline float sphereLodScaledRadius(
	const SphereLodConfig& cfg,
	uint32_t sphereIdx,
	float baseRadius,
	float dist
){
	if(cfg.numLevels <= 0) return baseRadius;
	float bestEff = -1.0f;
	float bestRadius = 0.0f;
	#pragma unroll
	for(int k = 0; k < MAX_SPHERE_LOD_LEVELS; k++){
		if(k >= cfg.numLevels) break;
		const SphereLodLevel& L = cfg.levels[k];
		if(L.stride > 1 && (sphereIdx % (uint32_t)L.stride) != 0u) continue;
		if(dist < L.minDist || dist > L.maxDist) continue;

		float fIn  = (L.overlap > 0.0f)
			? __saturatef((dist - L.minDist) / L.overlap)
			: 1.0f;
		float fOut = (L.overlap > 0.0f)
			? __saturatef((L.maxDist - dist) / L.overlap)
			: 1.0f;
		// smoothstep ish: 3t^2 - 2t^3
		fIn  = fIn  * fIn  * (3.0f - 2.0f * fIn);
		fOut = fOut * fOut * (3.0f - 2.0f * fOut);
		float fade = fminf(fIn, fOut);
		if(fade <= 0.0f) continue;

		float eff = L.scale * fade;
		if(eff > bestEff){
			bestEff = eff;
			bestRadius = baseRadius * eff;
		}
	}
	return (bestEff > 0.0f) ? bestRadius : 0.0f;
}
#endif