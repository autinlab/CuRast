#define CUB_DISABLE_BF16_SUPPORT

// === required by GLM ===
#define GLM_FORCE_CUDA
#define GLM_FORCE_NO_CTOR_INIT
#define CUDA_VERSION 12000
namespace std {
	using size_t = ::size_t;
};
// =======================

// #include <curand_kernel.h>
#include <cooperative_groups.h>
// #include <cooperative_groups/memcpy_async.h>

#include "./glm/glm/glm.hpp"
#include "./glm/glm/gtc/matrix_transform.hpp"
#include "./glm/glm/gtc/matrix_access.hpp"
#include "./glm/glm/gtx/transform.hpp"
#include "./glm/glm/gtc/quaternion.hpp"

#include "./utils.cuh"
#include "./HostDeviceInterface.h"
#include "../BitEdit.h"
#include "./rasterization_helpers.cuh"
// #include "../jpeg/JptInterface.cuh"

using glm::ivec2;
using glm::i8vec4;
using glm::vec4;

// uint32_t uvToMCUIndex(int width, int height, float u, float v) {
// 	int tx = (int(u * width) % width);
// 	int ty = (int(v * height) % height);
// 	return tx / 16 + ty / 16 * width / 16;
// }

__constant__ RenderTarget c_target;

// Environment lighting uploaded from the host (see src/EnvMap.h). When
// c_env.enabled == 0 the shading falls back to the studio coefficients baked in
// below, so the renderer looks the same as before if no map is loaded.
__constant__ EnvLighting c_env;

// AO response curve, uploaded from the host so it is tunable at runtime.
__constant__ AoParams c_ao;

// QuteMol-style halo parameters, uploaded from the host.
__constant__ HaloParams c_halo;

// Shading / diagnostic switches.
__constant__ ShadeParams c_shade;

__device__
vec4 getVertex(CMesh& mesh, uint32_t vertexIndex){
	vec4 position;
	if(!mesh.compressed){
		uint32_t resolvedIndex = mesh.indices[vertexIndex];
		position = vec4(mesh.positions[resolvedIndex], 1.0f);
	}else{
		uint32_t indexRange = mesh.index_max - mesh.index_min;
		uint64_t bitsPerIndex = ceil(log2f(float(indexRange + 1)));
		uint32_t resolvedIndex = BitEdit::readU32(mesh.indices, bitsPerIndex * vertexIndex, bitsPerIndex) + mesh.index_min;

		uint16_t* positions = (uint16_t*)mesh.positions;
		uint16_t X = positions[3 * resolvedIndex + 0];
		uint16_t Y = positions[3 * resolvedIndex + 1];
		uint16_t Z = positions[3 * resolvedIndex + 2];

		vec3 aabbSize = mesh.aabb.max - mesh.aabb.min;

		
		position.x = (float(X) / 65536.0f) * aabbSize.x + mesh.aabb.min.x;
		position.y = (float(Y) / 65536.0f) * aabbSize.y + mesh.aabb.min.y;
		position.z = (float(Z) / 65536.0f) * aabbSize.z + mesh.aabb.min.z;
		position.w = 1.0f;
	}

	return position;
};

__device__
vec4 getVertex_resolved(CMesh& mesh, uint32_t resolvedIndex){
	vec4 position;
	if(!mesh.compressed){
		position = vec4(mesh.positions[resolvedIndex], 1.0f);
	}else{
		uint16_t* positions = (uint16_t*)mesh.positions;
		uint16_t X = positions[3 * resolvedIndex + 0];
		uint16_t Y = positions[3 * resolvedIndex + 1];
		uint16_t Z = positions[3 * resolvedIndex + 2];

		vec3 aabbSize = mesh.aabb.max - mesh.aabb.min;

		
		position.x = (float(X) / 65536.0f) * aabbSize.x + mesh.aabb.min.x;
		position.y = (float(Y) / 65536.0f) * aabbSize.y + mesh.aabb.min.y;
		position.z = (float(Z) / 65536.0f) * aabbSize.z + mesh.aabb.min.z;
		position.w = 1.0f;
	}

	return position;
};

__device__
vec2 getUV(CMesh& mesh, uint32_t vertexIndex){

	bool isCompressed = mesh.positions == 0;

	if(mesh.uvs == nullptr){
		return vec2{0.0f, 0.0f};
	}else if(isCompressed){
		uint32_t indexRange = mesh.index_max - mesh.index_min;
		uint64_t bitsPerIndex = ceil(log2f(float(indexRange + 1)));
		uint32_t resolvedIndex = BitEdit::readU32(mesh.indices, bitsPerIndex * vertexIndex, bitsPerIndex) + mesh.index_min;
		return mesh.uvs[resolvedIndex];
	}else{
		uint32_t resolvedIndex = mesh.indices[vertexIndex];
		return mesh.uvs[resolvedIndex];
	}
};

__device__
vec2 getUV_resolved(CMesh& mesh, uint32_t resolvedIndex){

	bool isCompressed = mesh.positions == 0;

	if(mesh.uvs == nullptr){
		return vec2{0.0f, 0.0f};
	}else if(isCompressed){
		return mesh.uvs[resolvedIndex];
	}else{
		return mesh.uvs[resolvedIndex];
	}
};

__device__
uint32_t sampleColor_nearest(
	uint32_t* textureData,
	int width,
	int height,
	vec2 uv
){

	if(textureData == nullptr) return 0;
	// return 0xff660066;
	uv.x = uv.x - floor(uv.x);
	uv.y = uv.y - floor(uv.y);
	int tx = int(uv.x * float(width) + 0.5f) % width;
	int ty = int(uv.y * float(height) + 0.5f) % height;
	int texelID = tx + ty * width;
	texelID = clamp(texelID, 0, width * height - 1);

	uint32_t color = 0;
	color = textureData[texelID];
	//uint8_t *rgb = (uint8_t *)&color;

	return color;
}

__device__
uint32_t sampleColor_linear(
	uint32_t* textureData,
	int width,
	int height,
	vec2 uv
){

	if(textureData == nullptr) return 0;

	uint32_t color = 0xff000000;
	uint8_t* rgba = (uint8_t*)&color;

	// Only for ply with textures
	// if(uv.x > 1.0f) return 0;
	// if(uv.y > 1.0f) return 0;
	// uv.y = 1.0f - uv.y;

	float ftx = (uv.x - floor(uv.x)) * float(width);
	float fty = (uv.y - floor(uv.y)) * float(height);

	auto getTexel = [&](float ftx, float fty) -> vec4 {
		int tx = fmodf(ftx, float(width));
		int ty = fmodf(fty, float(height));
		int texelID = tx + ty * width;
		texelID = clamp(texelID, 0, width * height - 1);

		uint32_t texel = textureData[texelID];
		uint8_t* rgba = (uint8_t*)&texel;

		return vec4{rgba[0], rgba[1], rgba[2], rgba[3]};
	};

	vec4 t00 = getTexel(ftx - 0.5f, fty - 0.5f);
	vec4 t01 = getTexel(ftx - 0.5f, fty + 0.5f);
	vec4 t10 = getTexel(ftx + 0.5f, fty - 0.5f);
	vec4 t11 = getTexel(ftx + 0.5f, fty + 0.5f);

	float wx = fmodf(ftx + 0.5f, 1.0f);
	float wy = fmodf(fty + 0.5f, 1.0f);

	vec4 interpolated =
		(1.0f - wx) * (1.0f - wy) * t00 +
		wx * (1.0f - wy) * t10 +
		(1.0f - wx) * wy * t01 +
		wx * wy * t11;

	rgba[0] = interpolated.r;
	rgba[1] = interpolated.g;
	rgba[2] = interpolated.b;
	rgba[3] = interpolated.a;


	return color;
}

extern "C" __global__
void kernel_dummy(
	uint32_t* data
) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	if(grid.thread_rank() == 0) *data = 123;
}

extern "C" __global__
void kernel_clearFramebuffer(
	uint64_t* framebuffer,
	uint32_t numPixels,
	uint32_t clearColor,
	float clearDepth
) {
	auto grid = cg::this_grid();

	int pixelID = grid.thread_rank();
	if (pixelID >= numPixels) return;

	// uint64_t udepth = __float_as_uint(clearDepth);
	// uint64_t udepth = 0x00ffffff;
	// uint64_t pixel = udepth << 40;
	uint64_t pixel = 0xFFFFFFF0'00000000ULL;
	framebuffer[pixelID] = pixel;

	framebuffer[pixelID] = pixel;
}


__device__
float getEdlShadingFactor(uint64_t* colorbuffer, float depth, int x, int y, int distance){
	auto getNeighborDepth = [&](int x, int y) -> float{

		if(x < 0 || x >= c_target.width) return Infinity;
		if(y < 0 || y >= c_target.height) return Infinity;

		int pixelID = toFramebufferIndex(x, y, c_target.width);
		uint64_t pixel = colorbuffer[pixelID];

		float d = __uint_as_float(pixel >> 32);

		return d;
	};

	float sum = 0.0f;
	int numSamples = 8;
	for(int i = 0; i < numSamples; i++){
		float u = 2.0f * 3.1415f * float(i) / float(numSamples);
		float dx = float(distance) * cos(u);
		float dy = float(distance) * sin(u);
		
		sum += max(log2f(depth) - log2f(getNeighborDepth(x + dx, y + dy)), 0.0f);
	}

	// float response = sum / 4.0f;
	float response = sum / float(numSamples);
	float edlStrength = 0.9f;
	float shade = exp(-response * 300.0f * edlStrength);
	shade = clamp(shade, 0.3f, 1.0f);

	shade = shade * 0.8f + 0.2f;

	return shade;
}


__device__ __forceinline__ float fractf(float x) {
	return x - floorf(x);
}

// ── Normal G-buffer (RGBA8) ──────────────────────────────────────────────────

// Alpha byte doubles as a validity flag and a provenance tag: 0 = cleared, and
// non-zero = a stored normal. The two non-zero values distinguish an analytic
// ray-sphere normal from the sub-pixel fallback, which the impostor debug view reads.
// Alpha byte layout: [tag:2][bakedAO:6].
//
// The tag lives in the top two bits so a written pixel always has a non-zero alpha,
// which is what every "is there a normal here" test keys on. The remaining six bits
// carry the baked per-atom AO to the composite pass.
//
// Six bits is ample: baked AO is a low-frequency, view-independent term, and piggy-
// backing it here avoids a second full-resolution buffer for something the composite
// only needs one number from.
#define NORMAL_TAG_NONE     0x0u
#define NORMAL_TAG_ANALYTIC 0x1u
#define NORMAL_TAG_FALLBACK 0x2u

#define NORMAL_TAG_OF(v)     (((v) >> 30) & 0x3u)
#define NORMAL_BAKEDAO_OF(v) (((v) >> 24) & 0x3Fu)

// Pack a unit normal plus its baked AO. bakedAO is 0..1; 1 means "not baked", which is
// also the identity for the min() combine in the composite.
__device__ inline uint32_t packNormalGBuf(vec3 n, uint32_t tag = NORMAL_TAG_ANALYTIC,
                                          float bakedAO = 1.0f){
	float len = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
	if(len > 0.0f){ n.x /= len; n.y /= len; n.z /= len; }
	float fr = n.x * 0.5f + 0.5f; if(fr < 0.0f) fr = 0.0f; if(fr > 1.0f) fr = 1.0f;
	float fg = n.y * 0.5f + 0.5f; if(fg < 0.0f) fg = 0.0f; if(fg > 1.0f) fg = 1.0f;
	float fb = n.z * 0.5f + 0.5f; if(fb < 0.0f) fb = 0.0f; if(fb > 1.0f) fb = 1.0f;
	uint32_t r = (uint32_t)(fr * 255.0f + 0.5f);
	uint32_t g = (uint32_t)(fg * 255.0f + 0.5f);
	uint32_t b = (uint32_t)(fb * 255.0f + 0.5f);
	uint32_t ao6 = (uint32_t)(clamp(bakedAO, 0.0f, 1.0f) * 63.0f + 0.5f);
	return r | (g << 8) | (b << 16) | (ao6 << 24) | (tag << 30);
}

// Unpack a normal previously stored with packNormalGBuf. Returns vec3(0) if the
// pixel was cleared (alpha == 0).
__device__ inline vec3 unpackNormalGBuf(uint32_t v){
	if((v >> 24) == 0u) return vec3(0.0f, 0.0f, 0.0f); // sentinel
	float r = float(v        & 0xff) / 255.0f * 2.0f - 1.0f;
	float g = float((v >> 8) & 0xff) / 255.0f * 2.0f - 1.0f;
	float b = float((v >> 16)& 0xff) / 255.0f * 2.0f - 1.0f;
	return vec3(r, g, b);
}

// Convert a world-space surface normal to the SSAO's "+Z forward / depth = +Z"
// view space. Use a vec4 with w=0 to apply only the rotation part of the view
// matrix (avoids needing mat3(mat4) which is not always available under
// GLM_FORCE_CUDA). Standard view (camera looks toward -Z) needs a Z-flip to align.
//
// IMPORTANT: under GLM_FORCE_CUDA the 4-float vec4 constructor `vec4(x, y, z, w)`
// generates code that segfaults during sphere shading on this build. Using the
// vec3+float and vec4-to-vec3 conversion forms (which the working code in this
// file already uses) avoids it.
__device__ inline vec3 worldNormalToSsaoView(vec3 N_world){
	vec4 N_view4 = c_target.view * vec4(N_world, 0.0f);
	float nx = N_view4.x;
	float ny = N_view4.y;
	float nz = -N_view4.z; // flip Z to SSAO convention (+Z forward)
	if(nz > 0.0f){ nx = -nx; ny = -ny; nz = -nz; }
	float len = sqrtf(nx*nx + ny*ny + nz*nz);
	if(len > 0.0f){ float inv = 1.0f / len; nx *= inv; ny *= inv; nz *= inv; }
	vec3 out;
	out.x = nx; out.y = ny; out.z = nz;
	return out;
}

// ── Environment lighting (image-based) ───────────────────────────────────────
//
// Replaces the previous single hardcoded directional light
// (L = normalize(1,1,1); shade = d * 0.7 + 0.5), which gave every surface nearly
// the same tone regardless of orientation and was the main reason the render
// looked flat. No amount of AO can fix a constant ambient term, because AO only
// modulates ambient.
//
// Diffuse irradiance is stored as a 9-coefficient spherical-harmonic projection
// of an HDR environment map. SH9 is exact to within a fraction of a percent for
// Lambertian diffuse (Ramamoorthi & Hanrahan 2001), so this needs no cubemap, no
// prefilter pass and no texture fetch -- just 9 constants and ~30 flops. That is
// about what the old normalize()+dot() cost, and it gives every surface
// orientation its own colour.
//
// Coefficients below are A_l * L_lm projected from brown_photostudio_02_4k.hdr
// at 4096x2048, with world up = +Z to match OrbitControls. Regenerate for a
// different map with src/tools/sh9.py.
__device__ const float c_envSH[9][3] = {
	{  8.220565f,  8.002558f,  7.907027f},  // Y00
	{  3.049849f,  3.024605f,  3.004809f},  // Y1-1
	{  2.558317f,  2.577100f,  2.717383f},  // Y10
	{  3.082418f,  3.010968f,  2.954998f},  // Y11
	{  1.233800f,  1.196936f,  1.165705f},  // Y2-2
	{  1.296646f,  1.292287f,  1.292104f},  // Y2-1
	{  0.095700f,  0.133699f,  0.199860f},  // Y20
	{  1.322162f,  1.317976f,  1.304159f},  // Y21
	{  0.018696f,  0.006439f, -0.003603f},  // Y22
};

// Brightest 0.5% of the same map, used as a specular key so highlights have a
// direction. Diffuse SH alone cannot produce a highlight -- l<=2 is far too
// low-frequency to represent a light source.
__device__ const float c_envKeyDir[3]   = {0.664108f, 0.456129f, 0.592374f};
__device__ const float c_envKeyColor[3] = {1.000000f, 0.969328f, 0.942787f};

#define ENV_EXPOSURE     0.85f   // scales SH irradiance; this studio map integrates to slightly >1
#define ENV_SPEC_INTENS  0.28f   // specular key strength
#define ENV_SPEC_POWER   32.0f   // Blinn-Phong exponent
#define ENV_RIM_INTENS   0.18f   // Fresnel rim, the QuteMol-style silhouette cue

// Evaluate SH9 diffuse irradiance E(N). N must be a unit world-space normal
// (world up = +Z, matching the basis the coefficients were projected in).
__device__ inline void shIrradiance(vec3 N, float* out){
	// Environment lighting OFF means no environment: a uniform ambient at the map's
	// average brightness. Falling back to the baked studio coefficients instead made
	// the toggle look like a no-op, because those were projected from the same
	// photostudio map the user had loaded -- switching compared one projection of a
	// map against another projection of the same map.
	if(c_env.enabled == 0 && c_shade.envNeutralWhenOff != 0){
		out[0] = c_envSH[0][0] * 0.282095f;
		out[1] = c_envSH[0][1] * 0.282095f;
		out[2] = c_envSH[0][2] * 0.282095f;
		return;
	}

	// Rotating the environment must rotate the LIGHTING as well as the background,
	// or the two disagree about where the light is coming from. The background lookup
	// offsets its azimuth by +envRotation, which places the map at -envRotation in
	// world space, so the normal is rotated by +envRotation before the SH is evaluated.
	if(c_shade.envRotation != 0.0f){
		float c = cosf(c_shade.envRotation);
		float s = sinf(c_shade.envRotation);
		float nx = N.x * c - N.y * s;
		float ny = N.x * s + N.y * c;
		N.x = nx; N.y = ny;
	}

	float Y[9];
	Y[0] = 0.282095f;
	Y[1] = 0.488603f * N.y;
	Y[2] = 0.488603f * N.z;
	Y[3] = 0.488603f * N.x;
	Y[4] = 1.092548f * N.x * N.y;
	Y[5] = 1.092548f * N.y * N.z;
	Y[6] = 0.315392f * (3.0f * N.z * N.z - 1.0f);
	Y[7] = 1.092548f * N.x * N.z;
	Y[8] = 0.546274f * (N.x * N.x - N.y * N.y);

	out[0] = 0.0f; out[1] = 0.0f; out[2] = 0.0f;
	if(c_env.enabled != 0){
		// Coefficients projected from a map loaded at runtime (src/EnvMap.h).
		for(int i = 0; i < 9; i++){
			out[0] += c_env.sh[i].x * Y[i];
			out[1] += c_env.sh[i].y * Y[i];
			out[2] += c_env.sh[i].z * Y[i];
		}
	}else{
		for(int i = 0; i < 9; i++){
			out[0] += c_envSH[i][0] * Y[i];
			out[1] += c_envSH[i][1] * Y[i];
			out[2] += c_envSH[i][2] * Y[i];
		}
	}
	// A truncated SH series can ring slightly negative for high-contrast maps.
	out[0] = fmaxf(out[0], 0.0f);
	out[1] = fmaxf(out[1], 0.0f);
	out[2] = fmaxf(out[2], 0.0f);
}

// The old code multiplied 8-bit sRGB values by a lighting factor directly. That
// is wrong -- light adds linearly, sRGB does not -- and it is a large part of why
// the midtones looked washed out. Convert properly on the way in and out.
__device__ inline float srgbToLinear(float c){
	return (c <= 0.04045f) ? (c * (1.0f / 12.92f)) : powf((c + 0.055f) * (1.0f / 1.055f), 2.4f);
}
__device__ inline float linearToSrgb(float c){
	return (c <= 0.0031308f) ? (c * 12.92f) : (1.055f * powf(c, 1.0f / 2.4f) - 0.055f);
}

// Narkowicz ACES filmic approximation. Keeps the specular key from clipping to
// flat white and holds saturation in the bright end.
__device__ inline float tonemapACES(float x){
	x = fmaxf(x, 0.0f);
	float r = (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);
	return clamp(r, 0.0f, 1.0f);
}

// Full environment shade for one surface point.
//   base : packed albedo as stored by the colour themes
//   N    : unit world-space normal
//   V    : unit world-space direction from the surface toward the eye
// Byte order in / out matches the existing call sites (component 0,1,2 of the
// packed word, alpha forced opaque).
__device__ inline uint32_t shadeEnvironment(uint32_t base, vec3 N, vec3 V){
	uint8_t* bi = (uint8_t*)&base;

	bool flat = (c_shade.flatSpheres != 0);

	float E[3];
	if(flat){
		// Flat shading: uniform irradiance, no directional term. It still runs through
		// the same linearise -> light -> tonemap -> sRGB path as the lit mode, because
		// passing raw sRGB albedo straight out came out much darker: the albedo never
		// gets the exposure and filmic curve that lift the lit path's midtones, and
		// then AO and EDL multiply it down from there.
		E[0] = c_envSH[0][0] * 0.282095f;
		E[1] = c_envSH[0][1] * 0.282095f;
		E[2] = c_envSH[0][2] * 0.282095f;
		if(c_env.enabled != 0){
			E[0] = c_env.sh[0].x * 0.282095f;
			E[1] = c_env.sh[0].y * 0.282095f;
			E[2] = c_env.sh[0].z * 0.282095f;
		}
	}else{
		shIrradiance(N, E);
	}

	vec3 Lk = (c_env.enabled != 0)
		? vec3(c_env.keyDir.x, c_env.keyDir.y, c_env.keyDir.z)
		: vec3(c_envKeyDir[0], c_envKeyDir[1], c_envKeyDir[2]);
	vec3 H  = normalize(Lk + V);
	float ndoth = fmaxf(dot(N, H), 0.0f);
	float ndotl = fmaxf(dot(N, Lk), 0.0f);
	// Gate the highlight on N.L so the far side of a sphere cannot show a
	// specular lobe from a light it does not face.
	float spec = (!flat && ndotl > 0.0f) ? (ENV_SPEC_INTENS * powf(ndoth, ENV_SPEC_POWER)) : 0.0f;

	// Schlick-style Fresnel rim: brightens grazing angles, which outlines each
	// atom against its neighbours. This is the cue QuteMol gets from its
	// depth-aware silhouettes, obtained here for a few flops.
	float ndotv = fmaxf(dot(N, V), 0.0f);
	float omn   = 1.0f - ndotv;
	float rim   = flat ? 0.0f : (ENV_RIM_INTENS * omn * omn * omn * omn);

	float exposure = (c_env.enabled != 0) ? c_env.exposure : ENV_EXPOSURE;
	if(flat) exposure *= c_shade.flatBrightness;
	float keyRGB[3];
	if(c_env.enabled != 0){
		keyRGB[0] = c_env.keyColor.x; keyRGB[1] = c_env.keyColor.y; keyRGB[2] = c_env.keyColor.z;
	}else{
		keyRGB[0] = c_envKeyColor[0]; keyRGB[1] = c_envKeyColor[1]; keyRGB[2] = c_envKeyColor[2];
	}

	uint32_t outColor = 0xff000000;
	uint8_t* bo = (uint8_t*)&outColor;
	for(int c = 0; c < 3; c++){
		float albedo = srgbToLinear(float(bi[c]) * (1.0f / 255.0f));
		float lit    = albedo * E[c] * (exposure / 3.14159265f)
		             + keyRGB[c] * (spec + rim);
		bo[c] = (uint8_t)(linearToSrgb(tonemapACES(lit)) * 255.0f + 0.5f);
	}
	return outColor;
}

// Sample the environment map along a world-space direction.
//
// Equirectangular convention matches the SH projection in src/EnvMap.h: world up is
// +Z, the top row of the image is +Z, and the horizontal axis sweeps phi about Z. If
// these two ever disagree the background and the lighting come from different
// orientations, which is subtle and maddening to debug.
__device__ inline vec3 sampleEnvDirection(vec3 dir){
	float len = sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
	if(len > 0.0f){ dir.x /= len; dir.y /= len; dir.z /= len; }

	float theta = acosf(clamp(dir.z, -1.0f, 1.0f));
	float phi   = atan2f(dir.y, dir.x) + c_shade.envRotation;

	// Wrap phi after the rotation, otherwise the seam lands wherever the offset put it.
	phi = phi - 6.28318531f * floorf((phi + 3.14159265f) * (1.0f / 6.28318531f));

	float u = (phi + 3.14159265f) * (1.0f / 6.28318531f);
	float v = theta * (1.0f / 3.14159265f);

	int W = c_env.texWidth;
	int H = c_env.texHeight;
	if(W <= 0 || H <= 0 || c_env.pixels == nullptr) return vec3(0.0f, 0.0f, 0.0f);

	// Bilinear, wrapping in u and clamping in v.
	float fx = u * float(W) - 0.5f;
	float fy = v * float(H) - 0.5f;
	int   x0 = (int)floorf(fx);
	int   y0 = (int)floorf(fy);
	float tx = fx - float(x0);
	float ty = fy - float(y0);

	int x1 = x0 + 1;
	int y1 = y0 + 1;
	x0 = ((x0 % W) + W) % W;
	x1 = ((x1 % W) + W) % W;
	y0 = clamp(y0, 0, H - 1);
	y1 = clamp(y1, 0, H - 1);

	vec4 c00 = c_env.pixels[y0 * W + x0];
	vec4 c10 = c_env.pixels[y0 * W + x1];
	vec4 c01 = c_env.pixels[y1 * W + x0];
	vec4 c11 = c_env.pixels[y1 * W + x1];

	vec3 a = vec3(c00.x, c00.y, c00.z) * (1.0f - tx) + vec3(c10.x, c10.y, c10.z) * tx;
	vec3 b = vec3(c01.x, c01.y, c01.z) * (1.0f - tx) + vec3(c11.x, c11.y, c11.z) * tx;
	return a * (1.0f - ty) + b * ty;
}

// Background colour for a pixel the geometry did not cover. Returns false when the
// environment background is unavailable or switched off, leaving the caller on the
// solid background colour.
// Defined below with the rest of the ray generation; declared here so the background
// uses the identical generator rather than a second copy that could drift from it.
__device__ inline vec3 computeRayDirection(float px, float py, float width, float height);

__device__ inline bool envBackgroundColor(int x, int y, uint32_t* out){
	if(c_env.enabled == 0 || c_env.showBackground == 0 || c_env.pixels == nullptr) return false;

	// Same generator the primary rays use, so the background stays consistent with the
	// geometry under both projection modes. (Orthographic gives every pixel the same
	// direction, so the background becomes a single flat colour -- which is correct:
	// a parallel projection genuinely sees one direction of the environment.)
	vec3 dirWorld = computeRayDirection(float(x) + 0.5f, float(y) + 0.5f,
	                                    float(c_target.width), float(c_target.height));

	// Widen the background sample angle. An environment map is infinitely far away, so
	// at a 60 degree fov you only ever see about a sixth of the panorama and whatever
	// is behind the model looks enormous next to it. Spreading the direction away from
	// the view axis shows more of the map, which shrinks its features to a plausible
	// size. Not physical -- it is a framing control, and it does not touch the lighting.
	if(c_shade.envBgWiden > 1.0001f || c_shade.envBgWiden < 0.9999f){
		vec3 axis = computeRayDirection(float(c_target.width) * 0.5f, float(c_target.height) * 0.5f,
		                                float(c_target.width), float(c_target.height));
		float cosA = clamp(dot(dirWorld, axis), -1.0f, 1.0f);
		float ang  = acosf(cosA) * c_shade.envBgWiden;
		ang = fminf(ang, 3.14159265f);
		vec3 perp = dirWorld - axis * cosA;
		float pl = length(perp);
		if(pl > 1e-6f){
			perp = perp / pl;
			dirWorld = normalize(axis * cosf(ang) + perp * sinf(ang));
		}
	}

	vec3 radiance = sampleEnvDirection(dirWorld) * c_env.exposure;

	uint32_t color = 0xff000000;
	uint8_t* rgba = (uint8_t*)&color;
	rgba[0] = (uint8_t)(linearToSrgb(tonemapACES(radiance.x)) * 255.0f + 0.5f);
	rgba[1] = (uint8_t)(linearToSrgb(tonemapACES(radiance.y)) * 255.0f + 0.5f);
	rgba[2] = (uint8_t)(linearToSrgb(tonemapACES(radiance.z)) * 255.0f + 0.5f);

	*out = color;
	return true;
}

// ── Primary ray generation ───────────────────────────────────────────────────
//
// Perspective: every ray starts at the eye, the direction varies per pixel.
// Orthographic: the direction is constant and the ORIGIN varies per pixel.
//
// Both halves matter. The resolve pass traces four neighbouring rays to get UV
// derivatives for mip selection; under orthographic those rays are parallel, so
// keeping one shared origin would make all four identical and collapse the
// derivatives to zero.
__device__ inline vec3 computeRayOrigin(float px, float py, float width, float height){
	vec3 eye = vec3(c_target.viewI * vec4(0.0f, 0.0f, 0.0f, 1.0f));
	if(!CURAST_ORTHO(c_target)) return eye;

	float u = 2.0f * px / width  - 1.0f;
	float v = 2.0f * py / height - 1.0f;
	vec3 posView = vec3(u / c_target.proj[0][0], v / c_target.proj[1][1], 0.0f);
	return vec3(c_target.viewI * vec4(posView, 1.0f));
}

__device__ inline vec3 computeRayDirection(float px, float py, float width, float height){
	vec3 eye = vec3(c_target.viewI * vec4(0.0f, 0.0f, 0.0f, 1.0f));

	if(CURAST_ORTHO(c_target)){
		// Camera forward in world space. w=1 on both points and subtract, matching the
		// vec4 usage elsewhere in this file (the 4-float vec4 ctor is avoided
		// deliberately -- see the note on worldNormalToSsaoView).
		vec3 ahead = vec3(c_target.viewI * vec4(vec3(0.0f, 0.0f, -1.0f), 1.0f));
		return normalize(ahead - eye);
	}

	float u = 2.0f * px / width  - 1.0f;
	float v = 2.0f * py / height - 1.0f;
	vec3 dirView = normalize(vec3(u / c_target.proj[0][0], v / c_target.proj[1][1], -1.0f));
	return normalize(vec3(c_target.viewI * vec4(dirView, 1.0f)) - eye);
}

// Diagnostic visualisations, shared by the screenshot and on-screen composites.
__device__ inline uint32_t debugViewColor(
	int x, int y, int pixelID, float depth,
	float* ssaoShadeBuffer, uint32_t* normalbuffer
){
	uint32_t out = 0xff000000;
	uint8_t* p = (uint8_t*)&out;

	if(isinf(depth)) return out;   // background stays black in every debug mode

	if(c_shade.debugView == CURAST_DEBUG_AO){
		uint8_t v = (uint8_t)(clamp(ssaoShadeBuffer[pixelID], 0.0f, 1.0f) * 255.0f + 0.5f);
		p[0] = v; p[1] = v; p[2] = v;
		return out;
	}

	if(normalbuffer == nullptr) return out;
	uint32_t packed = normalbuffer[pixelID];
	uint32_t tag    = NORMAL_TAG_OF(packed);

	if(c_shade.debugView == CURAST_DEBUG_NORMAL){
		if(tag == NORMAL_TAG_NONE) return out;
		// Show the stored normal directly, remapped from [-1,1] to [0,255]. A correct
		// impostor normal buffer looks like a field of little shaded spheres; a flat
		// wash of one colour means the fallback is dominating.
		vec3 n = unpackNormalGBuf(packed);
		p[0] = (uint8_t)clamp((n.x * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f);
		p[1] = (uint8_t)clamp((n.y * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f);
		p[2] = (uint8_t)clamp((n.z * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f);
		return out;
	}

	if(c_shade.debugView == CURAST_DEBUG_IMPOSTOR){
		if(tag == NORMAL_TAG_ANALYTIC){ p[1] = 200; }        // green: real ray-sphere hit
		else if(tag == NORMAL_TAG_FALLBACK){ p[0] = 200; }   // red: sub-pixel fallback
		return out;
	}

	return out;
}

// Combine the screen-space AO with the baked per-atom AO carried in the normal
// buffer's alpha. min(), not a product: the two measure overlapping occlusion of the
// same geometry, so multiplying them double counts and crushes anywhere both agree --
// which is everywhere once the camera is inside a packed structure.
__device__ inline float combineAO(float screenAO, uint32_t* normalbuffer, int pixelID){
	if(normalbuffer == nullptr) return screenAO;
	uint32_t packed = normalbuffer[pixelID];
	if(NORMAL_TAG_OF(packed) == NORMAL_TAG_NONE) return screenAO;
	float baked = float(NORMAL_BAKEDAO_OF(packed)) * (1.0f / 63.0f);
	return fminf(screenAO, baked);
}

// Strength of the QuteMol-style halo at one pixel, in [0,1].
//
// Screen-space reformulation of QuteMol's billboard pass: instead of rasterising an
// enlarged quad per atom, look outward from this pixel for a surface that is NEARER
// than it. If one exists, this pixel is "behind a silhouette" and takes a halo whose
// opacity follows the depth gap, exactly as the original scales by 1/P_depth_full.
//
// The radial falloff is squared, matching the original's two multiplies -- QuteMol's
// comment on the second one is "again for smoother edges", and it does visibly matter:
// a linear falloff leaves a hard ring at the search radius.
//
// Sampling is sparse (dirs x steps) because the result is a max over the neighbourhood
// and the falloff is smooth, so a dense scan buys nothing. QuteMol reached the same
// conclusion from the other direction, rendering its halo into a reduced-size texture.
__device__ inline float haloStrengthAt(int x, int y, float centerDepth){
	if(c_halo.enabled == 0 || c_halo.strength <= 0.0f) return 0.0f;

	int   W = c_target.width;
	int   H = c_target.height;
	float radiusPx = c_halo.size * float(min(W, H));
	if(radiusPx < 1.0f) return 0.0f;

	int nDirs  = max(c_halo.dirs, 1);
	int nSteps = max(c_halo.steps, 1);

	bool centerIsBackground = isinf(centerDepth);

	// Rotate the sample pattern per pixel, otherwise the sparse directions show up as
	// a star-shaped artefact around isolated silhouettes.
	uint32_t h = uint32_t(x) * 2246822519u ^ uint32_t(y) * 3266489917u;
	h ^= h >> 13; h *= 0xbf58476du; h ^= h >> 31;
	float rot = float(h >> 8) * (6.28318531f / float(1 << 24));

	float best = 0.0f;

	for(int d = 0; d < nDirs; d++){
		float ang = rot + float(d) * (6.28318531f / float(nDirs));
		float dx = cosf(ang);
		float dy = sinf(ang);

		for(int s = 1; s <= nSteps; s++){
			float t  = float(s) / float(nSteps);
			int   sx = int(float(x) + dx * radiusPx * t);
			int   sy = int(float(y) + dy * radiusPx * t);
			if(sx < 0 || sy < 0 || sx >= W || sy >= H) continue;

			float dN = __uint_as_float(c_target.colorbuffer[toFramebufferIndex(sx, sy, W)] >> 32);
			if(isinf(dN)) continue;   // neighbour is background: nothing to cast a halo

			// Gap to a NEARER neighbour. Background has no finite depth, so it takes the
			// full gap from any geometry it sees -- which is the common case and the one
			// that produces the silhouette glow.
			float g;
			if(centerIsBackground){
				g = 1.0f;
			}else{
				float gap = centerDepth - dN;
				if(gap <= 0.0f) continue;
				g = clamp(gap / fmaxf(c_halo.depthFull * centerDepth, 1e-6f), 0.0f, 1.0f);
			}

			float fall = 1.0f - t * t;
			float v    = g * fall * fall;
			if(v > best) best = v;
		}
	}

	return clamp(best * c_halo.strength, 0.0f, 1.0f);
}

// Blend a colour toward the halo colour. Byte order matches the packed pixels.
__device__ inline uint32_t applyHalo(uint32_t color, float strength){
	if(strength <= 0.0f) return color;

	uint8_t* rgba = (uint8_t*)&color;
	float target = clamp(c_halo.color, 0.0f, 1.0f) * 255.0f;
	for(int c = 0; c < 3; c++){
		rgba[c] = (uint8_t)clamp(float(rgba[c]) * (1.0f - strength) + target * strength, 0.0f, 255.0f);
	}
	return color;
}

// ── SSAO Helpers ─────────────────────────────────────────────────────────────

// Reconstruct view-space position from screen pixel + linear depth.
// Depth is the positive Z distance from the camera; proj contains focal lengths.
__device__ vec3 ssao_viewPos(float px, float py, float depth, int width, int height) {
	float ndc_x = (2.0f * (px + 0.5f) / float(width))  - 1.0f;
	float ndc_y = (2.0f * (py + 0.5f) / float(height)) - 1.0f;

	// Orthographic: the ray footprint does not widen with distance, so x and y come
	// straight from NDC without the depth factor.
	float scale = CURAST_ORTHO(c_target) ? 1.0f : depth;

	return vec3(
		ndc_x * scale / c_target.proj[0][0],
		ndc_y * scale / c_target.proj[1][1],
		depth
	);
}

// Project a view-space position back to screen pixel coordinates.
__device__ vec2 ssao_screenPos(vec3 P, int width, int height) {
	float inv_z = CURAST_ORTHO(c_target) ? 1.0f : (1.0f / P.z);
	return vec2(
		((c_target.proj[0][0] * P.x * inv_z) + 1.0f) * 0.5f * float(width)  - 0.5f,
		((c_target.proj[1][1] * P.y * inv_z) + 1.0f) * 0.5f * float(height) - 0.5f
	);
}

// Reconstruct view-space normal from the depth buffer via edge-aware cross products.
// The returned normal points toward the camera (N.z <= 0 with depth = +Z into scene).
__device__ vec3 ssao_normal(int x, int y, float d0, uint64_t* cb, int w, int h) {
	auto getDepth = [&](int nx, int ny) -> float {
		nx = clamp(nx, 0, w - 1);
		ny = clamp(ny, 0, h - 1);
		return __uint_as_float(cb[ny * w + nx] >> 32);
	};

	float dr = getDepth(x + 1, y),  dl = getDepth(x - 1, y);
	float dd = getDepth(x, y + 1),  du = getDepth(x, y - 1);

	// Edge-aware: pick the neighbor with the smaller depth discontinuity
	bool use_right = fabsf(dr - d0) < fabsf(d0 - dl);
	bool use_down  = fabsf(dd - d0) < fabsf(d0 - du);
	int  sx = use_right ? +1 : -1;
	int  sy = use_down  ? +1 : -1;
	float dh = use_right ? dr : dl;
	float dv = use_down  ? dd : du;

	// Fallback if a neighbor is missing (silhouette against sky)
	if (isinf(dh) || isinf(dv)) return vec3(0.0f, 0.0f, -1.0f);

	vec3 P  = ssao_viewPos(float(x),       float(y),       d0, w, h);
	vec3 Ph = ssao_viewPos(float(x + sx),  float(y),       dh, w, h);
	vec3 Pv = ssao_viewPos(float(x),       float(y + sy),  dv, w, h);

	// Consistent forward-differences regardless of which neighbor was chosen
	vec3 dPh = (sx > 0) ? (Ph - P) : (P - Ph);
	vec3 dPv = (sy > 0) ? (Pv - P) : (P - Pv);

	// Cross product; negate so the normal points toward the camera
	vec3 N = normalize(cross(dPh, dPv));
	return (N.z > 0.0f) ? -N : N;
}

// ─────────────────────────────────────────────────────────────────────────────

// Scale-agnostic hemisphere SSAO with view-space normal reconstruction.
//
// The AO radius is world_radius = depth * RADIUS_FRACTION, which makes the
// projected screen-space footprint constant regardless of scene scale:
//   pixel_radius = RADIUS_FRACTION * proj[1][1] * height   (depth-independent)
// Doubling the scene (objects + distances) leaves the shading identical.
//
// Occlusion test uses a tangent-plane reference depth to prevent self-occlusion
// on steep triangles.  For a steep surface, depth changes rapidly across pixels,
// so S.z can easily be "behind" the surface at the reprojected pixel even though
// S is above the surface in 3D.  The tangent-plane reference (expected_z) gives
// the depth the *current* surface should have at each sample location; only
// geometry that protrudes above that plane counts as a real occluder.
// Multiscale variant: runs the same scale-agnostic hemisphere SSAO at multiple
// `radius_fraction` levels and combines via max(occlusion) so the strongest cue
// at any scale wins. This catches both fine cavities (small radius) and overall
// enclosure / skylight blocking (large radius), which is what makes large
// molecular assemblies read as solid rather than a mist of dots.
//
// `radius_fractions[k]` is the level's world radius expressed as a fraction of
// the centre pixel's depth (so screen-space footprint is depth-independent).
// `bias_factors[k]` is a per-level intensity multiplier.
// `samples_per_level` is sample count per level; total work scales linearly.
__device__ float getSSAOShadingFactor(
	uint64_t* colorbuffer,
	uint32_t* normalbuffer,
	float     center_depth,
	int x, int y,
	int width, int height,
	float /* focal_length — kept for API compatibility; use c_target.proj directly */,
	const float* radius_fractions,
	const float* bias_factors,
	int   num_levels,
	const int* samples_per_level,
	float intensity
) {
	if (isinf(center_depth) || center_depth <= 0.0f) return 1.0f;

	const float RANGE_MUL     = 2.5f;
	const float BIAS_FRACTION = 0.05f;

	// Reconstruct view-space geometry at the center pixel.
	vec3 P = ssao_viewPos(float(x), float(y), center_depth, width, height);

	// Prefer the analytic normal stashed by the resolve pass — for sphere imposters
	// the depth-gradient reconstruction below is unreliable in dense clusters where
	// every pixel sits near a silhouette edge. Fall back to depth gradients only
	// when the normal G-buffer has no entry (e.g. triangle-only pixels).
	vec3 N;
	if(normalbuffer != nullptr){
		uint32_t packed = normalbuffer[y * width + x];
		if((packed >> 24) != 0u){
			N = unpackNormalGBuf(packed);
		} else {
			N = ssao_normal(x, y, center_depth, colorbuffer, width, height);
		}
	} else {
		N = ssao_normal(x, y, center_depth, colorbuffer, width, height);
	}

	// Screen-space depth gradients for the tangent-plane reference.
	auto getDepth = [&](int nx, int ny) -> float {
		nx = clamp(nx, 0, width  - 1);
		ny = clamp(ny, 0, height - 1);
		return __uint_as_float(colorbuffer[ny * width + nx] >> 32);
	};
	float dz_left  = center_depth - getDepth(x - 1, y);
	float dz_right = getDepth(x + 1, y) - center_depth;
	float dz_up    = center_depth - getDepth(x, y - 1);
	float dz_down  = getDepth(x, y + 1) - center_depth;
	float dz_dx = (fabsf(dz_left) < fabsf(dz_right)) ? dz_left : dz_right;
	float dz_dy = (fabsf(dz_up)   < fabsf(dz_down))  ? dz_up   : dz_down;

	// Build orthonormal tangent frame (T, B, N) around the surface normal
	vec3 up = (fabsf(N.z) < 0.95f) ? vec3(0.0f, 0.0f, 1.0f) : vec3(1.0f, 0.0f, 0.0f);
	vec3 T  = normalize(cross(up, N));
	vec3 B  = cross(N, T);

	// Per-pixel decorrelation — integer hash avoids the banding of smooth spatial functions
	uint32_t h = uint32_t(x) * 2246822519u ^ uint32_t(y) * 3266489917u;
	h ^= h >> 13; h *= 0xbf58476du; h ^= h >> 31;
	float rand_angle_base = float(h >> 8) * (6.28318530f / float(1 << 24));

	const float GOLDEN_ANGLE = 2.39996323f;

	float best_occlusion = 0.0f;

	if(num_levels < 1) num_levels = 1;
	if(num_levels > 4) num_levels = 4;

	for(int level = 0; level < num_levels; level++){
		// Sample count is per level: far-scale occlusion is low-frequency, so it needs
		// far fewer samples than the close level and the bilateral blur absorbs the rest.
		int   n_samples = samples_per_level[level];
		if(n_samples < 1) n_samples = 1;
		const float INV_N = 1.0f / float(n_samples);

		float radius_fraction = radius_fractions[level];
		float bias_factor     = bias_factors[level];
		// Same reference-length rule as GTAO: depth under perspective, view half-height
		// under orthographic, where the screen-to-world scale is depth-independent.
		float world_radius    = (CURAST_ORTHO(c_target) ? c_target.orthoHalfH : center_depth)
		                      * radius_fraction;
		float bias            = BIAS_FRACTION * world_radius;
		// Decorrelate the sample direction across levels so they don't all probe
		// the exact same ray pattern (rotating phi by pi / numLevels per level).
		float rand_angle      = rand_angle_base + float(level) * (3.14159265f / float(num_levels));

		float occlusion = 0.0f;

		for (int i = 0; i < n_samples; ++i) {
			float fi        = (float(i) + 0.5f) * INV_N;
			float sin_theta = sqrtf(fi);
			float cos_theta = sqrtf(1.0f - fi);
			float phi       = float(i) * GOLDEN_ANGLE + rand_angle;

			vec3 dir = T * (sin_theta * cosf(phi))
			         + B * (sin_theta * sinf(phi))
			         + N *  cos_theta;

			float r = world_radius * sqrtf(float(i + 1) * INV_N);
			vec3 S  = P + dir * r;

			if (S.z <= 0.001f) continue;

			vec2 sp = ssao_screenPos(S, width, height);
			int  sx = clamp(int(sp.x), 0, width  - 1);
			int  sy = clamp(int(sp.y), 0, height - 1);

			float actual_depth = __uint_as_float(colorbuffer[sy * width + sx] >> 32);
			if (isinf(actual_depth)) continue;

			float dx_p = sp.x - float(x);
			float dy_p = sp.y - float(y);
			float expected_z = center_depth + dz_dx * dx_p + dz_dy * dy_p;
			float dz = expected_z - actual_depth;

			if (dz > bias && dz < world_radius * RANGE_MUL) {
				float range_falloff = 1.0f - dz / (world_radius * RANGE_MUL);
				occlusion += range_falloff * cos_theta;
			}
		}

		occlusion *= INV_N * bias_factor;
		if(occlusion > best_occlusion) best_occlusion = occlusion;
	}

	return clamp(1.0f - best_occlusion * intensity, 0.0f, 1.0f);
}

// How the AO buffer is turned into a shading multiplier.
//
// This used to be `ao * 0.4 + 0.6`, which remaps AO into [0.6, 1.0]. Measured on
// the mycoplasma cell the AO buffer itself spans roughly 0.55-0.95, so after that
// remap the darkest crevice and the most open surface differed by about 5% -- the
// occlusion was computed correctly and then almost entirely discarded on the way
// to the image. That, more than the sampling quality, is why AO "did nothing".
//
// Full range is used instead, with a floor so deep cavities stay readable rather
// than crushing to black, and a power > 1 to deepen contact shadows without
// darkening open surfaces.
__device__ inline float applyAO(float ao){
	ao = clamp(ao, 0.0f, 1.0f);
	float f = clamp(c_ao.floorValue, 0.0f, 1.0f);
	float p = fmaxf(c_ao.power, 0.01f);
	return f + (1.0f - f) * powf(ao, p);
}

// ── GTAO (Ground Truth Ambient Occlusion) ────────────────────────────────────
//
// Replaces the hemisphere point-sampling above. That sampler draws N directions
// in the hemisphere and counts hits -- a Monte Carlo estimate whose variance
// lands directly in the image as speckle, and which needs a lot of samples to
// settle. Averaging it down with a wide bilateral blur trades the speckle for
// mush, which is what "not looking good" was.
//
// GTAO (Jimenez et al. 2016) slices the hemisphere with planes containing the
// view vector. In each slice it searches screen space for the largest horizon
// angle either side of the pixel, then evaluates the cosine-weighted visibility
// integral over that entire slice ANALYTICALLY. Each slice returns a smooth
// exact answer for its plane instead of a noisy estimate of a few directions,
// so a few slices beat many hemisphere samples at equal cost.
//
// Slices are AVERAGED. The previous multiscale code combined levels with
// max(), which promotes whichever level is noisiest into the result; averaging
// is the correct combine for a visibility integral.
//
// Note the horizon defaults: an unoccluded direction has its horizon at pi/2
// from V, not pi, because a screen-space search can only see the hemisphere
// facing the camera. Starting the search at cos(h) = 0 rather than -1 is what
// makes an open surface integrate to visibility 1 instead of overshooting.
__device__ inline float gtaoArc(float h, float n, float cos_n, float sin_n){
	// Jimenez et al. 2016, eq. 7 -- one side of one slice.
	return -cosf(2.0f * h - n) + cos_n + 2.0f * h * sin_n;
}

// `radius_fraction` is the world-space search radius as a fraction of the pixel's
// depth, so the screen-space footprint is constant with distance (the same
// scale-agnostic trick the old sampler used).
// `thickness` widens the distance falloff; occluders beyond it stop counting,
// which prevents distant silhouettes from over-darkening foreground surfaces.
__device__ float getGTAOShadingFactor(
	uint64_t* colorbuffer,
	uint32_t* normalbuffer,
	float     center_depth,
	int x, int y,
	int width, int height,
	float radius_fraction,
	int   num_slices,
	int   num_steps,
	float intensity,
	float thickness
){
	if(isinf(center_depth) || center_depth <= 0.0f) return 1.0f;

	if(num_slices < 1) num_slices = 1;
	if(num_steps  < 1) num_steps  = 1;

	vec3 P = ssao_viewPos(float(x), float(y), center_depth, width, height);

	// Depth is +Z away from the eye, so the direction back to the eye is -P.
	vec3 V = normalize(P * -1.0f);

	vec3 N;
	if(normalbuffer != nullptr){
		uint32_t packed = normalbuffer[y * width + x];
		if((packed >> 24) != 0u){
			N = normalize(unpackNormalGBuf(packed));
		} else {
			N = ssao_normal(x, y, center_depth, colorbuffer, width, height);
		}
	} else {
		N = ssao_normal(x, y, center_depth, colorbuffer, width, height);
	}

	// Under perspective the world radius scales with depth, which is what keeps the
	// screen footprint constant. Under orthographic the screen-to-world scale does not
	// depend on depth at all, so the reference length is the view half-height instead;
	// using depth there collapses the search to a fraction of a pixel and the AO
	// silently vanishes.
	bool  isOrtho      = CURAST_ORTHO(c_target);
	float refLength    = isOrtho ? c_target.orthoHalfH : center_depth;
	float world_radius = refLength * radius_fraction;
	if(world_radius <= 0.0f) return 1.0f;

	// Screen-space search radius. A world offset r projects to
	// proj[i][i] * (r / refLength) * 0.5 * extent pixels, and r / refLength is exactly
	// radius_fraction, so this is depth-independent in both modes.
	float rx = c_target.proj[0][0] * world_radius * 0.5f * float(width)  / (isOrtho ? 1.0f : center_depth);
	float ry = c_target.proj[1][1] * world_radius * 0.5f * float(height) / (isOrtho ? 1.0f : center_depth);

	// Per-pixel slice rotation + step jitter. Interleaving the pattern spatially
	// lets the bilateral blur recover the between-slice directions cheaply.
	uint32_t h = uint32_t(x) * 2246822519u ^ uint32_t(y) * 3266489917u;
	h ^= h >> 13; h *= 0xbf58476du; h ^= h >> 31;
	float rot    = float(h >> 8) * (1.0f / float(1 << 24));
	float jitter = float((h >> 3) & 0xffffu) * (1.0f / 65536.0f);

	const float PI     = 3.14159265f;
	const float HALFPI = 1.57079633f;

	float visibility = 0.0f;

	for(int s = 0; s < num_slices; s++){
		float phi = (float(s) + rot) * (PI / float(num_slices));
		float dx  = cosf(phi);
		float dy  = sinf(phi);

		// View-space direction corresponding to moving along (dx, dy) in pixels at
		// constant depth. Needed because the pixel grid is anisotropic in view space,
		// so the slice tangent is not simply (dx, dy, 0).
		vec3 dirView = vec3(dx / (float(width) * c_target.proj[0][0]),
		                    dy / (float(height) * c_target.proj[1][1]),
		                    0.0f);

		// In-slice axis perpendicular to V, pointing along +(dx, dy).
		vec3 T = dirView - V * dot(dirView, V);
		float Tlen = length(T);
		if(Tlen < 1e-12f) continue;
		T = T / Tlen;

		// Normal projected into the slice plane, and its signed angle from V.
		vec3  sliceNormal = cross(V, T);
		vec3  projN       = N - sliceNormal * dot(N, sliceNormal);
		float projNlen    = length(projN);
		if(projNlen < 1e-6f) continue;
		vec3 projNn = projN / projNlen;

		float n     = atan2f(dot(projNn, T), dot(projNn, V));
		float cos_n = cosf(n);
		float sin_n = sinf(n);

		// Horizon search. cos(h) = 0 means "no occluder found" (horizon at pi/2).
		float cosH[2] = {0.0f, 0.0f};

		for(int side = 0; side < 2; side++){
			float sgn  = (side == 0) ? 1.0f : -1.0f;
			float best = 0.0f;

			for(int st = 1; st <= num_steps; st++){
				float t  = (float(st) - jitter) / float(num_steps);
				float px = float(x) + sgn * dx * rx * t;
				float py = float(y) + sgn * dy * ry * t;

				int sx = clamp(int(px), 0, width  - 1);
				int sy = clamp(int(py), 0, height - 1);

				float d = __uint_as_float(colorbuffer[sy * width + sx] >> 32);
				if(isinf(d) || d <= 0.0f) continue;

				vec3  S  = ssao_viewPos(px, py, d, width, height);
				vec3  ds = S - P;
				float len = length(ds);
				if(len < 1e-6f) continue;

				float c = dot(ds / len, V);

				// Distance falloff toward 0 == horizon at pi/2 == no occlusion, so a
				// receding occluder fades out smoothly instead of popping.
				float falloff = 1.0f - (len / (world_radius * thickness));
				falloff = clamp(falloff, 0.0f, 1.0f);

				c *= falloff;
				if(c > best) best = c;
			}
			cosH[side] = best;
		}

		// Signed horizon angles, clamped to the hemisphere around the normal.
		float h1 = n + fminf( acosf(clamp(cosH[0], -1.0f, 1.0f)) - n,  HALFPI);
		float h2 = n + fmaxf(-acosf(clamp(cosH[1], -1.0f, 1.0f)) - n, -HALFPI);

		visibility += projNlen * 0.25f * (gtaoArc(h1, n, cos_n, sin_n)
		                                + gtaoArc(h2, n, cos_n, sin_n));
	}

	visibility = clamp(visibility / float(num_slices), 0.0f, 1.0f);

	return clamp(1.0f - (1.0f - visibility) * intensity, 0.0f, 1.0f);
}

extern "C" __global__
void kernel_gtaoOcclusion(
	uint64_t* occlusionBuffer,
	float* ssaoShadeBuffer,
	uint32_t* normalbuffer,
	float radius_fraction,
	int   num_slices,
	int   num_steps,
	float intensity,
	float thickness
){
	auto grid = cg::this_grid();
	int x = grid.thread_index().x;
	int y = grid.thread_index().y;

	if(x >= c_target.width || y >= c_target.height) return;

	int pixelID = toFramebufferIndex(x, y, c_target.width);
	uint64_t pixel = c_target.colorbuffer[pixelID];
	float depth = __uint_as_float(pixel >> 32);

	float ao = getGTAOShadingFactor(
		c_target.colorbuffer, normalbuffer, depth, x, y,
		c_target.width, c_target.height,
		radius_fraction, num_slices, num_steps, intensity, thickness
	);

	occlusionBuffer[pixelID] = uint64_t(__float_as_uint(depth)) << 32 | uint64_t(__float_as_uint(ao));
}

extern "C" __global__
void kernel_enlarge(
	cudaSurfaceObject_t gl_desktop,
	float* ssaoShadeBuffer,
	uint64_t* fbo_enlarge,
	int width, 
	int height,
	int mouseX,
	int mouseY,
	DeviceState* state,
	bool enableEDL,
	bool enableSSAO
) {
	auto grid = cg::this_grid();

	int numPixels = width * height;

	int n = 20;
	uint64_t DEFAULT = uint64_t(__float_as_uint(INFINITY)) << 32 | 0xff0000ff;

	// Enlarge horizontally, write result to temp buffer
	process(numPixels, [&](int pixelID){
		int x = pixelID % c_target.width;
		int y = pixelID / c_target.width;

		uint64_t closest = DEFAULT;
		for(int dx = -n; dx <= n; dx++){
			
			int sx = x + dx;
			int sy = y;

			if(sx < 0 || sx >= c_target.width) continue;

			int sourcePixelID = sx + sy * c_target.width;
			uint64_t pixel = c_target.colorbuffer[sourcePixelID];
			
			// add offsets to the depth of points, based on how far they are from the center
			float depth = __uint_as_float(pixel >> 32);
			if(!isinf(depth)){
				uint64_t color = pixel & 0xffffffff;
				float f = 0.01f * abs(dx * dx) + 1.0f;
				depth = depth * f;
				pixel = (uint64_t(__float_as_uint(depth)) << 32) | color;
			}

			closest = min(closest, pixel);
		}

		if(closest != DEFAULT){
			fbo_enlarge[pixelID] = closest;
		}else{
			fbo_enlarge[pixelID] = c_target.colorbuffer[pixelID];
		}
	});

	grid.sync();

	// Enlarge vertically, write result back in main color buffer
	process(numPixels, [&](int pixelID){
		int x = pixelID % c_target.width;
		int y = pixelID / c_target.width;

		uint64_t closest = DEFAULT;
		for(int dy = -n; dy <= n; dy++){
			
			int sx = x;
			int sy = y + dy;

			if(sy < 0 || sy >= c_target.height) continue;

			int sourcePixelID = sx + sy * c_target.width;
			uint64_t pixel = fbo_enlarge[sourcePixelID];

			// add offsets to the depth of points, based on how far they are from the center
			float depth = __uint_as_float(pixel >> 32);
			if(!isinf(depth)){
				uint64_t color = pixel & 0xffffffff;
				float f = 0.01f * abs(dy * dy) + 1.0f;
				depth = depth * f;
				pixel = (uint64_t(__float_as_uint(depth)) << 32) | color;
			}

			closest = min(closest, pixel);
		}

		if(closest != DEFAULT){
			c_target.colorbuffer[pixelID] = closest;
		}else{
			c_target.colorbuffer[pixelID] = fbo_enlarge[pixelID];
		}
	});

}




extern "C" __global__
void kernel_ssaoOcclusion(
	uint64_t* occlusionBuffer,
	float* ssaoShadeBuffer,
	uint32_t* normalbuffer,
	float radius_l0, float radius_l1, float radius_l2, float radius_l3,
	float bias_l0,   float bias_l1,   float bias_l2,   float bias_l3,
	int   num_levels,
	int   samples_l0, int samples_l1, int samples_l2, int samples_l3,
	float intensity
){
	auto grid = cg::this_grid();
	int x = grid.thread_index().x;
	int y = grid.thread_index().y;

	if(x >= c_target.width || y >= c_target.height) return;

	int pixelID = toFramebufferIndex(x, y, c_target.width);
	uint64_t pixel = c_target.colorbuffer[pixelID];
	float depth = __uint_as_float(pixel >> 32);
	float focal_length = c_target.proj[1][1];

	float radii[4]  = { radius_l0,  radius_l1,  radius_l2,  radius_l3  };
	float biases[4] = { bias_l0,    bias_l1,    bias_l2,    bias_l3    };
	int   samples[4]= { samples_l0, samples_l1, samples_l2, samples_l3 };

	float ssao = getSSAOShadingFactor(
		c_target.colorbuffer, normalbuffer, depth, x, y,
		c_target.width, c_target.height, focal_length,
		radii, biases, num_levels, samples, intensity
	);

	uint64_t occ = uint64_t(__float_as_uint(depth)) << 32 | uint64_t(__float_as_uint(ssao));

	occlusionBuffer[pixelID] = occ;
}

extern "C" __global__
void kernel_ssaoBlur(
	uint64_t* occlusionBuffer,
	float* ssaoShadeBuffer
){
	auto grid = cg::this_grid();
	int x = grid.thread_index().x;
	int y = grid.thread_index().y;

	int width = c_target.width;
	int height = c_target.height;
	
	if(x >= width || y >= height) return;

	int centerIdx = y * width + x;

	// Fetch center depth for bilateral weighting
	float center_depth = __uint_as_float(occlusionBuffer[centerIdx] >> 32);

	// Background / sky pixel — no occlusion
	if(isinf(center_depth)){
		ssaoShadeBuffer[centerIdx] = 1.0f;
		return;
	}

	// Separable 1D Gaussian weights for offsets {-3, -2, -1, 0, +1, +2, +3} (sigma ≈ 1)
	const int   RADIUS = 3;
	const float gaussian[7] = { 0.015625f, 0.09375f, 0.234375f, 0.3125f, 0.234375f, 0.09375f, 0.015625f };

	// Depth-relative bilateral sigma: keeps edges sharp regardless of viewing distance.
	// Tuned so neighbours within ~1% of the centre's depth are accepted (smooths the
	// AO term across a single curved sphere) while inter-atom depth jumps are rejected
	// (prevents the AO from bleeding across silhouette edges).
	const float DEPTH_SIGMA = fmaxf(center_depth * 0.01f, 1e-3f);
	const float inv2sigma2  = 1.0f / (2.0f * DEPTH_SIGMA * DEPTH_SIGMA);

	float sum         = 0.0f;
	float totalWeight = 0.0f;

	for(int dy = -RADIUS; dy <= RADIUS; dy++)
	for(int dx = -RADIUS; dx <= RADIUS; dx++)
	{
		int nx = clamp(x + dx, 0, width  - 1);
		int ny = clamp(y + dy, 0, height - 1);
		int samplePixelID = ny * width + nx;

		uint64_t occ = occlusionBuffer[samplePixelID];
		float sample_depth     = __uint_as_float(occ >> 32);
		float sample_occlusion = __uint_as_float(occ & 0xffffffff);

		// Background neighbours skipped — they have no real occlusion contribution
		// and would drag the centre pixel toward 1.0 across silhouettes.
		if(isinf(sample_depth)) continue;

		float depth_diff   = sample_depth - center_depth;
		float depth_weight = expf(-depth_diff * depth_diff * inv2sigma2);

		float w = gaussian[dx + RADIUS] * gaussian[dy + RADIUS] * depth_weight;

		sum         += w * sample_occlusion;
		totalWeight += w;
	}

	float shade = 1.0f;
	if(totalWeight > 0.0f){
		shade = sum / totalWeight;
	}

	// shade = shade / 2.0f + 0.5f;

	ssaoShadeBuffer[centerIdx] = shade;
	// ssaoShadeBuffer[centerIdx] = 1.0f;
}


extern "C" __global__
void kernel_resolve_visbuffer_to_colorbuffer2D(
	CMesh* meshes,
	uint32_t numMeshes,
	uint64_t* triangleCountPrefixSum,
	int mouseX,
	int mouseY,
	DeviceState* state,
	RasterizationSettings rasterizationSettings,
	JpegPipeline jpp,
	SphereRasterArgs sphereArgs,
	uint32_t* normalbuffer
) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int x = grid.thread_index().x;
	int y = grid.thread_index().y;
	// int pixelID = x + c_target.width * y;
	int pixelID = toFramebufferIndex(x, y, c_target.width);

	if(x >= c_target.width) return;
	if(y >= c_target.height) return;

	mat4 viewI = inverse(c_target.view);

	float fbW = float(c_target.width);
	float fbH = float(c_target.height);

	vec3 origin    = computeRayOrigin(float(x) + 0.5f, float(y) + 0.5f, fbW, fbH);
	vec3 origin_10 = computeRayOrigin(float(x) + 1.5f, float(y) + 0.5f, fbW, fbH);
	vec3 origin_01 = computeRayOrigin(float(x) + 0.5f, float(y) + 1.5f, fbW, fbH);
	vec3 origin_11 = computeRayOrigin(float(x) + 1.5f, float(y) + 1.5f, fbW, fbH);

	vec3 rayDir    = computeRayDirection(float(x) + 0.5f, float(y) + 0.5f, fbW, fbH);
	vec3 rayDir_10 = computeRayDirection(float(x) + 1.5f, float(y) + 0.5f, fbW, fbH);
	vec3 rayDir_01 = computeRayDirection(float(x) + 0.5f, float(y) + 1.5f, fbW, fbH);
	vec3 rayDir_11 = computeRayDirection(float(x) + 1.5f, float(y) + 1.5f, fbW, fbH);

	uint64_t pixel = c_target.framebuffer[pixelID];
	uint64_t pixel_colorbuffer = c_target.colorbuffer[pixelID];
	
	float depth;
	uint32_t meshIndex;
	uint64_t totalTriangleIndex = 0;
	unpack_pixel(pixel, &depth, &totalTriangleIndex);

	float depth_colorbuffer = __uint_as_float(pixel_colorbuffer >> 32);

	if(depth_colorbuffer < depth) return;

	{// Find mesh corresponding to totalTriangleIndex via binary search
		int left = 0;
		int right = numMeshes - 1;
		int searchResult = 0;

		while(left <= right){
			int mid = left + (right - left) / 2;
			
			int64_t before = triangleCountPrefixSum[mid];
			int64_t after = (mid == (numMeshes - 1)) ? 0xffffffffff : triangleCountPrefixSum[mid + 1];

			if(totalTriangleIndex >= before && totalTriangleIndex < after){
				searchResult = mid;
				break;
			}else if(after <= totalTriangleIndex){
				left = mid + 1;
			}else{
				right = mid - 1;
			}

			if(mid == left) break;
		}

		meshIndex = searchResult;
	}
	
	uint32_t color = 0;
	uint8_t *rgb = (uint8_t *)&color;

	if(!isinf(depth)){
		CMesh mesh = meshes[meshIndex];
		uint32_t triangleIndex = totalTriangleIndex - mesh.cummulativeTriangleCount;

		if(x == mouseX && y == mouseY){
			state->hovered_meshId = mesh.id;
			state->hovered_triangleIndex = triangleIndex;
		}

		triangleIndex = min(triangleIndex, mesh.numTriangles - 1);

		if(!mesh.isLoaded){
			color = 0xffff00ff;
			goto shade_finished;
		}

		// resolve indices first
		uint32_t i0, i1, i2;
		if(!mesh.compressed){
			if(mesh.indices){
				i0 = mesh.indices[3 * triangleIndex + 0];
				i1 = mesh.indices[3 * triangleIndex + 1];
				i2 = mesh.indices[3 * triangleIndex + 2];
			}else{
				i0 = 3 * triangleIndex + 0;
				i1 = 3 * triangleIndex + 1;
				i2 = 3 * triangleIndex + 2;
			}
		}else{
			uint32_t indexRange = mesh.index_max - mesh.index_min;
			uint64_t bitsPerIndex = ceil(log2f(float(indexRange + 1)));
			i0 = BitEdit::readU32(mesh.indices, bitsPerIndex * (3 * triangleIndex + 0), bitsPerIndex) + mesh.index_min;
			i1 = BitEdit::readU32(mesh.indices, bitsPerIndex * (3 * triangleIndex + 1), bitsPerIndex) + mesh.index_min;
			i2 = BitEdit::readU32(mesh.indices, bitsPerIndex * (3 * triangleIndex + 2), bitsPerIndex) + mesh.index_min;
		}

		// Then load geometry data
		vec3 a_object = getVertex_resolved(mesh, i0);
		vec3 b_object = getVertex_resolved(mesh, i1);
		vec3 c_object = getVertex_resolved(mesh, i2);
		vec2 uv_a = getUV_resolved(mesh, i0);
		vec2 uv_b = getUV_resolved(mesh, i1);
		vec2 uv_c = getUV_resolved(mesh, i2);
		// ---------------------------------------------------


		// mat4 worldView = c_target.view * mesh.world;
		vec3 a_world = mesh.world * vec4(a_object, 1.0f);
		vec3 b_world = mesh.world * vec4(b_object, 1.0f);
		vec3 c_world = mesh.world * vec4(c_object, 1.0f);

		vec3 a_view = c_target.view * vec4(a_world, 1.0f);
		vec3 b_view = c_target.view * vec4(b_world, 1.0f);
		vec3 c_view = c_target.view * vec4(c_world, 1.0f);

		vec3 N = normalize(cross(b_view - a_view, c_view - a_view));
		
		// For mip map level:
		// - Find triangle intersection in current pixel for current uv
		// - Also find triangle intersection for pixels to the right and the top to 
		//   compute the change of uv coordinates.
		// - But actually do plane intersection because triangle may not extend to adjacent pixels
		// float t    = intersectTriangle(origin, rayDir, a, b, c, false);
		float t, t_10, t_01, t_11;
		{
			vec3 edge1 = b_world - a_world;
			vec3 edge2 = c_world - a_world;
			vec3 normal = normalize(cross(edge1, edge2));

			float d = dot(a_world, normal);

			t    = intersectPlane(origin,    rayDir,    normal, -d);
			t_10 = intersectPlane(origin_10, rayDir_10, normal, -d);
			t_01 = intersectPlane(origin_01, rayDir_01, normal, -d);
			t_11 = intersectPlane(origin_11, rayDir_11, normal, -d);
		}

		// Store 2-component barycentric coordinates because the 3rd component is deducted from the other two.
		vec2 stv, stv_10, stv_01, stv_11;
		{

			vec3 v0 = b_world - a_world;
			vec3 v1 = c_world - a_world;

			float d00 = dot(v0, v0);
			float d01 = dot(v0, v1);
			float d11 = dot(v1, v1);
			float denom = d00 * d11 - d01 * d01;
			float denomI = 1.0f / denom;
			
			// Takes the ray's own origin: under orthographic the four neighbouring rays
			// are parallel and differ only in origin, so sharing one would give all
			// four the same barycentrics and zero UV derivatives.
			auto computeSTV = [&](vec3 rayOrigin, float t, vec3 rayDir){
				vec3 p = rayOrigin + t * rayDir;
				vec3 v2 = p - a_world;

				float d20 = dot(v2, v0);
				float d21 = dot(v2, v1);

				vec2 stv;
				stv.x = (d11 * d20 - d01 * d21) * denomI;
				stv.y = (d00 * d21 - d01 * d20) * denomI;

				return stv;
			};

			stv    = computeSTV(origin,    t,    rayDir);
			stv_10 = computeSTV(origin_10, t_10, rayDir_10);
			stv_01 = computeSTV(origin_01, t_01, rayDir_01);
			stv_11 = computeSTV(origin_11, t_11, rayDir_11);
		}
		
		vec2 uv    = uv_a * (1.0f - stv.x    - stv.y   ) + uv_b * stv.x    + uv_c * stv.y;
		vec2 uv_10 = uv_a * (1.0f - stv_10.x - stv_10.y) + uv_b * stv_10.x + uv_c * stv_10.y;
		vec2 uv_01 = uv_a * (1.0f - stv_01.x - stv_01.y) + uv_b * stv_01.x + uv_c * stv_01.y;
		vec2 uv_11 = uv_a * (1.0f - stv_11.x - stv_11.y) + uv_b * stv_11.x + uv_c * stv_11.y;

		vec2 uvmax = {
			max(max(uv.x, uv_10.x), max(uv_01.x, uv_11.x)),
			max(max(uv.y, uv_10.y), max(uv_01.y, uv_11.y)),
		};
		vec2 uvmin = {
			min(min(uv.x, uv_10.x), min(uv_01.x, uv_11.x)),
			min(min(uv.y, uv_10.y), min(uv_01.y, uv_11.y)),
		};

		// Compute Mip Map level by delta of texcoords
		uv.x = uv.x - floor(uv.x);
		uv.y = uv.y - floor(uv.y);
		vec2 dx = (uvmax.x - uvmin.x) * vec2{mesh.texture.width, mesh.texture.height};
		vec2 dy = (uvmax.y - uvmin.y) * vec2{mesh.texture.width, mesh.texture.height};
		float mipLevel = 0.5f * log2(max(dot(dx, dx), dot(dy, dy)));
		mipLevel = int(mipLevel);
		mipLevel = min(mipLevel, 7.0f);

		// pick a mip map level
		int target_level = clamp(mipLevel, 0.0f, 7.0f);
		// target_level = 0;
		uint32_t* data = mesh.texture.data;
		uint32_t width = mesh.texture.width;
		uint32_t height = mesh.texture.height;
		for(int i = 1; i <= target_level; i++){
			data = data + width * height;
			width = (width + 2 - 1) / 2;
			height = (height + 2 - 1) / 2;
		}
		color = 0xffff00ff;
		
		// TODO: These blocks here are a major performance bottleneck. 
		// Removing them cuts registers from 88 to 56 and improves perf from 1.3ms to 0.88ms
		if(rasterizationSettings.displayAttribute == DisplayAttribute::NONE){
			color = 0xffffffff;
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::TRIANGLE_ID){
			color = triangleIndex * 12345678;
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::MESH_ID){
			color = (mesh.id + 1) * 12345678;
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::TEXTURE && mesh.uvs){
			if(!mesh.texture.huffmanTables){
				color = sampleColor_linear(data, width, height, uv);

				// Color frags that should have been discarded in pink
				// if(color >> 24 < 128) color = 0xffff00ff;
			}else{
				// uint32_t mcu = uvToMCUIndex(width, height, uv.x, uv.y);
				int tx = (int(uv.x * width) % width);
				int ty = (int(uv.y * height) % height);
				uint32_t mcu_x = tx / 16;
				uint32_t mcu_y = ty / 16;
				uint32_t mcus_x = width / 16;
				uint32_t mcu = mcu_x + mcus_x * mcu_y;
				uint32_t key = pack_mcuidx_textureidx_miplevel(mcu, mesh.texture.handle, uint32_t(mipLevel));

				// to avoid contention, make sure that for any MCU, only one thread per warp continues.
				{ 
					// mask of warp threads with same key
					// auto block = cg::this_thread_block();
					// auto warp = cg::tiled_partition<32>(block);
					auto warp = cg::coalesced_threads(); 
					uint32_t mask = warp.match_any(key);

					// find the lowest lane between threads with the same key
					int winningLane = __ffs(mask) - 1;

					// return early because another thread handles this MCU
					if(warp.thread_rank() != winningLane) goto shade_finished;
				}

				// Reserve a spot in the hash map. The value, the TB-Slot, will be acquired and set by the decode kernel
				bool alreadyExists = false;
				int location = 0;
				jpp.decodedMcuMap.set(key, 0, &location, &alreadyExists);

				if(!alreadyExists){
					// Add MCU to decoder queue
					uint32_t decodeIndex = atomicAdd(jpp.toDecodeCounter, 1);
					jpp.toDecode[decodeIndex] = key;
				}else{
					// MCU is in cache - flag as visible
					atomicOr(&jpp.decodedMcuMap.entries[location], 0x00000000'ff000000);
				}

			}
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::UV && mesh.uvs){
			rgb[0] = fmodf(uv.x - floor(uv.x), 1.0f) * 256.0f;
			rgb[1] = fmodf(uv.y - floor(uv.y), 1.0f) * 256.0f;
			rgb[2] = 0;
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::NORMAL && mesh.normals){
			
			vec3 n0 = mesh.normals[i0];
			vec3 n1 = mesh.normals[i1];
			vec3 n2 = mesh.normals[i2];

			n0 = mesh.world * vec4(n0, 0.0f);
			n1 = mesh.world * vec4(n1, 0.0f);
			n2 = mesh.world * vec4(n2, 0.0f);

			vec3 N = n0 * (1.0f - stv.x - stv.y) + n1 * stv.x + n2 * stv.y;
			N = normalize(N);

			rgb[0] = clamp(N.x * 255.0f, 0.0f, 255.0f);
			rgb[1] = clamp(N.y * 255.0f, 0.0f, 255.0f);
			rgb[2] = clamp(N.z * 255.0f, 0.0f, 255.0f);
			
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::NORMAL && !mesh.normals){
			vec3 edge1 = b_world - a_world;
			vec3 edge2 = c_world - a_world;
			vec3 N = normalize(cross(edge1, edge2));

			rgb[0] = clamp(N.x * 255.0f, 0.0f, 255.0f);
			rgb[1] = clamp(N.y * 255.0f, 0.0f, 255.0f);
			rgb[2] = clamp(N.z * 255.0f, 0.0f, 255.0f);
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::VERTEX_COLORS && mesh.colors){
			uint32_t C_a = mesh.colors[i0];
			uint32_t C_b = mesh.colors[i1];
			uint32_t C_c = mesh.colors[i2];
			color = C_a;

			auto toVec3 = [](uint32_t C){
				return vec3{
					(C >>  0) & 0xff,
					(C >>  8) & 0xff,
					(C >> 16) & 0xff,
				};
			};

			vec3 c_a = toVec3(C_a);
			vec3 c_b = toVec3(C_b);
			vec3 c_c = toVec3(C_c);

			vec3 c = c_a * (1.0f - stv.x - stv.y) + c_b * stv.x + c_c * stv.y;
			rgb[0] = clamp(c.x, 0.0f, 255.0f);
			rgb[1] = clamp(c.y, 0.0f, 255.0f);
			rgb[2] = clamp(c.z, 0.0f, 255.0f);
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::STAGE){
			// We don't actually know which stage a pixel was rasterized by, 
			// but we can assume based on the triangle's size and whether it intersects the near plane.
			float f = c_target.proj[1][1];
			float aspect = float(c_target.width) / float(c_target.height);

			vec3 a_ndc = viewToNDC(a_view, f, aspect);
			vec3 b_ndc = viewToNDC(b_view, f, aspect);
			vec3 c_ndc = viewToNDC(c_view, f, aspect);

			bool isNontrivial = (a_ndc.z <= 0.0f || b_ndc.z <= 0.0f || c_ndc.z <= 0.0f);

			vec2 a_screen = ndcToScreen(a_ndc, c_target.width, c_target.height);
			vec2 b_screen = ndcToScreen(b_ndc, c_target.width, c_target.height);
			vec2 c_screen = ndcToScreen(c_ndc, c_target.width, c_target.height);

			// screen-space bounding box of triangle
			float min_x = min(a_screen.x, min(b_screen.x, c_screen.x));
			float max_x = max(a_screen.x, max(b_screen.x, c_screen.x));
			float min_y = min(a_screen.y, min(b_screen.y, c_screen.y));
			float max_y = max(a_screen.y, max(b_screen.y, c_screen.y));

			// clip to screen
			min_x = max(min_x, 0.0f);
			max_x = min(max_x, float(c_target.width - 1));
			min_y = max(min_y, 0.0f);
			max_y = min(max_y, float(c_target.height - 1));

			int size_x = ceil(max_x) - floor(min_x);
			int size_y = ceil(max_y) - floor(min_y);
			int numFragments = size_x * size_y;

			int stage = 0;

			if(numFragments <= THRESHOLD_SMALL)      stage = 0;
			else if(numFragments <= THRESHOLD_LARGE) stage = 1;
			else                                     stage = 2;
			
			if(isNontrivial) stage = 2;

			color = 0xffffaaff;
			if(stage == 0) color = SCHEME_SPECTRAL[9];
			if(stage == 1) color = SCHEME_SPECTRAL[7];
			if(stage == 2) color = SCHEME_SPECTRAL[3];

			// if(numFragments <= 1)           color = SCHEME_SPECTRAL[9];
			// else if(numFragments <= 4)      color = SCHEME_SPECTRAL[8];
			// else if(numFragments <= 16)      color = SCHEME_SPECTRAL[7];
			// else if(numFragments <= 64)     color = SCHEME_SPECTRAL[6];
			// else if(numFragments <= 256)    color = SCHEME_SPECTRAL[5];
			// else if(numFragments <= 1024)    color = SCHEME_SPECTRAL[4];
			// else if(numFragments <= 4096)   color = SCHEME_SPECTRAL[3];
			// else if(numFragments <= 16384)   color = SCHEME_SPECTRAL[2];
			// else if(numFragments <= 262144)  color = SCHEME_SPECTRAL[1];
			// else                            color = SCHEME_SPECTRAL[0];



		}else{
			color = 0xff0000ff;
		}
		color = color | 0xff000000;

		if(rasterizationSettings.enableDiffuseLighting && mesh.normals){

			vec3 n0 = mesh.normals[i0];
			vec3 n1 = mesh.normals[i1];
			vec3 n2 = mesh.normals[i2];

			n0 = mesh.world * vec4(n0, 0.0f);
			n1 = mesh.world * vec4(n1, 0.0f);
			n2 = mesh.world * vec4(n2, 0.0f);

			vec3 N = normalize(n0 * (1.0f - stv.x - stv.y) + n1 * stv.x + n2 * stv.y);

			// Same image-based lighting the sphere path uses, so meshes and atoms
			// sit in the same environment instead of under two different lights.
			vec3 p_world = a_world * (1.0f - stv.x - stv.y) + b_world * stv.x + c_world * stv.y;
			vec3 V = normalize(c_target.cameraPos - p_world);

			color = shadeEnvironment(color, N, V);
		}

		// Highlight hovered mesh: draw borders
		if(rasterizationSettings.enableObjectPicking && mesh.id == state->hovered_meshId){

			bool isInside = true;

			for(int dx : {-1, 0, 1})
			for(int dy : {-1, 0, 1})
			{

				int nx = clamp(x + dx, 0, c_target.width - 1);
				int ny = clamp(y + dy, 0, c_target.height - 1);
				int neighbor_pixelID = toFramebufferIndex(nx, ny, c_target.width);
				uint64_t neighbor_pixel = c_target.framebuffer[neighbor_pixelID];

				float neighbor_depth;
				uint32_t neighbor_meshIndex;
				uint64_t neighbor_totalTriangleIndex = 0;
				unpack_pixel(neighbor_pixel, &neighbor_depth, &neighbor_totalTriangleIndex);

				// Check if neighbor's total triangle index is within this pixel's mesh
				if(neighbor_totalTriangleIndex < mesh.cummulativeTriangleCount) isInside = false;
				if(neighbor_totalTriangleIndex >= mesh.cummulativeTriangleCount + mesh.numTriangles) isInside = false;
			}

			if(!isInside){
				color = 0xff0000ff;
			}
		}
	}else{
		color = 0;
		// rgb[0] = 0.1f * 256.0f;
		// rgb[1] = 0.2f * 256.0f;
		// rgb[2] = 0.3f * 256.0f;

		if(x == mouseX && y == mouseY){
			state->hovered_meshId = 0xffffffff;
			state->hovered_triangleIndex = 0xffffffff;
		}
	}

	shade_finished:

	// WIREFRAME
	if(rasterizationSettings.showWireframe){
		uint32_t numPixels = c_target.width * c_target.height;
		int pid_00 = clamp(toFramebufferIndex(x + 0, y + 0, c_target.width), 0u, numPixels - 1);
		int pid_01 = clamp(toFramebufferIndex(x + 0, y + 1, c_target.width), 0u, numPixels - 1);
		int pid_10 = clamp(toFramebufferIndex(x + 1, y + 0, c_target.width), 0u, numPixels - 1);

		uint32_t p_00 = c_target.framebuffer[pid_00] & 0xffffffff;
		uint32_t p_01 = c_target.framebuffer[pid_01] & 0xffffffff;
		uint32_t p_10 = c_target.framebuffer[pid_10] & 0xffffffff;

		if(p_00 != p_10) color = 0xffff00ff;
		if(p_00 != p_01) color = 0xffff00ff;

		for(int dx : {0, 1})
		for(int dy : {0, 1})
		// for(int dx : {-1, 0, 1})
		// for(int dy : {-1, 0, 1})
		{
			int pid_neighbor = clamp(toFramebufferIndex(x + dx, y + dy, c_target.width), 0u, numPixels - 1);
			uint32_t p_neighbor = c_target.framebuffer[pid_neighbor] & 0xffffffff;

			if(p_00 != p_neighbor) color = 0xffff00ff;
		}
	}

	// 64x64 tile grid
	// if(x % 64 == 0 || y % 64 == 0){
	// 	color = 0xffff00ff;
	// }

	

	// Sphere composite — check if a sphere is closer than the triangle
	if(sphereArgs.sphere_framebuffer != nullptr && sphereArgs.numSpheres > 0) {
		uint64_t sphere_val = sphereArgs.sphere_framebuffer[pixelID];
		// The visbuffer stores index+1, so 0 means "no sphere" just as the all-ones
		// sentinel does. Subtracting 1 from a raw 0 wraps to 0xFFFFFFFF and indexes
		// ~17 GB past `colors`, which faults; the same wrap into the 4x smaller
		// atomTypes buffer often lands in mapped memory and passes silently. Bounds
		// check once here so every per-atom array below is safe.
		uint32_t sphere_raw = (uint32_t)(sphere_val & 0xFFFFFFFFull);
		if(sphere_val != 0xFFFFFFFFFFFFFFFFull && sphere_raw != 0u
			&& (sphere_raw - 1u) < sphereArgs.numSpheres) {
			uint32_t sphere_idx = sphere_raw - 1u;

			vec3 center_world = sphereArgs.positions[sphere_idx];
			float radius      = sphereArgs.radii[sphere_idx];

			// Mirror the LOD scaling used by the rasterizer so the ray-sphere
			// intersection covers the same painted footprint. Without this, pixels
			// inside the LOD-enlarged disc but outside the original radius would
			// miss the geometric sphere and fall through to the sub-pixel fallback.
			if(sphereArgs.lod.numLevels > 0) {
				float dist = length(c_target.cameraPos - center_world);
				radius = sphereLodScaledRadius(sphereArgs.lod, sphere_idx, radius, dist);
			}

			vec3 oc  = origin - center_world;
			float b  = dot(oc, rayDir);
			float cq = dot(oc, oc) - radius * radius;
			float disc = b * b - cq;

			float sphere_depth;
			vec3  N;
			bool  shade_sphere = false;
			uint32_t normalTag = NORMAL_TAG_ANALYTIC;

			if(disc >= 0.0f) {
				float t = -b - sqrtf(disc);
				if(t > 0.0f) {
					vec3 hit_world = origin + t * rayDir;
					vec4 hit_view  = c_target.view * vec4(hit_world, 1.0f);
					sphere_depth = -hit_view.z;
					N = normalize(hit_world - center_world);
					shade_sphere = true;
				}
			}
			if(!shade_sphere) {
				// Sub-pixel atom: the rasteriser painted at least a half-pixel disc so
				// the atom is not lost, but the ray misses the actual sphere.
				//
				// The old fallback was N = -rayDir, a normal facing straight back at the
				// camera. Every such pixel then got identical shading, and since the
				// normal G-buffer feeds GTAO, identical normals there too. At whole-cell
				// framing, where most atoms are sub-pixel, that is a large part of why
				// the image reads flat.
				//
				// Blend by how badly the ray missed, because the two limits want
				// different answers:
				//   near miss (the antialiased rim of a resolved atom) -> the true
				//     surface normal there is grazing, perpendicular to the ray;
				//   large miss (a genuinely sub-pixel atom painted at the rasteriser's
				//     half-pixel floor) -> the pixel covers the whole sphere, and the
				//     area-weighted average normal over the visible hemisphere really
				//     is camera-facing.
				// Picking either one alone is wrong at the other end: -rayDir everywhere
				// flattens the rim, and the grazing normal everywhere makes every
				// sub-pixel atom read as a dark silhouette.
				sphere_depth = __uint_as_float((uint32_t)(sphere_val >> 32));

				vec3  closest = oc + rayDir * (-b);   // centre -> nearest point on the ray
				float cl      = length(closest);
				float missRatio = (radius > 0.0f) ? (cl / radius) : 2.0f;
				float grazing   = clamp(2.0f - missRatio, 0.0f, 1.0f);

				vec3 nGraze  = (cl > 1e-8f) ? (closest / cl) : (-rayDir);
				vec3 blended = nGraze * grazing + (-rayDir) * (1.0f - grazing);
				float bl = length(blended);
				N = (bl > 1e-8f) ? (blended / bl) : (-rayDir);

				normalTag = NORMAL_TAG_FALLBACK;
				shade_sphere = true;
			}

			if(shade_sphere && sphere_depth < depth) {
				// Two color paths. Element coloring (default theme) is the
				// 1-byte-per-atom + 256-entry palette path — saves 4× the GPU
				// memory of the legacy uint32 buffer. Chain/entity themes still
				// use the per-atom uint32 buffer because their key spaces don't
				// fit in a uint8.
				// `colors` must be tested FIRST. atomTypes + palette are allocated once at
				// load and stay non-null for the lifetime of the node, so testing them
				// first made the colors branch unreachable and the chain / entity themes
				// silently had no effect. applyColorTheme only allocates colors for
				// CHAIN / ENTITY and frees it again for ELEMENT, so a non-null colors
				// buffer is exactly the signal that a non-element theme is active.
				uint32_t base;
				if(sphereArgs.colors != nullptr){
					base = sphereArgs.colors[sphere_idx];
				}else if(sphereArgs.atomTypes != nullptr && sphereArgs.colorPalette != nullptr){
					base = sphereArgs.colorPalette[sphereArgs.atomTypes[sphere_idx]];
				}else{
					base = 0xFFFFFFFFu;
				}
				// Image-based lighting from the environment SH, plus a directional
				// specular key and a Fresnel rim. V is the direction back toward the
				// eye, which for a primary ray is just -rayDir.
				// Flat shading: albedo straight through, no directional term. AO and the
				// halo still apply in the composite, which is the illustrative look --
				// shape comes from occlusion and outlines rather than from a light.
				// Baked per-atom AO (QuteMol style). Object-space and view-independent, so
				// it carries the large-scale enclosure that a screen-space method cannot see
				// at whole-cell framing, where ~90 atoms share a pixel and the depth buffer
				// is atom noise. Multiplies with GTAO rather than replacing it.
				// Baked AO is NOT multiplied into the albedo here. Doing that stacked it
				// with the screen-space AO applied later in the composite, and inside a
				// packed cell both terms say "occluded" about the same geometry -- the
				// product drove the interior to 26/255 with 9% of pixels near black. It is
				// carried to the composite instead, which takes the minimum of the two so
				// the most-occluded estimate wins without ever double counting.
				float bakedAO = 1.0f;
				if(sphereArgs.atomAO != nullptr){
					bakedAO = float(sphereArgs.atomAO[sphere_idx]) * (1.0f / 255.0f);
					// Applied here rather than baked in, so it retunes without a re-bake.
					float f = clamp(c_shade.atomAOFloor, 0.0f, 1.0f);
					bakedAO = f + (1.0f - f) * bakedAO;
				}

				color = shadeEnvironment(base, N, -rayDir);
				depth = sphere_depth;

				// Stash the analytic sphere normal (in SSAO view space) for the AO
				// pass. This is the dominant fix for "splotchy" AO on dense sphere
				// scenes — depth-gradient reconstruction only works on smooth
				// surfaces and breaks at every silhouette edge.
				if(normalbuffer != nullptr){
					normalbuffer[pixelID] = packNormalGBuf(worldNormalToSsaoView(N), normalTag, bakedAO);
				}
			}
		}
	}

	if(depth != Infinity){
		uint64_t udepth = __float_as_uint(depth);
		uint64_t pixel = (udepth << 32) | color;
		c_target.colorbuffer[pixelID] = pixel;
	}else{
		c_target.colorbuffer[pixelID] = uint64_t(__float_as_uint(INFINITY)) << 32 | 0;
	}
}

extern "C" __global__
void kernel_resolve_colorbuffer_to_opengl_2D(
	cudaSurfaceObject_t gl_desktop,
	float* ssaoShadeBuffer,
	int width, 
	int height,
	int mouseX,
	int mouseY,
	DeviceState* state,
	bool enableEDL,
	bool enableSSAO,
	bool showInset,
	uint32_t backgroundColor,
	uint32_t* normalbuffer
) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	RenderTarget& source = c_target;

	if(width == source.width && height == source.height){
		int x = grid.thread_index().x;
		int y = grid.thread_index().y;
		int pixelID = toFramebufferIndex(x, y, source.width);

		if(x >= source.width) return;
		if(y >= source.height) return;

		auto sample = [&](int x, int y){
			if(x < 0) return uint32_t(0);
			if(y < 0) return uint32_t(0);
			if(x >= source.width) return uint32_t(0);
			if(y >= source.height) return uint32_t(0);

			int pixelID = toFramebufferIndex(x, y, source.width);

			uint64_t pixel = c_target.colorbuffer[pixelID];
			float depth = __uint_as_float(pixel >> 32);
			uint32_t sampleColor = pixel & 0xffffffff;

			float edl = 1.0f;
			float ssao = 1.0f;

			if(enableEDL){
				edl = getEdlShadingFactor(c_target.colorbuffer, depth, x, y, 1);
			}

			if(enableSSAO){
				ssao = applyAO(combineAO(ssaoShadeBuffer[pixelID], normalbuffer, pixelID));
			}

			if(isinf(depth)){
				uint32_t envColor;
				sampleColor = envBackgroundColor(x, y, &envColor) ? envColor : backgroundColor;
			}
			sampleColor = applyHalo(sampleColor, haloStrengthAt(x, y, depth));

			if(c_shade.debugView != CURAST_DEBUG_OFF){
				return debugViewColor(x, y, pixelID, depth, ssaoShadeBuffer, normalbuffer);
			}

			float shade = edl * ssao;

			uint8_t* rgba = (uint8_t*)&sampleColor;
			rgba[0] = shade * float(rgba[0]);
			rgba[1] = shade * float(rgba[1]);
			rgba[2] = shade * float(rgba[2]);
			rgba[3] = 255;

			return sampleColor;
		};

		uint32_t color = sample(x, y);
		surf2Dwrite(color, gl_desktop, x * 4, y);

		if(showInset){
			struct Rect{
				float x;
				float y;
				float width;
				float height;
			};
			float insetSize = 32;
			Rect insetSource = {
				mouseX - insetSize / 2,
				mouseY - insetSize / 2,
				insetSize,
				insetSize,
			};
			Rect insetTarget = {
				0, 0,
				insetSize * 16, insetSize * 16
			};

			float u = (float(x) - insetTarget.x) / insetTarget.width;
			float v = (float(y) - insetTarget.y) / insetTarget.height;

			if((u >= 0.0f && u <= 1.0f) && (v >= 0.0f && v <= 1.0f))
			{
				int source_x = insetSource.x + u * insetSize;
				int source_y = insetSource.y + v * insetSize;

				color = sample(source_x, source_y);

				if(u == 1.0f || v == 1.0f){
					color = 0xffff00ff;
				}
			}else if(
				x == int(insetSource.x)
				|| x == int(insetSource.x + insetSize)
				|| y == int(insetSource.y)
				|| y == int(insetSource.y + insetSize)
			){
				color = 0xff0000ff;
			}

			surf2Dwrite(color, gl_desktop, x * 4, y);
		}


	}else{
		int target_x = grid.thread_index().x;
		int target_y = grid.thread_index().y;
		int pixelID = toFramebufferIndex(target_x, target_y, width);

		if(target_x >= width) return;
		if(target_y >= height) return;

		vec4 color = {0.0f, 0.0f, 0.0f, 0.0f};
		float edl = 0.0f;
		float ssao = 0.0f;

		int supersamplingFactor = source.width / width;

		int numSamples = 0;
		for(int dx = 0; dx < supersamplingFactor; dx++)
		for(int dy = 0; dy < supersamplingFactor; dy++)
		{
			int source_x = supersamplingFactor * target_x + dx;
			int source_y = supersamplingFactor * target_y + dy;
			int sourcePixelID = toFramebufferIndex(source_x, source_y, source.width);

			uint64_t pixel = c_target.colorbuffer[sourcePixelID];
			float depth = __uint_as_float(pixel >> 32);
			uint32_t C = pixel & 0xffffffff;
			color.r += (C >>  0) & 0xff;
			color.g += (C >>  8) & 0xff;
			color.b += (C >> 16) & 0xff;

			if(isinf(depth)){
				color.r += (BACKGROUND_COLOR >>  0) & 0xff;
				color.g += (BACKGROUND_COLOR >>  8) & 0xff;
				color.b += (BACKGROUND_COLOR >> 16) & 0xff;
			}

			if(enableEDL){
				edl += getEdlShadingFactor(c_target.colorbuffer, depth, source_x, source_y, supersamplingFactor);
			}
			if(enableSSAO){
				ssao += ssaoShadeBuffer[sourcePixelID];
			}
			numSamples++;
		}

		if(enableEDL){
			edl = edl / float(numSamples);
		}else{
			edl = 1.0f;
		}

		if(enableSSAO){
			ssao = applyAO(ssao / float(numSamples));
		}else{
			ssao = 1.0f;
		}
		

		float shade = edl * ssao;
		color = shade * color / float(numSamples);
		uint32_t C;
		uint8_t* rgba = (uint8_t*)&C;
		rgba[0] = clamp(color.r, 0.0f, 255.0f);
		rgba[1] = clamp(color.g, 0.0f, 255.0f);
		rgba[2] = clamp(color.b, 0.0f, 255.0f);
		rgba[3] = 255;

		surf2Dwrite(C, gl_desktop, target_x * 4, target_y);
	}
}


extern "C" __global__
void kernel_resolve_colorbuffer_to_screenshot(
	uint32_t* screenshot,
	float* ssaoShadeBuffer,
	bool enableEDL,
	bool enableSSAO,
	int windowWidth,
	int windowHeight,
	uint32_t backgroundColor,
	uint32_t* normalbuffer
) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	RenderTarget& source = c_target;

	int x = grid.thread_index().x;
	int y = grid.thread_index().y;
	int pixelID = toFramebufferIndex(x, y, source.width);

	if(x >= source.width) return;
	if(y >= source.height) return;

	uint64_t pixel = c_target.colorbuffer[pixelID];
	float depth = __uint_as_float(pixel >> 32);
	uint32_t color = pixel & 0xffffffff;

	// Diagnostic views. Judging AO or normals through albedo and lighting hides
	// whether either carries any signal at all.
	if(c_shade.debugView != CURAST_DEBUG_OFF){
		screenshot[pixelID] = debugViewColor(x, y, pixelID, depth, ssaoShadeBuffer, normalbuffer);
		return;
	}

	float edl = 1.0f;
	float ssao = 1.0f;

	if(enableEDL){
		int supersamplingFactor = source.width / windowWidth;
		edl = getEdlShadingFactor(c_target.colorbuffer, depth, x, y, supersamplingFactor);
	}

	if(enableSSAO){
		ssao = applyAO(combineAO(ssaoShadeBuffer[pixelID], normalbuffer, pixelID));
	}

	if(isinf(depth)){
		uint32_t envColor;
		color = envBackgroundColor(x, y, &envColor) ? envColor : backgroundColor;
	}
	color = applyHalo(color, haloStrengthAt(x, y, depth));

	float shade = edl * ssao;
	uint8_t* rgba = (uint8_t*)&color;
	rgba[0] = shade * float(rgba[0]);
	rgba[1] = shade * float(rgba[1]);
	rgba[2] = shade * float(rgba[2]);

	// surf2Dwrite(color, gl_desktop, x * 4, y);
	screenshot[pixelID] = color;
	
}


uint32_t sampleJpeg_nearest(
	uint32_t texID,
	vec2 uv, 
	Texture* textures, 
	uint32_t* decoded,
	HashMap& decodedMcuMap,
	uint32_t mipLevel
){

	uint32_t color = 0;
	auto tex = &textures[texID + mipLevel];

	int tx = (int(uv.x * tex->width) % tex->width);
	int ty = (int(uv.y * tex->height) % tex->height);
	uint32_t mcu_x = tx / 16;
	uint32_t mcu_y = ty / 16;
	uint32_t mcus_x = tex->width / 16;
	uint32_t mcu = mcu_x + mcus_x * mcu_y;

	// int mcu = tx / 16 + ty / 16 * tex->width / 16;
	// uint32_t key = ((mcu & 0xffff) << 16) | (texID & 0xffff);
	uint32_t key = pack_mcuidx_textureidx_miplevel(mcu, texID, mipLevel);


	uint32_t value;
	if(decodedMcuMap.get(key, &value)){
		uint32_t decodedMcuIndex = value & 0x00ffffff;
		bool isNewlyDecoded = (value >> 31) != 0;

		int tx = (int(uv.x * tex->width) % tex->width);
		int ty = (int(uv.y * tex->height) % tex->height);

		tx %= 16;
		ty %= 16;
		int offset = tx % 8 + (tx / 8) * 64 + (ty % 8) * 8 + (ty / 8) * 128;
		color = decoded[decodedMcuIndex * 256 + offset];

	}else{
		color = 0x00000000;
	}

	return color;
}

// retrieve the 4 texels around the given uv coordinate
void getTexels(
	uint32_t texID,
	vec2 uv, 
	Texture* tex, 
	uint32_t mipLevel,
	uint32_t* decoded,
	HashMap& decodedMcuMap,
	vec4* t00,
	vec4* t01,
	vec4* t10,
	vec4* t11
){
	float ftx = fmodf(uv.x, 1.0f) * float(tex->width);
	float fty = fmodf(uv.y, 1.0f) * float(tex->height);

	float ftlx = fmodf(ftx, 16.0f);
	float ftly = fmodf(fty, 16.0f);

	*t00 = {0.0f, 0.0f, 0.0f, 255.0f};
	*t01 = {0.0f, 0.0f, 0.0f, 255.0f};
	*t10 = {0.0f, 0.0f, 0.0f, 255.0f};
	*t11 = {0.0f, 0.0f, 0.0f, 0.0f};

	auto toVec4 = [](uint32_t color){
		return vec4{
			(color >>  0) & 0xff,
			(color >>  8) & 0xff,
			(color >> 16) & 0xff,
			(color >> 24) & 0xff,
		}; 
	};

	if(ftlx > 0.5f && ftlx < 15.5f && ftly > 0.5f && ftly < 15.5f){
		// Easy and fast case: All texels in same MCU

		int tx = ftx - 0.5f;
		int ty = fty - 0.5f;

		uint32_t mcu_x = tx / 16;
		uint32_t mcu_y = ty / 16;
		uint32_t mcus_x = tex->width / 16;
		uint32_t mcu = mcu_x + mcus_x * mcu_y;
		// uint32_t key = ((mcu & 0xffff) << 16) | (texID & 0xffff);
		uint32_t key = pack_mcuidx_textureidx_miplevel(mcu, tex->handle - mipLevel, mipLevel);

		uint32_t value;
		if(decodedMcuMap.get(key, &value)){
			uint32_t decodedMcuIndex = value & 0x00ffffff;
			
			tx %= 16;
			ty %= 16;
			int offset_00 = (tx + 0) % 8 + ((tx + 0) / 8) * 64 + ((ty + 0) % 8) * 8 + ((ty + 0) / 8) * 128;
			int offset_01 = (tx + 0) % 8 + ((tx + 0) / 8) * 64 + ((ty + 1) % 8) * 8 + ((ty + 1) / 8) * 128;
			int offset_10 = (tx + 1) % 8 + ((tx + 1) / 8) * 64 + ((ty + 0) % 8) * 8 + ((ty + 0) / 8) * 128;
			int offset_11 = (tx + 1) % 8 + ((tx + 1) / 8) * 64 + ((ty + 1) % 8) * 8 + ((ty + 1) / 8) * 128;

			*t00 = toVec4(decoded[decodedMcuIndex * 256 + offset_00]);
			*t01 = toVec4(decoded[decodedMcuIndex * 256 + offset_01]);
			*t10 = toVec4(decoded[decodedMcuIndex * 256 + offset_10]);
			*t11 = toVec4(decoded[decodedMcuIndex * 256 + offset_11]);

			// *t00 = {0.0f, 1.0f, 0.0f, 255.0f};
		}
	}else{

		// Trickier case: texels reside in adjacent MCUs, which may or may not be available.
		uint32_t v_00, v_01, v_10, v_11 = 0;

		// return;

		auto texelCoordToKey = [&](int tx, int ty){
			uint32_t mcu_x = tx / 16;
			uint32_t mcu_y = ty / 16;
			uint32_t mcus_x = tex->width / 16;
			uint32_t mcu = mcu_x + mcus_x * mcu_y;
			// uint32_t key = ((mcu & 0xffff) << 16) | (texID & 0xffff);
			uint32_t key = pack_mcuidx_textureidx_miplevel(mcu, tex->handle - mipLevel, mipLevel);
			return key;
		};

		bool v00Exists = decodedMcuMap.get(texelCoordToKey(ftx - 0.5f, fty - 0.5f), &v_00);
		bool v01Exists = decodedMcuMap.get(texelCoordToKey(ftx - 0.5f, fty + 0.5f), &v_01);
		bool v10Exists = decodedMcuMap.get(texelCoordToKey(ftx + 0.5f, fty - 0.5f), &v_10);
		bool v11Exists = decodedMcuMap.get(texelCoordToKey(ftx + 0.5f, fty + 0.5f), &v_11);

		auto toTexel = [&](uint32_t value, int tx, int ty){
			uint32_t decodedMcuIndex = value & 0x00ffffff;

			tx %= 16;
			ty %= 16;
			int offset = tx % 8 + (tx / 8) * 64 + (ty% 8) * 8 + (ty / 8) * 128;

			vec4 color = toVec4(decoded[decodedMcuIndex * 256 + offset]);

			return color;
		};

		// If a texel's MCU is not decoded, clamp to one of the decoded MCUs
		
		// texel 00
		if (v00Exists)        *t00 = toTexel(v_00, ftx - 0.5f, fty - 0.5f);
		else *t00 = vec4(0,0,0,0);
		
		// texel 01
		if(v01Exists)        *t01 = toTexel(v_01, ftx - 0.5f, fty + 0.5f);
		else *t01 = vec4(0, 0, 0, 0);
		// texel 10
		if(v10Exists)        *t10 = toTexel(v_10, ftx + 0.5f, fty - 0.5f);
		else *t10 = vec4(0, 0, 0, 0);
		// texel 11
		if(v11Exists)        *t11 = toTexel(v_11, ftx + 0.5f, fty + 0.5f);
		else *t11 = vec4(0, 0, 0, 0);
	}
}

uint32_t sampleJpeg_linear(
	uint32_t texID,
	vec2 uv,
	Texture* textures,
	uint32_t* decoded,
	HashMap& decodedMcuMap,
	uint32_t mipLevel
) {
	uint32_t color = 0xff000000;
	uint8_t* rgba = (uint8_t*)&color;
	
	auto tex = &textures[texID + mipLevel];

	float ftx = fmodf(uv.x - 0.5f / float(tex->width), 1.0f) * float(tex->width);
	float fty = fmodf(uv.y - 0.5f / float(tex->height), 1.0f) * float(tex->height);

	float wx = fmodf(ftx, 1.0f);
	float wy = fmodf(fty, 1.0f);

	// --- read bilinear samples from mip i0 ---
	vec4 t00, t01, t10, t11;
	getTexels(texID + mipLevel, uv, tex, mipLevel, decoded, decodedMcuMap, &t00, &t01, &t10, &t11);
	vec4 result =
		(1.0f - wx) * (1.0f - wy) * t00 +
		wx * (1.0f - wy) * t10 +
		(1.0f - wx) * wy * t01 +
		wx * wy * t11;

	rgba[0] = result.r;
	rgba[1] = result.g;
	rgba[2] = result.b;
	rgba[3] = 255;

	return color;
}

extern "C" __global__
void kernel_resolve_jpeg(
	CMesh* meshes,
	uint32_t numMeshes,
	uint64_t* triangleCountPrefixSum,
	int mouseX,
	int mouseY,
	DeviceState* state,
	RasterizationSettings rasterizationSettings,
	JpegPipeline jpp,
	Texture* textures
) {
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();

	int x = grid.thread_index().x;
	int y = grid.thread_index().y;
	int pixelID = toFramebufferIndex(x, y, c_target.width);

	if(x >= c_target.width) return;
	if(y >= c_target.height) return;

	mat4 viewI = inverse(c_target.view);

	float fbW = float(c_target.width);
	float fbH = float(c_target.height);

	vec3 origin    = computeRayOrigin(float(x) + 0.5f, float(y) + 0.5f, fbW, fbH);
	vec3 origin_10 = computeRayOrigin(float(x) + 1.5f, float(y) + 0.5f, fbW, fbH);
	vec3 origin_01 = computeRayOrigin(float(x) + 0.5f, float(y) + 1.5f, fbW, fbH);
	vec3 origin_11 = computeRayOrigin(float(x) + 1.5f, float(y) + 1.5f, fbW, fbH);

	vec3 rayDir    = computeRayDirection(float(x) + 0.5f, float(y) + 0.5f, fbW, fbH);
	vec3 rayDir_10 = computeRayDirection(float(x) + 1.5f, float(y) + 0.5f, fbW, fbH);
	vec3 rayDir_01 = computeRayDirection(float(x) + 0.5f, float(y) + 1.5f, fbW, fbH);
	vec3 rayDir_11 = computeRayDirection(float(x) + 1.5f, float(y) + 1.5f, fbW, fbH);

	uint64_t pixel = c_target.framebuffer[pixelID];
	uint64_t pixel_colorbuffer = c_target.colorbuffer[pixelID];
	
	float depth;
	uint32_t meshIndex;
	uint64_t totalTriangleIndex = 0;
	unpack_pixel(pixel, &depth, &totalTriangleIndex);

	float depth_colorbuffer = __uint_as_float(pixel_colorbuffer >> 32);

	if(depth_colorbuffer < depth) return;

	{// Find mesh corresponding to totalTriangleIndex via binary search
		int left = 0;
		int right = numMeshes - 1;
		int searchResult = 0;

		while(left <= right){
			int mid = left + (right - left) / 2;
			
			int64_t before = triangleCountPrefixSum[mid];
			int64_t after = (mid == (numMeshes - 1)) ? 0xffffffffff : triangleCountPrefixSum[mid + 1];

			if(totalTriangleIndex >= before && totalTriangleIndex < after){
				searchResult = mid;
				break;
			}else if(after <= totalTriangleIndex){
				left = mid + 1;
			}else{
				right = mid - 1;
			}

			if(mid == left) break;
		}

		meshIndex = searchResult;
	}

	uint32_t color = 0;
	uint8_t *rgb = (uint8_t *)&color;

	if(!isinf(depth)){
		CMesh mesh = meshes[meshIndex];
		uint32_t triangleIndex = totalTriangleIndex - mesh.cummulativeTriangleCount;

		if(x == mouseX && y == mouseY){
			state->hovered_meshId = mesh.id;
			state->hovered_triangleIndex = triangleIndex;
		}

		triangleIndex = min(triangleIndex, mesh.numTriangles - 1);

		if(!mesh.isLoaded){
			color = 0xffff00ff;
			goto shade_finished;
		}

		// resolve indices first
		uint32_t i0, i1, i2;
		if(!mesh.compressed){
			if(mesh.indices){
				i0 = mesh.indices[3 * triangleIndex + 0];
				i1 = mesh.indices[3 * triangleIndex + 1];
				i2 = mesh.indices[3 * triangleIndex + 2];
			}else{
				i0 = 3 * triangleIndex + 0;
				i1 = 3 * triangleIndex + 1;
				i2 = 3 * triangleIndex + 2;
			}
		}else{
			uint32_t indexRange = mesh.index_max - mesh.index_min;
			uint64_t bitsPerIndex = ceil(log2f(float(indexRange + 1)));
			i0 = BitEdit::readU32(mesh.indices, bitsPerIndex * (3 * triangleIndex + 0), bitsPerIndex) + mesh.index_min;
			i1 = BitEdit::readU32(mesh.indices, bitsPerIndex * (3 * triangleIndex + 1), bitsPerIndex) + mesh.index_min;
			i2 = BitEdit::readU32(mesh.indices, bitsPerIndex * (3 * triangleIndex + 2), bitsPerIndex) + mesh.index_min;
		}

		// Then load geometry data
		vec3 a_object = getVertex_resolved(mesh, i0);
		vec3 b_object = getVertex_resolved(mesh, i1);
		vec3 c_object = getVertex_resolved(mesh, i2);
		vec2 uv_a = getUV_resolved(mesh, i0);
		vec2 uv_b = getUV_resolved(mesh, i1);
		vec2 uv_c = getUV_resolved(mesh, i2);
		// ---------------------------------------------------


		// mat4 worldView = c_target.view * mesh.world;
		vec3 a_world = mesh.world * vec4(a_object, 1.0f);
		vec3 b_world = mesh.world * vec4(b_object, 1.0f);
		vec3 c_world = mesh.world * vec4(c_object, 1.0f);

		vec3 a_view = c_target.view * vec4(a_world, 1.0f);
		vec3 b_view = c_target.view * vec4(b_world, 1.0f);
		vec3 c_view = c_target.view * vec4(c_world, 1.0f);

		vec3 N = normalize(cross(b_view - a_view, c_view - a_view));
		
		// For mip map level:
		// - Find triangle intersection in current pixel for current uv
		// - Also find triangle intersection for pixels to the right and the top to 
		//   compute the change of uv coordinates.
		// - But actually do plane intersection because triangle may not extend to adjacent pixels
		// float t    = intersectTriangle(origin, rayDir, a, b, c, false);
		float t, t_10, t_01, t_11;
		{
			vec3 edge1 = b_world - a_world;
			vec3 edge2 = c_world - a_world;
			vec3 normal = normalize(cross(edge1, edge2));

			float d = dot(a_world, normal);

			t    = intersectPlane(origin,    rayDir,    normal, -d);
			t_10 = intersectPlane(origin_10, rayDir_10, normal, -d);
			t_01 = intersectPlane(origin_01, rayDir_01, normal, -d);
			t_11 = intersectPlane(origin_11, rayDir_11, normal, -d);
		}

		// Store 2-component barycentric coordinates because the 3rd component is deducted from the other two.
		vec2 stv, stv_10, stv_01, stv_11;
		{

			vec3 v0 = b_world - a_world;
			vec3 v1 = c_world - a_world;

			float d00 = dot(v0, v0);
			float d01 = dot(v0, v1);
			float d11 = dot(v1, v1);
			float denom = d00 * d11 - d01 * d01;
			float denomI = 1.0f / denom;
			
			// Takes the ray's own origin: under orthographic the four neighbouring rays
			// are parallel and differ only in origin, so sharing one would give all
			// four the same barycentrics and zero UV derivatives.
			auto computeSTV = [&](vec3 rayOrigin, float t, vec3 rayDir){
				vec3 p = rayOrigin + t * rayDir;
				vec3 v2 = p - a_world;

				float d20 = dot(v2, v0);
				float d21 = dot(v2, v1);

				vec2 stv;
				stv.x = (d11 * d20 - d01 * d21) * denomI;
				stv.y = (d00 * d21 - d01 * d20) * denomI;

				return stv;
			};

			stv    = computeSTV(origin,    t,    rayDir);
			stv_10 = computeSTV(origin_10, t_10, rayDir_10);
			stv_01 = computeSTV(origin_01, t_01, rayDir_01);
			stv_11 = computeSTV(origin_11, t_11, rayDir_11);
		}
		
		vec2 uv    = uv_a * (1.0f - stv.x    - stv.y   ) + uv_b * stv.x    + uv_c * stv.y;
		vec2 uv_10 = uv_a * (1.0f - stv_10.x - stv_10.y) + uv_b * stv_10.x + uv_c * stv_10.y;
		vec2 uv_01 = uv_a * (1.0f - stv_01.x - stv_01.y) + uv_b * stv_01.x + uv_c * stv_01.y;
		vec2 uv_11 = uv_a * (1.0f - stv_11.x - stv_11.y) + uv_b * stv_11.x + uv_c * stv_11.y;

		vec2 uvmax = {
			max(max(uv.x, uv_10.x), max(uv_01.x, uv_11.x)),
			max(max(uv.y, uv_10.y), max(uv_01.y, uv_11.y)),
		};
		vec2 uvmin = {
			min(min(uv.x, uv_10.x), min(uv_01.x, uv_11.x)),
			min(min(uv.y, uv_10.y), min(uv_01.y, uv_11.y)),
		};

		// Compute Mip Map level by delta of texcoords
		vec2 dx = (uvmax.x - uvmin.x) * vec2{mesh.texture.width, mesh.texture.height};
		vec2 dy = (uvmax.y - uvmin.y) * vec2{mesh.texture.width, mesh.texture.height};
		float mipLevel = 0.5f * log2(max(dot(dx, dx), dot(dy, dy)));
		mipLevel = int(mipLevel);
		mipLevel = min(mipLevel, 7.0f);

		// pick a mip map level
		int target_level = clamp(mipLevel, 0.0f, 7.0f);
		// target_level = 0;
		uint32_t* data = mesh.texture.data;
		uint32_t width = mesh.texture.width;
		uint32_t height = mesh.texture.height;
		for(int i = 1; i <= target_level; i++){
			data = data + width * height;
			width = (width + 2 - 1) / 2;
			height = (height + 2 - 1) / 2;
		}
		color = 0xffff00ff;
		
		// TODO: These blocks here are a major performance bottleneck. 
		// Removing them cuts registers from 88 to 56 and improves perf from 1.3ms to 0.88ms
		if(rasterizationSettings.displayAttribute == DisplayAttribute::NONE){
			color = 0xffffffff;
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::TRIANGLE_ID){
			color = triangleIndex * 12345678;
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::MESH_ID){
			color = (mesh.id + 1) * 12345678;
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::TEXTURE && mesh.uvs){
			if(!mesh.texture.huffmanTables){
				color = sampleColor_linear(data, width, height, uv);
			}else{
				// uint32_t mcu = uvToMCUIndex(width, height, uv.x, uv.y);
				uv.x = uv.x - floor(uv.x);
				uv.y = uv.y - floor(uv.y);
				int tx = (int(uv.x * width) % width);
				int ty = (int(uv.y * height) % height);
				uint32_t mcu_x = tx / 16;
				uint32_t mcu_y = ty / 16;
				uint32_t mcus_x = width / 16;
				uint32_t mcu = mcu_x + mcus_x * mcu_y;
				uint32_t key = pack_mcuidx_textureidx_miplevel(mcu, mesh.texture.handle, uint32_t(mipLevel));

				uint32_t texID = mesh.texture.handle;

				// color = sampleJpeg_nearest(texID, uv, textures, jpp.decoded, jpp.decodedMcuMap, mipLevel);
				color = sampleJpeg_linear(texID, uv, textures, jpp.decoded, jpp.decodedMcuMap, mipLevel);

				// if(x == mouseX && y == mouseY){
				// 	state->dbg_hovered_textureHandle = mesh.texture.handle;
				// 	state->dbg_hovered_mipLevel      = mipLevel;
				// 	state->dbg_hovered_tx            = tx;
				// 	state->dbg_hovered_ty            = ty;
				// 	state->dbg_hovered_mcu_x         = mcu_x;
				// 	state->dbg_hovered_mcu_y         = mcu_y;
				// 	state->dbg_hovered_mcu           = mcu;
				// 	state->dbg_hovered_decoded_color = color;
				// }
			}

		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::UV && mesh.uvs){
			rgb[0] = fmodf(uv.x - floor(uv.x), 1.0f) * 256.0f;
			rgb[1] = fmodf(uv.y - floor(uv.y), 1.0f) * 256.0f;
			rgb[2] = 0;
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::NORMAL && mesh.normals){
			
			vec3 n0 = mesh.normals[i0];
			vec3 n1 = mesh.normals[i1];
			vec3 n2 = mesh.normals[i2];

			n0 = mesh.world * vec4(n0, 0.0f);
			n1 = mesh.world * vec4(n1, 0.0f);
			n2 = mesh.world * vec4(n2, 0.0f);

			vec3 N = n0 * (1.0f - stv.x - stv.y) + n1 * stv.x + n2 * stv.y;
			N = normalize(N);

			rgb[0] = clamp(N.x * 255.0f, 0.0f, 255.0f);
			rgb[1] = clamp(N.y * 255.0f, 0.0f, 255.0f);
			rgb[2] = clamp(N.z * 255.0f, 0.0f, 255.0f);
			
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::NORMAL && !mesh.normals){
			vec3 edge1 = b_world - a_world;
			vec3 edge2 = c_world - a_world;
			vec3 N = normalize(cross(edge1, edge2));

			rgb[0] = clamp(N.x * 255.0f, 0.0f, 255.0f);
			rgb[1] = clamp(N.y * 255.0f, 0.0f, 255.0f);
			rgb[2] = clamp(N.z * 255.0f, 0.0f, 255.0f);
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::VERTEX_COLORS && mesh.colors){
			uint32_t C_a = mesh.colors[i0];
			uint32_t C_b = mesh.colors[i1];
			uint32_t C_c = mesh.colors[i2];
			color = C_a;

			auto toVec3 = [](uint32_t C){
				return vec3{
					(C >>  0) & 0xff,
					(C >>  8) & 0xff,
					(C >> 16) & 0xff,
				};
			};

			vec3 c_a = toVec3(C_a);
			vec3 c_b = toVec3(C_b);
			vec3 c_c = toVec3(C_c);

			vec3 c = c_a * (1.0f - stv.x - stv.y) + c_b * stv.x + c_c * stv.y;
			rgb[0] = clamp(c.x, 0.0f, 255.0f);
			rgb[1] = clamp(c.y, 0.0f, 255.0f);
			rgb[2] = clamp(c.z, 0.0f, 255.0f);
		}else if(rasterizationSettings.displayAttribute == DisplayAttribute::STAGE){
			// We don't actually know which stage a pixel was rasterized by, 
			// but we can assume based on the triangle's size and whether it intersects the near plane.
			float f = c_target.proj[1][1];
			float aspect = float(c_target.width) / float(c_target.height);

			vec3 a_ndc = viewToNDC(a_view, f, aspect);
			vec3 b_ndc = viewToNDC(b_view, f, aspect);
			vec3 c_ndc = viewToNDC(c_view, f, aspect);

			bool isNontrivial = (a_ndc.z <= 0.0f || b_ndc.z <= 0.0f || c_ndc.z <= 0.0f);

			vec2 a_screen = ndcToScreen(a_ndc, c_target.width, c_target.height);
			vec2 b_screen = ndcToScreen(b_ndc, c_target.width, c_target.height);
			vec2 c_screen = ndcToScreen(c_ndc, c_target.width, c_target.height);

			// screen-space bounding box of triangle
			float min_x = min(a_screen.x, min(b_screen.x, c_screen.x));
			float max_x = max(a_screen.x, max(b_screen.x, c_screen.x));
			float min_y = min(a_screen.y, min(b_screen.y, c_screen.y));
			float max_y = max(a_screen.y, max(b_screen.y, c_screen.y));

			// clip to screen
			min_x = max(min_x, 0.0f);
			max_x = min(max_x, float(c_target.width - 1));
			min_y = max(min_y, 0.0f);
			max_y = min(max_y, float(c_target.height - 1));

			int size_x = ceil(max_x) - floor(min_x);
			int size_y = ceil(max_y) - floor(min_y);
			int numFragments = size_x * size_y;

			int stage = 0;

			if(numFragments <= THRESHOLD_SMALL)      stage = 0;
			else if(numFragments <= THRESHOLD_LARGE) stage = 1;
			else                                     stage = 2;
			
			if(isNontrivial) stage = 2;

			color = 0xffffaaff;
			if(stage == 0) color = 0xffbd8832;
			if(stage == 1) color = 0xffa4ddab;
			if(stage == 2) color = 0xff61aefd;
		}else{
			color = 0xff0000ff;
		}
		color = color | 0xff000000;

		if(rasterizationSettings.enableDiffuseLighting && mesh.normals){

			vec3 n0 = mesh.normals[i0];
			vec3 n1 = mesh.normals[i1];
			vec3 n2 = mesh.normals[i2];

			n0 = mesh.world * vec4(n0, 0.0f);
			n1 = mesh.world * vec4(n1, 0.0f);
			n2 = mesh.world * vec4(n2, 0.0f);

			vec3 N = normalize(n0 * (1.0f - stv.x - stv.y) + n1 * stv.x + n2 * stv.y);

			// Same image-based lighting the sphere path uses, so meshes and atoms
			// sit in the same environment instead of under two different lights.
			vec3 p_world = a_world * (1.0f - stv.x - stv.y) + b_world * stv.x + c_world * stv.y;
			vec3 V = normalize(c_target.cameraPos - p_world);

			color = shadeEnvironment(color, N, V);
		}

		// Highlight hovered mesh: draw borders
		if(rasterizationSettings.enableObjectPicking && mesh.id == state->hovered_meshId){

			bool isInside = true;

			for(int dx : {-1, 0, 1})
			for(int dy : {-1, 0, 1})
			{

				int nx = clamp(x + dx, 0, c_target.width - 1);
				int ny = clamp(y + dy, 0, c_target.height - 1);
				int neighbor_pixelID = toFramebufferIndex(nx, ny, c_target.width);
				uint64_t neighbor_pixel = c_target.framebuffer[neighbor_pixelID];

				float neighbor_depth;
				uint32_t neighbor_meshIndex;
				uint64_t neighbor_totalTriangleIndex = 0;
				unpack_pixel(neighbor_pixel, &neighbor_depth, &neighbor_totalTriangleIndex);

				// Check if neighbor's total triangle index is within this pixel's mesh
				if(neighbor_totalTriangleIndex < mesh.cummulativeTriangleCount) isInside = false;
				if(neighbor_totalTriangleIndex >= mesh.cummulativeTriangleCount + mesh.numTriangles) isInside = false;
			}

			if(!isInside){
				color = 0xff0000ff;
			}
		}
		

	}else{
		color = 0;

		if(x == mouseX && y == mouseY){
			state->hovered_meshId = 0xffffffff;
			state->hovered_triangleIndex = 0xffffffff;
		}
	}

	shade_finished:

	// WIREFRAME
	if(rasterizationSettings.showWireframe){
		uint32_t numPixels = c_target.width * c_target.height;
		int pid_00 = clamp(toFramebufferIndex(x + 0, y + 0, c_target.width), 0u, numPixels - 1);
		int pid_01 = clamp(toFramebufferIndex(x + 0, y + 1, c_target.width), 0u, numPixels - 1);
		int pid_10 = clamp(toFramebufferIndex(x + 1, y + 0, c_target.width), 0u, numPixels - 1);

		uint32_t p_00 = c_target.framebuffer[pid_00] & 0xffffffff;
		uint32_t p_01 = c_target.framebuffer[pid_01] & 0xffffffff;
		uint32_t p_10 = c_target.framebuffer[pid_10] & 0xffffffff;

		if(p_00 != p_10) color = 0xffff00ff;
		if(p_00 != p_01) color = 0xffff00ff;

		// for(int dx : {-1, 0, 1})
		// for(int dy : {-1, 0, 1})
		for(int dx : {0, 1})
		for(int dy : {0, 1})
		{
			int pid_neighbor = clamp(toFramebufferIndex(x + dx, y + dy, c_target.width), 0u, numPixels - 1);
			uint32_t p_neighbor = c_target.framebuffer[pid_neighbor] & 0xffffffff;

			if(p_00 != p_neighbor) color = 0xffff00ff;
		}
	}

	// 64x64 tile grid
	// if(x % 64 == 0 || y % 64 == 0){
	// 	color = 0xffff00ff;
	// }

	if(depth != Infinity){
		uint64_t udepth = __float_as_uint(depth);
		uint64_t pixel = (udepth << 32) | color;
		c_target.colorbuffer[pixelID] = pixel;
	}else{
		// c_target.colorbuffer[pixelID] = uint64_t(__float_as_uint(INFINITY)) << 32 | 0xff776655;
		c_target.colorbuffer[pixelID] = uint64_t(__float_as_uint(INFINITY)) << 32 | 0;
	}
}