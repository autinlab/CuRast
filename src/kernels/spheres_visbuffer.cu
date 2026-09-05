#define CUB_DISABLE_BF16_SUPPORT

#define GLM_FORCE_CUDA
#define GLM_FORCE_NO_CTOR_INIT
#define CUDA_VERSION 12000
namespace std {
	using size_t = ::size_t;
};

#include "./glm/glm/glm.hpp"
#include "./glm/glm/gtc/matrix_transform.hpp"
#include "./glm/glm/gtx/transform.hpp"

#include "./HostDeviceInterface.h"

using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat4;

__constant__ RenderTarget c_target;

// One thread per sphere.
// Projects the sphere to screen space, iterates its bounding box, and
// atomicMin-writes into sphere_framebuffer: (packed_depth << 32) | sphereIdx.
// The sphere index fits in 32 bits — 4 billion spheres per draw call.
extern "C" __global__
void kernel_draw_spheres(SphereRasterArgs args) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	// LOD per-launch dispatch: thread tid maps to atom tid * dispatchStride.
	int stride = (args.dispatchStride > 0) ? args.dispatchStride : 1;
	int sphereIdx = tid * stride;
	if(sphereIdx >= (int)args.numSpheres) return;

	vec3 center_world = args.positions[sphereIdx];
	float radius      = args.radii[sphereIdx];

	// Apply only this launch's level (scale + smooth fade).
	if(args.activeLevel >= 0 && args.activeLevel < MAX_SPHERE_LOD_LEVELS){
		const SphereLodLevel& L = args.lod.levels[args.activeLevel];
		float dist = length(c_target.cameraPos - center_world);
		if(dist < L.minDist || dist > L.maxDist) return;

		// Partition: defer to a coarser active level if it would cover this atom.
		// Without this guard, atoms whose index is divisible by multiple active
		// strides get drawn once per active level → "doughnut" artifact.
		int  ss = args.skipStride;
		bool skip = (ss > 0)
			&& ((sphereIdx % ss) == 0)
			&& (dist >= args.skipMinDist)
			&& (dist <= args.skipMaxDist);
		if(skip) return;

		float ov   = (L.overlap > 1e-6f) ? L.overlap : 1.0f;
		float fIn  = __saturatef((dist - L.minDist) / ov);
		float fOut = __saturatef((L.maxDist - dist) / ov);
		fIn  = fIn  * fIn  * (3.0f - 2.0f * fIn);
		fOut = fOut * fOut * (3.0f - 2.0f * fOut);
		float fade = fminf(fIn, fOut);
		if(fade <= 0.0f) return;
		radius *= L.scale * fade;
	}

	// Transform center to view space
	vec4 center_view4 = c_target.view * vec4(center_world, 1.0f);
	vec3 center_view  = vec3(center_view4);

	// Cull if entirely behind near plane
	constexpr float NEAR = 0.1f;
	if(center_view.z + radius >= -NEAR) return;

	float depth_center = -center_view.z;
	float depth_front  = max(depth_center - radius, NEAR);

	// Use projection matrix for screen-space mapping (avoids needing c_target.f / aspect)
	// proj[0][0] = f/aspect (X scale),  proj[1][1] = f (Y scale)
	float px_scale = c_target.proj[0][0];
	float py_scale = c_target.proj[1][1];

	// NDC coordinates of sphere center
	float cx_ndc =  px_scale * center_view.x / depth_center;
	float cy_ndc =  py_scale * center_view.y / depth_center;

	// NDC radius (using max of X/Y scale for a conservative circular bound)
	float r_ndc_x = px_scale * radius / depth_center;
	float r_ndc_y = py_scale * radius / depth_center;

	float halfW = float(c_target.width)  * 0.5f;
	float halfH = float(c_target.height) * 0.5f;

	// NDC [-1,1] → screen pixels.  Note: framebuffer y=0 is the bottom (v=-1).
	float cx_s =  cx_ndc * halfW + halfW;
	float cy_s =  cy_ndc * halfH + halfH;
	float rx_s = r_ndc_x * halfW;
	float ry_s = r_ndc_y * halfH;

	// Sub-pixel atoms (huge molecular assemblies viewed from far away) would fail the
	// ellipse test and render nothing. Guarantee at least a half-pixel radius so the
	// atom always paints its center pixel.
	const float MIN_PIX = 0.5f;
	if(rx_s < MIN_PIX){ rx_s = MIN_PIX; r_ndc_x = MIN_PIX / halfW; }
	if(ry_s < MIN_PIX){ ry_s = MIN_PIX; r_ndc_y = MIN_PIX / halfH; }

	int x0 = max(0, (int)(cx_s - rx_s));
	int x1 = min(c_target.width  - 1, (int)(cx_s + rx_s + 1.0f));
	int y0 = max(0, (int)(cy_s - ry_s));
	int y1 = min(c_target.height - 1, (int)(cy_s + ry_s + 1.0f));

	if(x0 > x1 || y0 > y1) return;

	// Pack: high 32 bits = float depth (front face), low 32 bits = sphereIdx+1
	uint64_t packed_depth = (uint64_t)__float_as_uint(depth_front);
	uint64_t my_val = (packed_depth << 32) | uint64_t(sphereIdx + 1);

	for(int py = y0; py <= y1; py++)
	for(int px = x0; px <= x1; px++)
	{
		// Ellipse test in NDC
		float dx_ndc = ((float(px) + 0.5f - halfW) / halfW) - cx_ndc;
		float dy_ndc = ((float(py) + 0.5f - halfH) / halfH) - cy_ndc;
		if((dx_ndc * dx_ndc) / (r_ndc_x * r_ndc_x) + (dy_ndc * dy_ndc) / (r_ndc_y * r_ndc_y) > 1.0f) continue;

		int pixelID = px + py * c_target.width;
		atomicMin((unsigned long long*)&args.sphere_framebuffer[pixelID], (unsigned long long)my_val);
	}
}

// Clear sphere framebuffer to "empty" (all-ones = very large depth, index=0)
extern "C" __global__
void kernel_clearSphereFramebuffer(uint64_t* framebuffer, uint32_t numPixels) {
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= numPixels) return;
	framebuffer[i] = 0xFFFFFFFFFFFFFFFFull;
}
