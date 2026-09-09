// QuteMol-style baked per-atom ambient occlusion.
//
// QuteMol (Tarini et al. 2006) renders the molecule from ~128 directions and, for each
// atom-surface texel, tests visibility against that view's depth buffer. Here the same
// idea is applied to a single scalar per atom, because a per-texel atlas does not
// survive the scale jump: 8x8 texels for 158.9M atoms is 10.2 GB, while one byte per
// atom is 159 MB. The intra-atom variation that is lost is invisible at any framing
// where you can see a whole cell -- which is exactly the framing that needs the help.
//
// Why bake at all when GTAO exists: at whole-cell framing ~90 atoms fall in each pixel,
// so the depth buffer is atom noise rather than a surface, and a screen-space horizon
// search returns near-uniform occlusion. Object-space baking does not care how many
// atoms land in a pixel. The two are complementary and multiply -- baked supplies the
// large-scale enclosure, GTAO the contact detail.
//
// Method per direction d:
//   1. splat every atom's depth along d into an orthographic depth buffer (atomicMax,
//      so the winner is the atom closest to a viewer sitting at +infinity along d)
//   2. an atom is "open" along d if nothing at its pixel is more than `tolerance`
//      closer to that viewer
// Accumulate over directions, normalise, store as uint8.
//
// The splat is one pixel per atom rather than a rasterised disc: at these resolutions a
// pixel is comparable to an atom, so a centre sample is a fair visibility test at a
// fraction of the cost of a full rasterisation.

#define GLM_FORCE_CUDA
#define GLM_FORCE_NO_CTOR_INIT
#define CUDA_VERSION 12000
namespace std {
	using size_t = ::size_t;
};

#include "./glm/glm/glm.hpp"
#include "./HostDeviceInterface.h"

using glm::vec3;

// Depth is stored as float bits reinterpreted to uint32 so the atomic works directly.
// Depths are biased to be strictly positive, and for positive floats the IEEE bit
// pattern is monotonic, so an integer max is a float max.
#define AO_EMPTY_DEPTH 0u

extern "C" __global__
void kernel_atomAO_clear(uint32_t* depthbuffer, int numPixels){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= numPixels) return;
	depthbuffer[i] = AO_EMPTY_DEPTH;
}

// Project an atom onto the sweep plane. Returns false when it falls outside the window.
__device__ inline bool aoProject(vec3 p, const AoSweep& s, int* outPixel, float* outDepth){
	vec3 rel = p - s.origin;
	float u = dot(rel, s.right);
	float v = dot(rel, s.up);
	float d = dot(rel, s.dir);

	float fx = (u / s.halfExtent * 0.5f + 0.5f) * float(s.res);
	float fy = (v / s.halfExtent * 0.5f + 0.5f) * float(s.res);

	int px = int(fx);
	int py = int(fy);
	if(px < 0 || py < 0 || px >= s.res || py >= s.res) return false;

	*outPixel = py * s.res + px;
	*outDepth = d + s.depthBias;   // biased strictly positive
	return true;
}

extern "C" __global__
void kernel_atomAO_splat(
	vec3* positions, uint32_t numSpheres,
	uint32_t* depthbuffer, AoSweep* sweep
){
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= numSpheres) return;

	int pixel; float d;
	if(!aoProject(positions[i], *sweep, &pixel, &d)) return;
	if(d < 0.0f) d = 0.0f;

	atomicMax(&depthbuffer[pixel], __float_as_uint(d));
}

// Accumulate one direction's visibility into a per-atom counter.
//
// `tolerance` is world-space slack. Without it only the single closest atom in each
// pixel would ever count as open and a packed structure would come out uniformly
// black; a pixel here spans a couple of angstroms, so several genuinely exposed
// surface atoms can share one.
extern "C" __global__
void kernel_atomAO_gather(
	vec3* positions, uint32_t numSpheres,
	uint32_t* depthbuffer, AoSweep* sweep,
	uint32_t* visibleCounts
){
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= numSpheres) return;

	int pixel; float d;
	if(!aoProject(positions[i], *sweep, &pixel, &d)){
		// Outside the sweep window: open, rather than picking up occlusion that is not
		// there. Should not happen since the window covers the bounding sphere.
		visibleCounts[i] += 1u;
		return;
	}

	uint32_t raw = depthbuffer[pixel];
	if(raw == AO_EMPTY_DEPTH) { visibleCounts[i] += 1u; return; }

	float nearest = __uint_as_float(raw);   // largest d at this pixel == closest to the viewer
	if(nearest - d <= sweep->tolerance) visibleCounts[i] += 1u;
}

// Normalise the accumulated counts into the uint8 buffer the renderer samples.
extern "C" __global__
void kernel_atomAO_normalize(
	uint32_t* visibleCounts, uint8_t* ao, uint32_t numSpheres,
	int numDirections, float intensity
){
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= numSpheres) return;

	int dirs = numDirections > 1 ? numDirections : 1;
	float open = float(visibleCounts[i]) / float(dirs);

	// Normalise against the best a real surface can do. A point on a flat, unobstructed
	// plane sees exactly half the sphere, so raw openness tops out near 0.5 and a fully
	// exposed atom would otherwise come out mid-grey and darken the whole model. This
	// maps "as exposed as a flat surface can be" to 1.0, which is what the shading
	// expects to multiply by.
	open = open * 2.0f;

	open = powf(fminf(fmaxf(open, 0.0f), 1.0f), intensity);

	ao[i] = (uint8_t)(open * 255.0f + 0.5f);
}
