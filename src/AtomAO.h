#pragma once

// Host driver for the QuteMol-style per-atom AO bake. See src/kernels/atom_ao.cu for
// the method and docs/qutemol-ao.md for why this shape was chosen over a texture atlas.
//
// One-time cost at load; the result is view-independent, so nothing is recomputed while
// orbiting. Storage is one byte per atom (159 MB for the 158.9M-atom mycoplasma cell).

#include <cmath>
#include <vector>
#include <print>

#include "glm/glm.hpp"
#include "CudaModularProgram.h"
#include "MemoryManager.h"
#include "Timer.h"
#include "unsuck.hpp"

namespace atomao {

using namespace std;
using glm::vec3;

struct BakeSettings {
	int   numDirections = 64;
	int   resolution    = 2048;   // sweep buffer is resolution^2 uint32
	float toleranceMul  = 1.5f;   // slack in units of the largest atom radius
	float intensity     = 1.0f;   // exponent on openness; >1 darkens
};

// Fibonacci sphere: near-uniform directions without the clustering a naive
// lat/long sweep gets at the poles. QuteMol uses a comparable stratified set; what
// matters is that no direction band is over-represented, since each direction
// contributes equally to the average.
inline vector<vec3> fibonacciDirections(int n){
	vector<vec3> dirs;
	dirs.reserve(n);
	const double golden = 3.14159265358979323846 * (3.0 - sqrt(5.0));
	for(int i = 0; i < n; i++){
		double z    = 1.0 - 2.0 * (i + 0.5) / double(n);
		double r    = sqrt(max(0.0, 1.0 - z * z));
		double theta = golden * i;
		dirs.push_back(vec3(float(r * cos(theta)), float(r * sin(theta)), float(z)));
	}
	return dirs;
}

// Builds an orthonormal frame around `dir`. The choice of up-vector is arbitrary but
// must not be parallel to dir, hence the branch.
inline void frameFor(vec3 dir, vec3& right, vec3& up){
	vec3 ref = (fabsf(dir.z) < 0.9f) ? vec3(0.0f, 0.0f, 1.0f) : vec3(1.0f, 0.0f, 0.0f);
	right = normalize(cross(ref, dir));
	up    = cross(dir, right);
}

// Returns a device pointer to numSpheres bytes of AO, or 0 on failure.
inline CUdeviceptr bake(
	CUdeviceptr cptr_positions,
	uint32_t numSpheres,
	vec3 aabbMin, vec3 aabbMax,
	float maxRadius,
	const BakeSettings& settings
){
	if(numSpheres == 0) return 0;

	static CudaModularProgram* prog = new CudaModularProgram({
		.modules = {"./src/kernels/atom_ao.cu"},
		.useLTO  = false,
	});

	int res      = max(settings.resolution, 64);
	int numDirs  = max(settings.numDirections, 1);
	uint64_t numPix   = uint64_t(res) * uint64_t(res);

	vec3  center = (aabbMin + aabbMax) * 0.5f;
	vec3  extent = aabbMax - aabbMin;
	// Half-extent of the sweep window. The diagonal covers the structure from every
	// direction, so no atom ever falls outside the window and picks up a spurious
	// "outside the bounds" answer.
	float halfExtent = 0.5f * length(extent) * 1.05f;
	float depthBias  = halfExtent * 2.0f;   // keeps projected depths positive
	float tolerance  = maxRadius * settings.toleranceMul;

	double t_start = now();

	CUdeviceptr cptr_depth  = MemoryManager::alloc(numPix * sizeof(uint32_t), "atomAO_depth");
	CUdeviceptr cptr_counts = MemoryManager::alloc(uint64_t(numSpheres) * sizeof(uint32_t), "atomAO_counts");
	CUdeviceptr cptr_ao     = MemoryManager::alloc(uint64_t(numSpheres), "atomAO");

	// MemoryManager::alloc returns 0 when it refuses. Launching against that is an
	// illegal access that takes the whole app down on the next kernel, which is exactly
	// what happened when the bake was enabled on a heavily replicated structure: the
	// 1.8 GB atomAO request was refused with 468 MB free, and the following launch
	// killed the session. The counters are the big ones -- 4 bytes/atom for counts on
	// top of 1 byte/atom for the result -- so this trips well before the GPU is full.
	if(cptr_depth == 0 || cptr_counts == 0 || cptr_ao == 0){
		println("AtomAO: not enough GPU memory to bake {:L} atoms "
			"(needs ~{} MB counters + {} MB result + {} MB sweep). Skipping.",
			numSpheres,
			(uint64_t(numSpheres) * sizeof(uint32_t)) >> 20,
			uint64_t(numSpheres) >> 20,
			(numPix * sizeof(uint32_t)) >> 20);

		if(cptr_depth)  MemoryManager::free(cptr_depth);
		if(cptr_counts) MemoryManager::free(cptr_counts);
		if(cptr_ao)     MemoryManager::free(cptr_ao);
		return 0;
	}

	cuMemsetD8(cptr_counts, 0, uint64_t(numSpheres) * sizeof(uint32_t));

	vector<vec3> dirs = fibonacciDirections(numDirs);

	// Sweep parameters travel through device memory as one struct; see AoSweep.
	CUdeviceptr cptr_sweep = MemoryManager::alloc(sizeof(AoSweep), "atomAO_sweep");
	if(cptr_sweep == 0){
		println("AtomAO: could not allocate the sweep struct. Skipping.");
		MemoryManager::free(cptr_depth);
		MemoryManager::free(cptr_counts);
		MemoryManager::free(cptr_ao);
		return 0;
	}

	println("AtomAO: baking {} directions at {}x{} for {:L} atoms ({} MB)",
		numDirs, res, res, numSpheres, (uint64_t(numSpheres)) / (1024 * 1024));

	for(int k = 0; k < numDirs; k++){
		AoSweep sweep = {};
		sweep.origin     = center;
		sweep.dir        = dirs[k];
		frameFor(sweep.dir, sweep.right, sweep.up);
		sweep.halfExtent = halfExtent;
		sweep.depthBias  = depthBias;
		sweep.tolerance  = tolerance;
		sweep.res        = res;
		cuMemcpyHtoD(cptr_sweep, &sweep, sizeof(sweep));

		int   numPixInt = (int)numPix;
		void* argsClear[] = { &cptr_depth, &numPixInt };
		prog->launch("kernel_atomAO_clear", argsClear, (int)numPix);

		void* argsSplat[] = { &cptr_positions, &numSpheres, &cptr_depth, &cptr_sweep };
		prog->launch("kernel_atomAO_splat", argsSplat, numSpheres);

		void* argsGather[] = { &cptr_positions, &numSpheres, &cptr_depth, &cptr_sweep, &cptr_counts };
		prog->launch("kernel_atomAO_gather", argsGather, numSpheres);
	}

	float intensity = settings.intensity;
	void* argsNorm[] = { &cptr_counts, &cptr_ao, &numSpheres, &numDirs, &intensity };
	prog->launch("kernel_atomAO_normalize", argsNorm, numSpheres);

	cuCtxSynchronize();

	MemoryManager::free(cptr_sweep);
	MemoryManager::free(cptr_depth);
	MemoryManager::free(cptr_counts);

	println("AtomAO: bake finished in {:.2f} s", now() - t_start);

	// Full-buffer statistics. A coarse sample is not enough here: a packed cell is
	// genuinely occluded almost everywhere, so only a handful of percent of atoms
	// should be bright, and a contiguous sample can miss them entirely.
	{
		// Sampled in chunks spread across the buffer rather than copied whole: at a
		// billion-plus atoms a full copy is gigabytes of host RAM for a diagnostic, and
		// this runs on a machine already holding the structure twice over. Chunks
		// rather than a stride because a packed cell is occluded nearly everywhere, so
		// the few percent of bright atoms have to be sampled from throughout the index
		// space or the statistics look like the bake failed.
		const uint64_t CHUNKS     = 64;
		const uint64_t CHUNK_SIZE = 256 * 1024;
		uint64_t total = min<uint64_t>(numSpheres, CHUNKS * CHUNK_SIZE);
		uint64_t per   = total / CHUNKS;

		vector<uint8_t> all;
		all.reserve(total);
		vector<uint8_t> chunk(per);
		for(uint64_t c = 0; c < CHUNKS && per > 0; c++){
			uint64_t offset = (uint64_t)((double(c) / double(CHUNKS)) * double(numSpheres));
			if(offset + per > numSpheres) offset = numSpheres - per;
			cuMemcpyDtoH(chunk.data(), cptr_ao + offset, per);
			all.insert(all.end(), chunk.begin(), chunk.end());
		}

		uint64_t sum = 0, nonZero = 0, above64 = 0;
		int mx = 0;
		for(uint8_t v : all){
			sum += v;
			if(v > 0)  nonZero++;
			if(v > 64) above64++;
			if(v > mx) mx = v;
		}
		uint64_t sampled = all.empty() ? 1 : all.size();
		println("AtomAO: mean={:.2f}/255 max={} nonzero={:.2f}% above64={:.2f}% (sampled {:L})",
			double(sum) / double(sampled), mx,
			100.0 * double(nonZero) / double(sampled),
			100.0 * double(above64) / double(sampled),
			sampled);
	}

	return cptr_ao;
}

} // namespace atomao
