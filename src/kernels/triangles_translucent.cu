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
#include <cooperative_groups/memcpy_async.h>

#include "./libs/glm/glm/glm.hpp"
#include "./libs/glm/glm/gtc/matrix_transform.hpp"
#include "./libs/glm/glm/gtc/matrix_access.hpp"
#include "./libs/glm/glm/gtx/transform.hpp"
#include "./libs/glm/glm/gtc/quaternion.hpp"

#include "./../types.h"

#include "./utils.cuh"
#include "./HostDeviceInterface.h"
#include "../BitEdit.h"
#include "./rasterization_helpers.cuh"

using glm::ivec2;
using glm::i8vec4;
using glm::vec4;

vec4 getVertex(const CMesh& mesh, u32 vertexIndex){

	u32 resolvedIndex;
	if(mesh.indices){
		resolvedIndex = mesh.indices[vertexIndex];
	}else{
		resolvedIndex = vertexIndex;
	}

	vec4 pos = vec4(mesh.positions[resolvedIndex], 1.0f);
	
	return pos;
}

void binning(
	const RasterArgs args,
	const CMesh& sh_mesh,
	int meshIndex,
	const mat4& worldView,
	int triangleIndex,
	int instanceIndex,
	vec4 a_object,
	vec4 b_object,
	vec4 c_object,
	TranslucentTriangle* queueTriangles,
	uint64_t* queueKeyValues,
	u32* queueSize
){
	
	if(triangleIndex >= sh_mesh.numTriangles) return;
	
	// if(meshIndex != 0) return;
	// if(triangleIndex != 0) return;
	
	float f = args.target.proj[1][1];
	float aspect = float(args.target.width) / float(args.target.height);

	vec3 a_view = worldView * a_object;
	vec3 b_view = worldView * b_object;
	vec3 c_view = worldView * c_object;
	vec3 a_ndc = viewToNDC(a_view, f, aspect);
	vec3 b_ndc = viewToNDC(b_view, f, aspect);
	vec3 c_ndc = viewToNDC(c_view, f, aspect);
	
	// vec2 a_screen = ndcToScreen(a_ndc, args.target.width, args.target.height);
	// vec2 b_screen = ndcToScreen(b_ndc, args.target.width, args.target.height);
	// vec2 c_screen = ndcToScreen(c_ndc, args.target.width, args.target.height);
	
	float min_x, min_y, max_x, max_y;
	computeScreenSpaceBoundingBox(
		a_ndc, b_ndc, c_ndc, a_view, b_view, c_view,
		worldView, f, aspect, args.target.width, args.target.height,
		&min_x, &max_x, &min_y, &max_y
	);
	
	if(a_ndc.z < 0.0f && b_ndc.z < 0.0f && c_ndc.z < 0.0f) return;
	// if(a_ndc.z < 0.0f || b_ndc.z < 0.0f || c_ndc.z < 0.0f) return;
	if(min_x >= max_x) return;
	if(min_y >= max_y) return;
	
	auto drawLine = [&](vec2 start, vec2 end){
		for(float i = 0; i <= 200; i++){
			float u = float(i) / 200.0f;
			
			vec2 pos = (1.0f - u) * start + u * end;
			
			int px = pos.x;
			int py = pos.y;
			
			if(px < 0 || px >= args.target.width) continue;
			if(py < 0 || py >= args.target.height) continue;
			
			int pixelID = px + args.target.width * py;
			
			u64 pixel = 0x00000000'ff0000ff;
			atomicMin(&args.target.colorbuffer[pixelID], pixel);
			
			
		}
	};
	
	// drawLine({min_x, min_y}, {max_x, min_y});
	// drawLine({min_x, max_y}, {max_x, max_y});
	// drawLine({min_x, min_y}, {min_x, max_y});
	// drawLine({max_x, min_y}, {max_x, max_y});
	
	// printf("%.3f, %.3f \n", min_x, max_x);
	
	constexpr int TILE_SIZE = 16;
	
	int size_x = max_x - min_x;
	int size_y = max_y - min_y;
	int tiles_x = (size_x + TILE_SIZE - 1) / TILE_SIZE;
	int tiles_y = (size_y + TILE_SIZE - 1) / TILE_SIZE;
	int numTiles = tiles_x * tiles_y;
	
	if(numTiles == 0) return;
	
	u32 queueIndex = atomicAdd(queueSize, numTiles);
	
	for(int i = 0; i < numTiles; i++){
		int lx = i % tiles_x;
		int ly = i / tiles_x;
		
		TranslucentTriangle t;
		t.meshIndex = meshIndex;
		t.tile_x = int(min_x) / TILE_SIZE + lx;
		t.tile_y = int(min_y) / TILE_SIZE + ly;
		t.triangleIndex = triangleIndex;
		
		float depth = a_ndc.z;
		uint32_t udepth = __float_as_uint(depth) & 0xffffff00;
		
		// Used to sort queued translucent triangles by tile and depth.
		// attributes: <tile><depth><queueIndex>
		// tile_x: 8 bit
		// tile_y: 8 bit
		// depth: 24 bit
		// triangle part index: 24 bit
		uint64_t keyValue = 
			uint64_t(t.tile_x) << 56 ||
			uint64_t(t.tile_y) << 48 ||
			uint64_t(t.tile_y) << 48 ||
			uint64_t(udepth)   << 24 ||
			uint64_t(queueIndex + i);
			
		queueTriangles[queueIndex + i] = t;
		queueKeyValues[queueIndex + i] = keyValue;
	}
	
	
	
}


extern "C" __global__
void kernel_stage1_binning(
	RasterArgs args,
	TranslucentTriangle* queueTriangles,
	uint64_t* queueKeyValues,
	u32* queueSize
){
	auto grid = cg::this_grid();
	auto block = cg::this_thread_block();
	
	// Initialize gridwide state
	if(grid.thread_rank() == 0){
		*args.numProcessedBatches = 0;
		*args.numProcessedBatches_nontrivial = 0;
		*args.hugeTrianglesCounter = 0;
		*args.nontrivialTrianglesCounter = 0;
		*args.numProcessedHugeTriangles = 0;
		*queueSize = 0;
	}
	

	// Initialize block state
	__shared__ int sh_blockBatchIndex;
	__shared__ int sh_blockLocalBatchIndex;
	__shared__ int sh_meshIndex;
	__shared__ CMesh sh_mesh;

	if (block.thread_rank() == 0){
		sh_blockBatchIndex = 0;
		sh_blockLocalBatchIndex = 0;
		sh_meshIndex = 0;
		sh_mesh = args.meshes[0];
		args.state->dbg_fragcount = 0;
	}

	grid.sync();
	
	

	// LOOP THROUGH TRIANGLES
	while (true){

		// Claim Work: Check which batch of triangles this block should render next.
		block.sync();
		if (block.thread_rank() == 0){

			u32 next = atomicAdd(args.numProcessedBatches, 1);
			u32 diff = next - sh_blockBatchIndex;
			
			sh_blockBatchIndex = next;
			sh_blockLocalBatchIndex += diff;
			
			// Next batch is outside of current mesh.
			// Advance through meshes to the one containing the next batch
			int numBatchesInMesh = (sh_mesh.numTriangles + TRIANGLES_PER_SWEEP - 1) / TRIANGLES_PER_SWEEP;
			while (sh_blockLocalBatchIndex >= numBatchesInMesh){
				sh_meshIndex++;

				if (sh_meshIndex >= args.numMeshes) break;
				
				sh_blockLocalBatchIndex -= numBatchesInMesh;
				sh_mesh = args.meshes[sh_meshIndex];
				numBatchesInMesh = (sh_mesh.numTriangles + TRIANGLES_PER_SWEEP - 1) / TRIANGLES_PER_SWEEP;
			}
		}
		block.sync();

		if (sh_meshIndex >= args.numMeshes) return;
		
		u32 firstTriangleInBatch = sh_blockLocalBatchIndex * TRIANGLES_PER_SWEEP;
		int triangleIndex = firstTriangleInBatch + block.thread_rank();

		if(triangleIndex >= sh_mesh.numTriangles) continue;

		vec4 a_object = getVertex(sh_mesh, 3 * triangleIndex + 0);
		vec4 b_object = getVertex(sh_mesh, 3 * triangleIndex + 1);
		vec4 c_object = getVertex(sh_mesh, 3 * triangleIndex + 2);
		
		for(int instanceIndex = 0; instanceIndex < sh_mesh.instances.count; instanceIndex++){
			mat4 worldView = args.transforms[sh_mesh.instances.offset + instanceIndex];
			binning(
				args, sh_mesh, sh_meshIndex, worldView, triangleIndex, instanceIndex, 
				a_object, b_object, c_object,
				queueTriangles, queueKeyValues, queueSize
			);
		}
	}

}