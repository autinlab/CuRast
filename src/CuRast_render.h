
#include <unordered_set>
#include <execution>

#include "jpeg/JpegTextures.h"

#include "Timer.h"
#include "EnvMap.h"
#include "AtomAO.h"
#include "VKRenderer.h"
#include "TextureManager.h"
#include "types.h"

using namespace std;

CudaVirtualMemory* cvm_framebuffer        = nullptr;
CudaVirtualMemory* cvm_colorbuffer        = nullptr;
CudaVirtualMemory* cvm_sphere_framebuffer = nullptr;
// Per-pixel normal G-buffer. uint32 RGBA8: bytes 0..2 = (n.x, n.y, n.z) mapped from
// [-1,1] to [0,255], byte 3 = 0xFF when valid / 0x00 sentinel when no surface.
// Used by SSAO instead of reconstructing the normal from depth gradients, which
// is unreliable on dense sphere-imposter scenes where every pixel sits near a
// silhouette edge.
CudaVirtualMemory* cvm_normalbuffer       = nullptr;
CudaModularProgram* sphereProg            = nullptr;
bool initialized = false;
JpegTextures* jpegTextures = nullptr;

// Cuda-Vulkan interop
struct MappedTextures{
	vector<shared_ptr<VKTexture>> textures;
	vector<CUsurfObject> surfaces;
};

static unordered_map<int64_t, int64_t> lastImportedVersion;

// implemented in lines.cu 
void launch_drawBoundingBoxes(
	RenderTarget target,
	CMesh* meshes,
	u32 numMeshes,
	u32* numProcessedBatches
);

MappedTextures mapCudaVk(vector<shared_ptr<VKTexture>> textures){
	MappedTextures mappings;
	for(auto& tex : textures){
		if(tex->cudaSurface == 0 || lastImportedVersion[tex->ID] != tex->version){
			tex->importToCuda();
			lastImportedVersion[tex->ID] = tex->version;
		}
		mappings.textures.push_back(tex);
		mappings.surfaces.push_back(tex->cudaSurface);
	}
	return mappings;
}

void unmapCudaVk(MappedTextures& mappings){
	// Ensure CUDA writes are complete before Vulkan blits the image
	cuStreamSynchronize((CUstream)CU_STREAM_DEFAULT);
}

void saveScreenshot(RenderTarget target, View view, CUdeviceptr cptr_ssaoShadebuffer, CudaModularProgram* prog_resolve){

	u64 numPixels = target.width * target.height;
	CUdeviceptr cptr_screenshot = MemoryManager::alloc(numPixels * 4, "screenshot");

	u32 backgroundColor = 0;
	uint8_t* bgRgba = (uint8_t*)&backgroundColor;
	bgRgba[0] = clamp(CuRastSettings::background.x * 256.0f, 0.0f, 255.0f);
	bgRgba[1] = clamp(CuRastSettings::background.y * 256.0f, 0.0f, 255.0f);
	bgRgba[2] = clamp(CuRastSettings::background.z * 256.0f, 0.0f, 255.0f);
	bgRgba[3] = 255;

	void* args[] = {
		&cptr_screenshot,
		&cptr_ssaoShadebuffer,
		&CuRastSettings::enableEDL,
		&CuRastSettings::enableSSAO,
		&view.framebuffer->width,
		&view.framebuffer->height,
		&backgroundColor,
		&cvm_normalbuffer->cptr
	};
	prog_resolve->launch2D("kernel_resolve_colorbuffer_to_screenshot", args, target.width, target.height);

	void* screenshot_host = nullptr;
	cuMemAllocHost(&screenshot_host, 4 * numPixels);
	cuMemcpyDtoH(screenshot_host, cptr_screenshot, 4 * numPixels);

	string path = "";
	if(*CuRastSettings::requestScreenshot == ""){
		for(int i = 0; i <= 10'000'000; i++){
			fs::create_directories("./screenshots");
			path = format("./screenshots/screenshot_{}.png", i);

			if(!fs::exists(path)) break;
		}
	}else{
		path = *CuRastSettings::requestScreenshot;
	}

	int stride_in_bytes = target.width * 4;
	stbi_flip_vertically_on_write(1);
	stbi_write_png(path.c_str(), target.width, target.height, 4, screenshot_host, stride_in_bytes);

	MemoryManager::free(cptr_screenshot);
	cuMemFreeHost(screenshot_host);
}

#include "CuRast_vulkanRender.h"

void drawPoints(Scene* scene, View view, RenderTarget& target){
	
	static CudaModularProgram* prog = new CudaModularProgram({
		.modules = {"./src/kernels/points.cu",}
	});
	
	
	vector<SNCPoints*> nodes;
	scene->forEach<SNCPoints>([&](SNCPoints* node){
		nodes.push_back(node);
	});
	
	u64 totalPoints = 0;
	for(SNCPoints* node : nodes){
		
		mat4 worldView = mat4(view.view * node->transform_global);
		
		void* args[] = {
			&target,
			&node->cptr_positions,
			&node->cptr_colors,
			&node->numPoints,
			&worldView
		};
		prog->launchCooperative("kernel_drawPoints", args, {.blocksize = 256});
		
		totalPoints += node->numPoints;
	}
	
	auto& dvlist = Runtime::debugValueList;
	dvlist.push_back({"num points", format("{:L}", totalPoints)});
	
}

void drawTrianglesVisbuffer(
	Scene* scene, View view, vector<CMesh>& meshes, 
	vector<CMesh>& instances, CUdeviceptr cptr_meshes,
	CUdeviceptr cptr_instances, CUdeviceptr cptr_transforms, CUdeviceptr cptr_triangleCountPrefixsum,
	RenderTarget& target, MappedTextures& mappings
){
	auto editor = CuRast::instance;

	if(meshes.size() == 0) return;

	static CUdeviceptr cptr_numProcessedBatches             = MemoryManager::alloc(4, "cptr_numProcessedBatches");
	static CUdeviceptr cptr_numProcessedBatches_nontrivial  = MemoryManager::alloc(4, "cptr_numProcessedBatches_nontrivial");
	static CUdeviceptr cptr_hugeTriangles                   = MemoryManager::alloc(MAX_HUGE_TRIANGLES * sizeof(HugeTriangle), "cptr_hugeTriangles");
	static CUdeviceptr cptr_hugeTrianglesCounter            = MemoryManager::alloc(4, "cptr_hugeTrianglesCounter");
	static CUdeviceptr cptr_nontrivialCounter               = MemoryManager::alloc(4, "cptr_nontrivialCounter");
	static CUdeviceptr cptr_nontrivialList                  = MemoryManager::alloc(8 * MAX_NONTRIVIAL_TRIANGLES, "cptr_nontrivialList");
	static CUdeviceptr cptr_numProcessedHugeTriangles       = MemoryManager::alloc(4, "cptr_numProcessedHugeTriangles");
	
	static CudaModularProgram* prog = new CudaModularProgram({
		.modules = {"./src/kernels/triangles_visbuffer.cu",},
		.useLTO  = false,
	});

	if(instances.size() == 0) return;

	auto custart = Timer::recordCudaTimestamp();

	bool isCompressed = meshes[0].compressed;
	string strCompressed = isCompressed ? "_compressed" : "_uncompressed";
	string strInstanced = (CuRastSettings::rasterizer == RASTERIZER_VISBUFFER_INSTANCED) ? "_instanced" : "";

	string strKernelStage1 = format("kernel_stage1_drawSmallTriangles_indexbuffer{}{}", strCompressed, strInstanced);
	string strKernelStage2 = format("kernel_stage2_drawMediumTriangles_indexbuffer{}", strCompressed);
	string strKernelStage3 = format("kernel_stage3_drawHugeTriangles_indexbuffer{}", strCompressed);

	RasterArgs args;
	args.meshes                          = (CMesh*)cptr_meshes;
	args.numMeshes                       = meshes.size(); 
	args.instances                       = (CMesh*)cptr_instances;
	args.numInstances                    = instances.size();
	args.transforms                      = (mat4*)cptr_transforms;
	args.numProcessedBatches             = (u32*)cptr_numProcessedBatches;
	args.numProcessedBatches_nontrivial  = (u32*)cptr_numProcessedBatches_nontrivial;
	args.hugeTriangles                   = (HugeTriangle*)cptr_hugeTriangles;
	args.hugeTrianglesCounter            = (u32*)cptr_hugeTrianglesCounter;
	args.numProcessedHugeTriangles       = (u32*)cptr_numProcessedHugeTriangles;
	args.nontrivialTrianglesCounter      = (u32*)cptr_nontrivialCounter;
	args.nontrivialTrianglesList         = (u64*)cptr_nontrivialList;
	args.target                          = target;
	args.state                           = (DeviceState*)CuRast::instance->cptr_state;
	
	prog->launchCooperative(strKernelStage1, vector<void*>{&args}, {.blocksize = TRIANGLES_PER_SWEEP});
	prog->launchCooperative(strKernelStage2, vector<void*>{&args});
	prog->launchCooperative(strKernelStage3, vector<void*>{&args}, {.blocksize = 64});

	auto cuend = Timer::recordCudaTimestamp();
	Timer::recordDuration("<triangles visbuffer pipeline>", custart, cuend);
}

void cubSortUint64Keys(uint64_t* d_keys_in, uint64_t* d_keys_out, int num_items);

void drawTrianglesTranslucent(
	Scene* scene, View view, vector<CMesh>& meshes,
	vector<CMesh>& instances, CUdeviceptr cptr_meshes,
	CUdeviceptr cptr_instances, CUdeviceptr cptr_transforms, 
	CUdeviceptr cptr_triangleCountPrefixsum,
	RenderTarget& target, MappedTextures& mappings
){
	auto editor = CuRast::instance;

	if(meshes.size() == 0) return;

	static CUdeviceptr cptr_numProcessedBatches             = MemoryManager::alloc(4, "cptr_numProcessedBatches");
	static CUdeviceptr cptr_numProcessedBatches_nontrivial  = MemoryManager::alloc(4, "cptr_numProcessedBatches_nontrivial");
	static CUdeviceptr cptr_hugeTriangles                   = MemoryManager::alloc(MAX_HUGE_TRIANGLES * sizeof(HugeTriangle), "cptr_hugeTriangles");
	static CUdeviceptr cptr_hugeTrianglesCounter            = MemoryManager::alloc(4, "cptr_hugeTrianglesCounter");
	static CUdeviceptr cptr_nontrivialCounter               = MemoryManager::alloc(4, "cptr_nontrivialCounter");
	static CUdeviceptr cptr_nontrivialList                  = MemoryManager::alloc(8 * MAX_NONTRIVIAL_TRIANGLES, "cptr_nontrivialList");
	static CUdeviceptr cptr_numProcessedHugeTriangles       = MemoryManager::alloc(4, "cptr_numProcessedHugeTriangles");
	
	static CUdeviceptr cptr_queueTriangles                  = MemoryManager::alloc(MAX_TRANSLUCENT_TRIANGLES * sizeof(TranslucentTriangle), "cptr_queue");
	static CUdeviceptr cptr_queueKeyValue                   = MemoryManager::alloc(MAX_TRANSLUCENT_TRIANGLES * sizeof(uint64_t), "cptr_queueKeyValue");
	static CUdeviceptr cptr_queueKeyValueSorted             = MemoryManager::alloc(MAX_TRANSLUCENT_TRIANGLES * sizeof(uint64_t), "cptr_queueKeyValueSorted");
	static CUdeviceptr cptr_queueSize                       = MemoryManager::alloc(4, "cptr_queueSize");
	
	int maxTiles = 512 * 512; // Each tile is 16x16, so allows 8k x 8k 
	static CUdeviceptr cptr_tileRanges                      = MemoryManager::alloc(sizeof(ivec2) * maxTiles, "cptr_tileRanges");
	
	static CudaModularProgram* prog = new CudaModularProgram({
		.modules = {"./src/kernels/triangles_translucent.cu",}
	});

	if(instances.size() == 0) return;

	RasterArgs args;
	args.meshes                          = (CMesh*)cptr_meshes;
	args.numMeshes                       = meshes.size(); 
	args.instances                       = (CMesh*)cptr_instances;
	args.numInstances                    = instances.size();
	args.transforms                      = (mat4*)cptr_transforms;
	args.numProcessedBatches             = (u32*)cptr_numProcessedBatches;
	args.numProcessedBatches_nontrivial  = (u32*)cptr_numProcessedBatches_nontrivial;
	args.hugeTriangles                   = (HugeTriangle*)cptr_hugeTriangles;
	args.hugeTrianglesCounter            = (u32*)cptr_hugeTrianglesCounter;
	args.numProcessedHugeTriangles       = (u32*)cptr_numProcessedHugeTriangles;
	args.nontrivialTrianglesCounter      = (u32*)cptr_nontrivialCounter;
	args.nontrivialTrianglesList         = (u64*)cptr_nontrivialList;
	args.target                          = target;
	args.state                           = (DeviceState*)CuRast::instance->cptr_state;
	
	// Stage 1: Binning
	// Stage 2: Sorting
	// Stage 3: Compute Tile Ranges
	// Stage 3: Blending
	
	// string strKernelStage1 = format("kernel_binning", strCompressed, strInstanced);
	// string strKernelStage2 = format("kernel_", strCompressed);
	
	cuMemsetD8(cptr_queueSize, 0, 4);
	
	auto custart = Timer::recordCudaTimestamp();
	prog->launchCooperative("kernel_stage1_binning", vector<void*>{&args, &cptr_queueTriangles, &cptr_queueKeyValue, &cptr_queueSize}, {.blocksize = TRIANGLES_PER_SWEEP});
	// prog->launchCooperative(strKernelStage2, vector<void*>{&args});
	
	// Stage 2: Sort - read queue size to host (syncs the stream), then CUB-sort on GPU
	u32 queueSize = 0;
	cuMemcpyDtoH(&queueSize, cptr_queueSize, 4);
	if (queueSize > 0) {
		cubSortUint64Keys(
			reinterpret_cast<uint64_t*>(cptr_queueKeyValue),
			reinterpret_cast<uint64_t*>(cptr_queueKeyValueSorted),
			static_cast<int>(queueSize)
		);
	}
	
	u32 tiles_x = (target.width + 16 - 1) / 16;
	u32 tiles_y = (target.height + 16 - 1) / 16;
	u32 numTiles = tiles_x * tiles_y;
	cuMemsetD8(cptr_tileRanges, 0, sizeof(ivec2) * numTiles);
	
	prog->launchCooperative("kernel_stage3_computeRanges", vector<void*>{
		&args, 
		&cptr_queueTriangles, 
		&cptr_queueKeyValueSorted, 
		&cptr_queueSize,
		&cptr_tileRanges,
		&numTiles,
	});
	
	prog->launch("kernel_stage4_blend", vector<void*>{
		&args, 
		&cptr_queueTriangles, 
		&cptr_queueKeyValueSorted, 
		&cptr_queueSize,
		&cptr_tileRanges,
		&numTiles,
	}, {.gridsize = numTiles, .blocksize = 256});

	auto cuend = Timer::recordCudaTimestamp();
	Timer::recordDuration("<triangles translucent pipeline>", custart, cuend);
	
	auto& dvlist = Runtime::debugValueList;
	dvlist.push_back({"num tiles", format("{:L}", queueSize)});
}

void CuRast::draw(Scene* scene, vector<View> views){

	static vector<View> frustumViews;
	if(!CuRastSettings::freezeFrustum){
		frustumViews = views;
	}

	double t_start = now();

	View view = views[0]; // We discarded support for multiple views for now.
	mat4 viewI = inverse(view.view);
	vec3 cameraPos = vec3(viewI * vec4(0.0f, 0.0f, 0.0f, 1.0f));

	int supersamplingFactor = CuRastSettings::supersamplingFactor;

	RenderTarget target;
	target.framebuffer = (u64*)cvm_framebuffer->cptr;
	target.colorbuffer = (u64*)cvm_colorbuffer->cptr;
	target.width = supersamplingFactor * view.framebuffer->width;
	target.height = supersamplingFactor * view.framebuffer->height;
	target.view = view.view;
	target.viewI = viewI;
	target.proj = view.proj;
	target.cameraPos = cameraPos;
	target.projMode   = CuRastSettings::orthographic ? 1 : 0;
	target.orthoHalfH = (float)VKRenderer::camera->orthoHalfH;

	

	// Since processing thousands of nodes can become expensive on CPU side:
	// - Use a persistent std::vector that keeps the capacity over multiple frames
	// - Collect a list of all mesh nodes
	// - Then update them concurrently
	bool hasJpegCompressedTextures = false; 
	Mesh* hoveredMesh = nullptr;
	static vector<SNTriangles*> nodes;
	nodes.clear();
	scene->forEach<SNTriangles>([&](SNTriangles* node){
		
		// if(!node->texture->isTranslucent) return;
		// if(nodes.size() >= 1) return;
		// if(node->mesh->numTriangles != 400) return;
		
		nodes.push_back(node);
		if(node->texture){
			hasJpegCompressedTextures = hasJpegCompressedTextures || node->texture->huffmanTables != nullptr;
		}
	});

	process_parallel(nodes, [&](SNTriangles* node, int64_t index){
		if(node->id == CuRast::deviceState->hovered_meshId){
			Runtime::hovered_node_name = node->name;
			Runtime::hovered_mesh_name = node->mesh->name;
			hoveredMesh = node->mesh;
		}
		node->update(view);
	});

	u64 numTotalTriangles = 0;
	u64 numTotalNodes = 0;
	u64 numVisibleTriangles = 0;
	u64 numVisibleNodes = 0;
	for(int64_t i = 0; i < nodes.size(); i++){
		SNTriangles* node = nodes[i];

		numTotalTriangles += node->mesh->numTriangles;
		numTotalNodes++;

		if(!node->visible) continue;
		if(!node->mesh->isLoaded) continue;

		nodes[numVisibleNodes] = node;

		numVisibleTriangles += node->mesh->numTriangles;
		numVisibleNodes++;
	}
	nodes.resize(numVisibleNodes);

	// Sort/Group by instance
	sort(std::execution::par, nodes.begin(), nodes.end(), [](SNTriangles* a, SNTriangles* b){

		if(a->mesh->numTriangles == b->mesh->numTriangles){
			return u64(a->mesh) < u64(b->mesh);
		}else{
			return a->mesh->numTriangles > b->mesh->numTriangles;
		}
	});

	if(CuRastSettings::rasterizer == RASTERIZER_VULKAN_INDEXED_DRAW){
		drawVulkan_indexed_draw(scene, nodes, view);
	}else if(CuRastSettings::rasterizer == RASTERIZER_VULKAN_INDEXPULLING_INSTANCED){
		drawVulkan_indexpulling_instanced_forward(scene, nodes, view);
	}else if(CuRastSettings::rasterizer == RASTERIZER_VULKAN_INDEXPULLING_VISBUFFER){
		drawVulkan_indexpulling_visibilitybuffer(scene, views);
	}else{
		VKRenderer::vulkanMeshDrawFn = nullptr; // use CUDA blit path in recordCommandBuffer

		auto toCMesh = [&](SNTriangles* node){

			u32 indexRange = node->mesh->index_max - node->mesh->index_min;
			u64 bitsPerIndex = ceil(log2f(float(indexRange + 1)));

			CMesh mesh;
			mesh.world                    = node->transform_global;
			mesh.positions                = (vec3*)node->mesh->cptr_position;
			mesh.uvs                      = (vec2*)node->mesh->cptr_uv;
			mesh.colors                   = (u32*)node->mesh->cptr_color;
			mesh.normals                  = (vec3*)node->mesh->cptr_normal;
			mesh.indices                  = (u32*)node->mesh->cptr_indices;
			mesh.index_min                = node->mesh->index_min;
			mesh.index_max                = node->mesh->index_max;
			mesh.bitsPerIndex             = bitsPerIndex;
			mesh.numTriangles             = node->mesh->numTriangles;
			mesh.numVertices              = node->mesh->numVertices;
			if (node->texture) {
				mesh.texture                  = *node->texture;
			}
			mesh.aabb                     = node->aabb;
			mesh.compressed               = node->mesh->compressed;
			mesh.compressionFactor        = (node->aabb.max - node->aabb.min) / 65536.0f;
			mesh.isLoaded                 = node->mesh->isLoaded;
			mesh.id                       = node->id;
			mesh.address                  = u64(node->mesh);

			vec3 c0 = node->transform_global[0];
			vec3 c1 = node->transform_global[1];
			vec3 c2 = node->transform_global[2];
			float s = dot(cross(c0, c1), c2);
			mesh.flipTriangles = s < 0.0f;

			return mesh;
		};

		//----------------------------------------------
		// Organize into unique meshes and per-instance data
		//----------------------------------------------
		static vector<CMesh> meshes_unique;
		static vector<CMesh> meshes_allInstances;
		static vector<mat4> transforms;
		static vector<u64> triangleCountPrefixsum;
		int64_t sum = 0;
		
		meshes_unique.resize(nodes.size());
		meshes_allInstances.resize(nodes.size());
		transforms.resize(nodes.size());
		triangleCountPrefixsum.resize(nodes.size());
		
		process_parallel(nodes, [&](SNTriangles* node, int64_t index){
			CMesh cmesh = toCMesh(node);
			meshes_allInstances[index] = cmesh;
			transforms[index] = target.view * cmesh.world;
		});

		CMesh* uniqueMesh = nullptr;
		u64 uniqueMeshCounter = 0;
		for(int i = 0; i < nodes.size(); i++){
			CMesh& cmesh = meshes_allInstances[i];

			cmesh.cummulativeTriangleCount = sum;
			cmesh.instances.offset = i;

			triangleCountPrefixsum[i] = sum;
			sum += cmesh.numTriangles;

			// Encountered a new unique mesh
			if(uniqueMesh == nullptr || uniqueMesh->address != cmesh.address || CuRastSettings::disableInstancing){
				meshes_unique[uniqueMeshCounter] = cmesh;
				uniqueMesh = &meshes_unique[uniqueMeshCounter];
				uniqueMesh->instances.count = 0;
				uniqueMeshCounter++;
			}

			uniqueMesh->instances.count++;
		}
		meshes_unique.resize(uniqueMeshCounter);
		
		// Group into opaque and translucent meshes to render each with the corresponding cuda kernels
		static vector<CMesh> meshes_unique_opaque;
		static vector<CMesh> meshes_unique_translucent;
		meshes_unique_opaque.resize(0);
		meshes_unique_translucent.resize(0);
		
		for(CMesh mesh : meshes_unique){
			bool isTranslucent = mesh.texture.isTranslucent;
			if(!CuRastSettings::enableTranslucency){
				isTranslucent = false;
			}
			
			if(isTranslucent){
				meshes_unique_translucent.push_back(mesh);
			}else{
				meshes_unique_opaque.push_back(mesh);
			}
		}

		// prep virtual memory for lots of nodes
		static CudaVirtualMemory* cvm_meshes                    = MemoryManager::allocVirtualCuda(1'000'000 * sizeof(CMesh), "cvm_meshes");
		static CudaVirtualMemory* cvm_meshes_opaque             = MemoryManager::allocVirtualCuda(1'000'000 * sizeof(CMesh), "cvm_meshes_opaque");
		static CudaVirtualMemory* cvm_meshes_translucent        = MemoryManager::allocVirtualCuda(1'000'000 * sizeof(CMesh), "cvm_meshes_translucent");
		static CudaVirtualMemory* cvm_instances                 = MemoryManager::allocVirtualCuda(1'000'000 * sizeof(CMesh), "cvm_instances");
		static CudaVirtualMemory* cvm_transforms                = MemoryManager::allocVirtualCuda(1'000'000 * sizeof(mat4), "cvm_transforms");
		static CudaVirtualMemory* cvm_triangleCountPrefixsum    = MemoryManager::allocVirtualCuda(1'000'000 * sizeof(u64), "cvm_triangleCountPrefixsum");
		
		// commit physical memory for actual amount of nodes
		cvm_meshes                 ->commit(meshes_unique.size()             * sizeof(CMesh));
		cvm_meshes_opaque          ->commit(meshes_unique_opaque.size()      * sizeof(CMesh));
		cvm_meshes_translucent     ->commit(meshes_unique_translucent.size() * sizeof(CMesh));
		cvm_instances              ->commit(meshes_allInstances.size()       * sizeof(CMesh));
		cvm_transforms             ->commit(transforms.size()                * sizeof(mat4));
		cvm_triangleCountPrefixsum ->commit(triangleCountPrefixsum.size()    * sizeof(u64));

		// submit per-frame geometry metadata to GPU
		cuMemcpyHtoDAsync(cvm_meshes->cptr                 , meshes_unique.data(),             byteSizeOf(meshes_unique), 0);
		cuMemcpyHtoDAsync(cvm_meshes_opaque->cptr          , meshes_unique_opaque.data(),      byteSizeOf(meshes_unique_opaque), 0);
		cuMemcpyHtoDAsync(cvm_meshes_translucent->cptr     , meshes_unique_translucent.data(), byteSizeOf(meshes_unique_translucent), 0);
		cuMemcpyHtoDAsync(cvm_instances->cptr              , meshes_allInstances.data(),       byteSizeOf(meshes_allInstances), 0);
		cuMemcpyHtoDAsync(cvm_transforms->cptr             , transforms.data(),                byteSizeOf(transforms), 0);
		cuMemcpyHtoDAsync(cvm_triangleCountPrefixsum->cptr , triangleCountPrefixsum.data(),    byteSizeOf(triangleCountPrefixsum), 0);

		Runtime::numVisibleNodes = numVisibleNodes;
		Runtime::numVisibleTriangles = numVisibleTriangles;
		Runtime::numNodes = numTotalNodes;
		Runtime::numTriangles = numTotalTriangles;

		auto& dvlist = Runtime::debugValueList;
		dvlist.push_back({"#total nodes           ", format("{:40L}", u64(numTotalNodes))});
		dvlist.push_back({"#total triangles       ", format("{:40L}", u64(numTotalTriangles))});
		dvlist.push_back({"#visible nodes         ", format("{:40L}", u64(numVisibleNodes))});
		dvlist.push_back({"#visible triangles     ", format("{:40L}", u64(numVisibleTriangles))});
		dvlist.push_back({"hovered mesh id        ", format("{:40L}", CuRast::deviceState->hovered_meshId)});
		dvlist.push_back({"hovered triangle index ", format("{:40L}", CuRast::deviceState->hovered_triangleIndex)});
		dvlist.push_back({"hovered node name      ", format("{:}", Runtime::hovered_node_name)});
		dvlist.push_back({"hovered mesh name      ", format("{:}", Runtime::hovered_mesh_name)});
		dvlist.push_back({"tris in hovered mesh   ", format("{:40L}", hoveredMesh ? hoveredMesh->numTriangles : 0)});
		dvlist.push_back({"verts in hovered mesh  ", format("{:40L}", hoveredMesh ? hoveredMesh->numVertices : 0)});
		dvlist.push_back({"CPU draw() duration    ", format("{:40.1f} ms", Runtime::duration_draw * 1000.0)});

		// We measure CPU draw time until here, where CPU has finished its stuff and now just invokes cuda kernels.
		Runtime::duration_draw = now() - t_start;

		int numPixels = target.width * target.height;

		vector<shared_ptr<VKTexture>> attachments = {view.framebuffer->colorAttachment};
		auto mappings = mapCudaVk(attachments);

		static CudaModularProgram* prog = new CudaModularProgram(CudaModularProgram::CudaModularProgramArgs{
			.modules = {"./src/kernels/resolve.cu",},
			.useLTO  = false,
		});
		// memcpy arguments to constant buffer
		CUdeviceptr cptr_target = prog->getGlobalsPointer("c_target");
		cuMemcpyHtoDAsync(cptr_target, &target, sizeof(target), 0);

		// Environment lighting. Loading a map and projecting it to SH is host work on a
		// multi-megapixel image, so it runs only when the path actually changes, not
		// per frame. c_env.enabled == 0 leaves the device on its baked-in coefficients.
		{
			static EnvLighting env = {};
			static string loadedPath = "\x01"; // sentinel: differs from any real path, including ""
			static CUdeviceptr cptr_envPixels = 0;
			static bool haveMap = false;

			if(CuRastSettings::envMapReload || loadedPath != CuRastSettings::envMapPath){
				CuRastSettings::envMapReload = false;
				loadedPath = CuRastSettings::envMapPath;

				if(cptr_envPixels != 0){
					MemoryManager::free(cptr_envPixels);
					cptr_envPixels = 0;
				}

				if(loadedPath.empty()){
					env = {};
					env.exposure = CuRastSettings::envExposure;
					env.enabled  = 0;
					haveMap      = false;
					println("EnvMap: using built-in studio coefficients");
				}else{
					envmap::Image img = envmap::load(loadedPath);
					env = envmap::project(img, CuRastSettings::envExposure);
					haveMap = (env.enabled != 0);

					if(img.ok()){
						println("EnvMap: '{}' {}x{}", loadedPath, img.width, img.height);
						println("EnvMap: SH Y00 = ({:.3f}, {:.3f}, {:.3f})  key dir = ({:.3f}, {:.3f}, {:.3f})",
							env.sh[0].x, env.sh[0].y, env.sh[0].z,
							env.keyDir.x, env.keyDir.y, env.keyDir.z);

						// Keep the pixels around for the background pass. RGBA32F rather than
						// a texture object: a manual bilinear fetch is a few lines and avoids
						// plumbing CUDA array + texture object lifetimes through this path.
						u64 numTexels = u64(img.width) * u64(img.height);
						vector<vec4> rgba(numTexels);
						for(u64 i = 0; i < numTexels; i++){
							rgba[i] = vec4(img.rgb[i*3+0], img.rgb[i*3+1], img.rgb[i*3+2], 1.0f);
						}

						cptr_envPixels = MemoryManager::alloc(numTexels * sizeof(vec4), "envmap");
						cuMemcpyHtoD(cptr_envPixels, rgba.data(), numTexels * sizeof(vec4));

						env.texWidth  = img.width;
						env.texHeight = img.height;
						println("EnvMap: background buffer {} MB", (numTexels * sizeof(vec4)) / (1024*1024));
					}
				}
			}

			// Live sliders / toggles: keep in sync without reloading the map.
			env.exposure       = CuRastSettings::envExposure;
			env.pixels         = (vec4*)cptr_envPixels;
			env.showBackground = CuRastSettings::envShowBackground ? 1 : 0;
			// The master toggle only gates use of the loaded map; it never discards it,
			// so flipping it back on costs nothing. haveMap is set from the projection
			// result, so a path that failed to load stays disabled rather than being
			// resurrected by the toggle.
			env.enabled = (CuRastSettings::envEnabled && haveMap) ? 1 : 0;

			CUdeviceptr cptr_env = prog->getGlobalsPointer("c_env");
			if(cptr_env != 0) cuMemcpyHtoDAsync(cptr_env, &env, sizeof(env), 0);

			AoParams ao = {};
			ao.floorValue = CuRastSettings::aoFloor;
			ao.power      = CuRastSettings::aoPower;
			CUdeviceptr cptr_ao = prog->getGlobalsPointer("c_ao");
			if(cptr_ao != 0) cuMemcpyHtoDAsync(cptr_ao, &ao, sizeof(ao), 0);

			HaloParams halo = {};
			halo.enabled   = CuRastSettings::haloEnabled ? 1 : 0;
			halo.size      = CuRastSettings::haloSize;
			halo.strength  = CuRastSettings::haloStrength;
			halo.color     = CuRastSettings::haloColor;
			halo.depthFull = CuRastSettings::haloDepthFull;
			halo.dirs      = CuRastSettings::haloDirs;
			halo.steps     = CuRastSettings::haloSteps;
			CUdeviceptr cptr_halo = prog->getGlobalsPointer("c_halo");
			if(cptr_halo != 0) cuMemcpyHtoDAsync(cptr_halo, &halo, sizeof(halo), 0);

			ShadeParams shade = {};
			shade.flatSpheres = CuRastSettings::flatSpheres ? 1 : 0;
			shade.debugView   = CuRastSettings::debugView;
			shade.envRotation = CuRastSettings::envRotation * 3.14159265f / 180.0f;
			shade.envBgWiden  = CuRastSettings::envBgWiden;
			CUdeviceptr cptr_shade = prog->getGlobalsPointer("c_shade");
			if(cptr_shade != 0) cuMemcpyHtoDAsync(cptr_shade, &shade, sizeof(shade), 0);
		}

		// Let the first kernel in the frame be a dummy kernel to take the hit for CUDA-OpenGL interop overhead
		// (so that we get more accurate timings for the other kernels)
		static CUdeviceptr dummydata = MemoryManager::alloc(16, "dummydata");
		prog->launch("kernel_dummy", {&dummydata}, 1);
		
		{ // resize and clear cuda framebuffer
			u32 clearColor = 0xff000000;
			float clearDepth = Infinity;

			u64 requiredBytes = numPixels * 8;
			cvm_framebuffer->commit(requiredBytes);
			cvm_colorbuffer->commit(requiredBytes);
			cvm_sphere_framebuffer->commit(requiredBytes);
			cvm_normalbuffer->commit((size_t)numPixels * 4);
			// Sentinel-clear: 0 in alpha byte = "no normal stored at this pixel" so SSAO
			// can fall back to depth-based reconstruction for triangle-only pixels.
			cuMemsetD8Async(cvm_normalbuffer->cptr, 0x00, (size_t)numPixels * 4, 0);

			prog->launch("kernel_clearFramebuffer", {
				&cvm_framebuffer->cptr,
				&numPixels,
				&clearColor,
				&clearDepth
			}, numPixels);

			prog->launch("kernel_clearFramebuffer", {
				&cvm_colorbuffer->cptr,
				&numPixels,
				&clearColor,
				&clearDepth
			}, numPixels);

			// Fill sphere framebuffer with 0xFF = sentinel (0xFFFFFFFFFFFFFFFF = no sphere)
			cuMemsetD8Async(cvm_sphere_framebuffer->cptr, 0xFF, (size_t)numPixels * 8, 0);
		}

		drawTrianglesVisbuffer(
			scene, view, meshes_unique_opaque, meshes_allInstances, 
			cvm_meshes_opaque->cptr, 
			cvm_instances->cptr, cvm_transforms->cptr, cvm_triangleCountPrefixsum->cptr,
			target, mappings
		);

		// Per-frame LOD config scaled to the current scene size. The default config
		// in CuRastSettings stores normalised min/max/overlap; here we multiply by
		// `sceneScale` (the orbit radius — a good proxy for the molecule's extent)
		// so the bands track the camera's working distance regardless of structure
		// size. Built once per frame and reused by both the rasterizer and the
		// resolve sphere composite (so ray-sphere shading uses the same scaled radius).
		SphereLodConfig frameLod{};
		if(CuRastSettings::enableSphereLOD){
			const SphereLodConfig& base = CuRastSettings::sphereLodConfig;
			float sceneScale = (float)Runtime::controls->radius;
			if(sceneScale <= 0.0f) sceneScale = 1.0f;
			frameLod.numLevels = base.numLevels;
			for(int k = 0; k < MAX_SPHERE_LOD_LEVELS; k++){
				frameLod.levels[k] = base.levels[k];
				frameLod.levels[k].minDist *= sceneScale;
				frameLod.levels[k].maxDist *= sceneScale;
				frameLod.levels[k].overlap *= sceneScale;
			}
		}

		{ // DRAW SPHERES (sphere imposter visbuffer)
			// Upload c_target constant to the sphere program
			CUdeviceptr cptr_sphere_target = sphereProg->getGlobalsPointer("c_target");
			cuMemcpyHtoDAsync(cptr_sphere_target, &target, sizeof(target), 0);

			vector<SNSpheres*> sphereNodes;
			scene->forEach<SNSpheres>([&](SNSpheres* node){ sphereNodes.push_back(node); });

			// Determine which LOD levels are active for the current camera distance.
			// We dispatch one kernel call per active level with thread count =
			// numSpheres / stride, so distant frames touch ~stride× less memory.
			float cameraDist = (float)Runtime::controls->radius;
			int   activeLevels[MAX_SPHERE_LOD_LEVELS];
			int   numActive = 0;
			if(frameLod.numLevels > 0){
				for(int k = 0; k < frameLod.numLevels; k++){
					const SphereLodLevel& L = frameLod.levels[k];
					if(cameraDist >= L.minDist && cameraDist <= L.maxDist){
						activeLevels[numActive++] = k;
					}
				}
				// Safety: if camera is outside every band (e.g. very close zoom or extreme far),
				// fall back to the band whose centre is nearest the camera so we still draw
				// *something*.
				if(numActive == 0){
					int   bestK   = 0;
					float bestDsq = 1e30f;
					for(int k = 0; k < frameLod.numLevels; k++){
						float c = 0.5f * (frameLod.levels[k].minDist + frameLod.levels[k].maxDist);
						float d = (cameraDist - c); d *= d;
						if(d < bestDsq){ bestDsq = d; bestK = k; }
					}
					activeLevels[numActive++] = bestK;
				}
			}

			for(auto sn : sphereNodes) {
				if(sn->numSpheres == 0) continue;

				if(!CuRastSettings::enableSphereLOD || frameLod.numLevels == 0){
					SphereRasterArgs sArgs;
					sArgs.positions         = (vec3*)sn->cptr_positions;
					sArgs.radii             = (float*)sn->cptr_radii;
					sArgs.colors            = (uint32_t*)sn->cptr_colors;
					sArgs.atomTypes         = (uint8_t*)sn->cptr_atomTypes;
					sArgs.colorPalette      = (uint32_t*)sn->cptr_palette;
					sArgs.numSpheres        = sn->numSpheres;
					sArgs.sphere_framebuffer = (uint64_t*)cvm_sphere_framebuffer->cptr;
					sArgs.lod               = {};
					sArgs.activeLevel       = -1;
					sArgs.dispatchStride    = 1;
					sArgs.skipStride        = 0;
					sArgs.skipMinDist       = 0.0f;
					sArgs.skipMaxDist       = 0.0f;
					sphereProg->launch("kernel_draw_spheres", {&sArgs}, sn->numSpheres);
					continue;
				}

				// activeLevels is in level-index order (level 0 finest first). For each
				// active level we identify the *next coarser active level* and pass its
				// stride + distance band so the kernel can skip atoms that the coarser
				// dispatch will already render. Without this each atom is drawn once
				// per active level → doughnut artifact at level boundaries.
				for(int idx = 0; idx < numActive; idx++){
					int k = activeLevels[idx];
					int stride = frameLod.levels[k].stride;
					if(stride < 1) stride = 1;

					int   skipStride  = 0;
					float skipMinDist = 0.0f;
					float skipMaxDist = 0.0f;
					if(idx + 1 < numActive){
						int kNextCoarser = activeLevels[idx + 1];
						skipStride  = frameLod.levels[kNextCoarser].stride;
						skipMinDist = frameLod.levels[kNextCoarser].minDist;
						skipMaxDist = frameLod.levels[kNextCoarser].maxDist;
					}

					SphereRasterArgs sArgs;
					sArgs.positions         = (vec3*)sn->cptr_positions;
					sArgs.radii             = (float*)sn->cptr_radii;
					sArgs.colors            = (uint32_t*)sn->cptr_colors;
					sArgs.atomTypes         = (uint8_t*)sn->cptr_atomTypes;
					sArgs.colorPalette      = (uint32_t*)sn->cptr_palette;
					sArgs.numSpheres        = sn->numSpheres;
					sArgs.sphere_framebuffer = (uint64_t*)cvm_sphere_framebuffer->cptr;
					sArgs.lod               = frameLod;
					sArgs.activeLevel       = k;
					sArgs.dispatchStride    = stride;
					sArgs.skipStride        = skipStride;
					sArgs.skipMinDist       = skipMinDist;
					sArgs.skipMaxDist       = skipMaxDist;

					int dispatchN = (sn->numSpheres + stride - 1) / stride;
					sphereProg->launch("kernel_draw_spheres", {&sArgs}, dispatchN);
				}
			}
		}

		// DRAW BOUNDING BOXES
		if(CuRastSettings::showBoundingBoxes){
			RenderTarget target_lines = target;
			target_lines.framebuffer = (u64*)cvm_colorbuffer->cptr;

			vector<CMesh> boundingBoxNodes;
			scene->forEach<SNTriangles>([&](SNTriangles* node){
				CMesh mesh;
				mesh.world                    = node->transform_global;
				mesh.aabb                     = node->aabb;

				boundingBoxNodes.push_back(mesh);
			});
			
			static CUdeviceptr cptr_numProcessedBatches = MemoryManager::alloc(4, "cptr_numProcessedBatches");
			// static CUdeviceptr cptr_meshes_boxes = MemoryManager::alloc(40'000 * sizeof(CMesh), "cptr_meshes_boxes");
			static CudaVirtualMemory* cvm_meshes_boxes = MemoryManager::allocVirtualCuda(40'000 * sizeof(CMesh), "boxes");
			cvm_meshes_boxes->commit(boundingBoxNodes.size() * sizeof(CMesh));

			cuMemcpyHtoDAsync(cvm_meshes_boxes->cptr, boundingBoxNodes.data(), boundingBoxNodes.size() * sizeof(CMesh), 0);
			cuMemsetD8Async(cptr_numProcessedBatches, 0, 4, 0);
			
			u32 numMeshes = boundingBoxNodes.size();
			launch_drawBoundingBoxes(
				target_lines, 
				(CMesh*)cvm_meshes_boxes->cptr,
				numMeshes,
				(u32*)cptr_numProcessedBatches
			);
		}

		int mouse_X = Runtime::mousePosition.x;
		int mouse_Y = target.height - Runtime::mousePosition.y;
		u32 numInstances = meshes_allInstances.size();

		RasterizationSettings rasterSettings;
		rasterSettings.showWireframe = CuRastSettings::showWireframe;
		rasterSettings.enableDiffuseLighting = CuRastSettings::enableDiffuseLighting;
		rasterSettings.displayAttribute = CuRastSettings::displayAttribute;
		rasterSettings.enableObjectPicking = CuRastSettings::enableObjectPicking;

		JpegPipeline jpp;
		jpp.toDecode             = (u32*)jpegTextures->cptr_toDecode;
		jpp.toDecodeCounter      = (u32*)jpegTextures->cptr_toDecodeCounter;
		jpp.decoded              = (u32*)jpegTextures->cptr_decoded;
		jpp.TBSlots              = (u32*)jpegTextures->cptr_TBSlots;
		jpp.TBSlotsCounter       = (u32*)jpegTextures->cptr_TBSlotsCounter;
		jpp.decodedMcuMap        = *jpegTextures->decodedMcuMap;
		
		static CUdeviceptr cptr_textures = MemoryManager::alloc(MAX_TEXTURES * sizeof(Texture), "texture list");

		if(hasJpegCompressedTextures){
			cuMemcpyHtoD(cptr_textures, TextureManager::textures, TextureManager::numTextures * sizeof(Texture));
			cuMemsetD32(jpegTextures->cptr_toDecodeCounter, 0, 1);
			// cuMemsetD32(cptr_TBSlotsCounter, 0, 1);
		}


		{ // RESOLVE VISIBILITY BUFFER (write colors to colorbuffer)
			// Build sphere args for the resolve kernel (composite sphere over triangles)
			SphereRasterArgs sphereResolveArgs;
			sphereResolveArgs.sphere_framebuffer = (uint64_t*)cvm_sphere_framebuffer->cptr;
			sphereResolveArgs.numSpheres         = 0;
			sphereResolveArgs.positions          = nullptr;
			sphereResolveArgs.radii              = nullptr;
			sphereResolveArgs.colors             = nullptr;
			sphereResolveArgs.atomTypes          = nullptr;
			sphereResolveArgs.colorPalette       = nullptr;
			sphereResolveArgs.atomAO             = nullptr;
			sphereResolveArgs.lod                = frameLod;

			// Collect all sphere geometry pointers for resolve-time raycast
			// We use the first SNSpheres node for the resolve pass (multi-batch handled by the visbuffer sphereIdx encoding)
			// For multi-node support the sphere arrays would need to be concatenated; for now one node is the common case.
			scene->forEach<SNSpheres>([&](SNSpheres* node){
				if(node->numSpheres > 0 && sphereResolveArgs.numSpheres == 0) {
					sphereResolveArgs.positions    = (vec3*)node->cptr_positions;
					sphereResolveArgs.radii        = (float*)node->cptr_radii;
					sphereResolveArgs.colors       = (uint32_t*)node->cptr_colors;
					sphereResolveArgs.atomTypes    = (uint8_t*)node->cptr_atomTypes;
					sphereResolveArgs.colorPalette = (uint32_t*)node->cptr_palette;
					sphereResolveArgs.numSpheres   = node->numSpheres;

					// Bake on demand. One-time and view-independent, so it is not redone
					// while orbiting; the result stays on the node until the scene changes.
					if(CuRastSettings::atomAOEnabled && node->cptr_atomAO == 0){
						atomao::BakeSettings bs;
						bs.numDirections = CuRastSettings::atomAODirections;
						bs.resolution    = CuRastSettings::atomAOResolution;
						bs.intensity     = CuRastSettings::atomAOIntensity;

						Box3 box = node->aabb;
						node->cptr_atomAO = atomao::bake(
							node->cptr_positions, node->numSpheres,
							box.min, box.max, CuRastSettings::atomAOMaxRadius, bs);
					}

					if(CuRastSettings::atomAOEnabled){
						sphereResolveArgs.atomAO = (uint8_t*)node->cptr_atomAO;
					}
				}
			});

			void* args[] = {
				&cvm_instances->cptr,
				&numInstances,
				&cvm_triangleCountPrefixsum->cptr,
				&mouse_X,
				&mouse_Y,
				&cptr_state,
				&rasterSettings,
				&jpp,
				&sphereResolveArgs,
				&cvm_normalbuffer->cptr,
			};
			prog->launch2D("kernel_resolve_visbuffer_to_colorbuffer2D", args, target.width, target.height);
		}
		
		drawPoints(scene, view, target);
		
		drawTrianglesTranslucent(
			scene, view, meshes_unique_translucent, meshes_allInstances, 
			cvm_meshes_translucent->cptr, 
			cvm_instances->cptr, cvm_transforms->cptr, cvm_triangleCountPrefixsum->cptr,
			target, mappings
		);

		if(hasJpegCompressedTextures){
			u32 toDecodeCounter;
			cuMemcpyDtoH(&toDecodeCounter, (CUdeviceptr)jpp.toDecodeCounter, 4);
			dvlist.push_back({"toDecodeCounter ", format("{}", toDecodeCounter)});

			// DECODE JPEG TEXTURES
			jpegTextures->prog->launch("kernel_launch_decode", {
				&jpegTextures->cptr_toDecodeCounter,
				&jpegTextures->cptr_TBSlots, 
				&jpegTextures->cptr_TBSlotsCounter,
				&jpegTextures->cptr_toDecode,
				&jpegTextures->cptr_decoded,
				&cptr_textures,
				// &jpegTextures->cptr_texture_pointer,
				jpegTextures->decodedMcuMap,
			}, 1);


			{ // RESOLVE JPEG
				void* args[] = {
					&cvm_instances->cptr,
					&numInstances,
					&cvm_triangleCountPrefixsum->cptr,
					&mouse_X,
					&mouse_Y,
					&cptr_state,
					&rasterSettings,
					&jpp,
					&cptr_textures
				};
				prog->launch2D("kernel_resolve_jpeg", args, target.width, target.height);
			}

			{// DEBUG
				u32 C = CuRast::deviceState->dbg_hovered_decoded_color;
				uint8_t* rgba = (uint8_t*)&C;
				string strColor = format("{:3}, {:3}, {:3}", rgba[0], rgba[1], rgba[2]);

				dvlist.push_back({"CPU draw() duration    ", format("{:.1f} ms", Runtime::duration_draw * 1000.0)});
				dvlist.push_back({"hovered_textureHandle  ", format("{:12}", CuRast::deviceState->dbg_hovered_textureHandle)});
				dvlist.push_back({"hovered_mipLevel       ", format("{:12}", CuRast::deviceState->dbg_hovered_mipLevel)});
				dvlist.push_back({"hovered_tx             ", format("{:12}", CuRast::deviceState->dbg_hovered_tx)});
				dvlist.push_back({"hovered_ty             ", format("{:12}", CuRast::deviceState->dbg_hovered_ty)});
				dvlist.push_back({"hovered_mcu_x          ", format("{:12}", CuRast::deviceState->dbg_hovered_mcu_x)});
				dvlist.push_back({"hovered_mcu_y          ", format("{:12}", CuRast::deviceState->dbg_hovered_mcu_y)});
				dvlist.push_back({"hovered_mcu            ", format("{:12}", CuRast::deviceState->dbg_hovered_mcu)});
				dvlist.push_back({"hovered_decoded_color  ", format("{:12}", strColor)});
			}
		
			cuMemsetD8((CUdeviceptr)jpegTextures->decodedMcuMap_tmp->entries, 0xff, jpegTextures->decodedMcuMap_tmp->capacity * 8);
			// bool freezeCache = editor->settings.freezeCache;
			bool freezeCache = false;
			jpegTextures->prog->launch("kernel_update_cache", {
				jpegTextures->decodedMcuMap, 
				jpegTextures->decodedMcuMap_tmp, 
				&jpegTextures->cptr_TBSlots,
				&jpegTextures->cptr_TBSlotsCounter,
				&freezeCache
			}, jpegTextures->decodedMcuMap->capacity);
			cuMemcpy((CUdeviceptr)jpegTextures->decodedMcuMap->entries, (CUdeviceptr)jpegTextures->decodedMcuMap_tmp->entries, jpegTextures->decodedMcuMap_tmp->capacity * 8);

			// {
			// 	// Disable caching by fully clearing the MCU slot list and hash map at the end of each frame.
			// 	// This let's us see how much slower the decode kernel becomes.
			// 	cuMemsetD8((CUdeviceptr)jpegTextures->decodedMcuMap->entries, 0xff, jpegTextures->decodedMcuMap->capacity * 8);
			// 	u32 capacity = JPEG_NUM_DECODED_MCU_CAPACITY;
			// 	jpegTextures->prog->launch("kernel_init_availableMcuSlots", {
			// 		&jpegTextures->cptr_TBSlots, 
			// 		&jpegTextures->cptr_TBSlotsCounter, 
			// 		&capacity
			// 	}, capacity,  0);

			// 	cuMemsetD32(jpegTextures->cptr_TBSlotsCounter, 0, 1);
			// }
		}

		// { // TEST: Draw Heightmap
		// 	static CudaModularProgram* prog = new CudaModularProgram({"./src/kernels/triangles_heightmap.cu",});

		// 	CUdeviceptr cptr_target = prog->getGlobalsPointer("c_target");
		// 	cuMemcpyHtoDAsync(cptr_target, &target, sizeof(target), 0);

		// 	float w = settings.threshold;
		// 	int minCells = 128;
		// 	int maxCells = 40 * 1024;
		// 	int numCells = (1.0f - w) * float(minCells) + w * float(maxCells);
			

		// 	// int numCells = 5 * 1024;
		// 	int blocksize = 16;
		// 	int numBlocks = (numCells + blocksize - 1) / blocksize;

		// 	void* args[] = {
		// 		&numCells,
		// 		&cptr_colorbuffer
		// 	};

		// 	auto custart = Timer::recordCudaTimestamp();

		// 	auto res_launch = cuLaunchKernel(prog->kernels["kernel_drawHeightmap"],
		// 		numBlocks, numBlocks, 1,
		// 		blocksize, blocksize, 1,
		// 		0, 0, args, nullptr);

		// 	Timer::recordDuration("kernel_drawHeightmap", custart, Timer::recordCudaTimestamp());
		// }



		// SCREEN SPACE AMBIENT OCCLUSION
		static CudaVirtualMemory* cvm_ssaoShadebuffer = MemoryManager::allocVirtualCuda(2'000'000'000, "cvm_ssaoShadebuffer");
		if(CuRastSettings::enableSSAO){
			// save mem by using reusing the visibility buffer, which is no longer used in this frame
			CUdeviceptr cptr_occlusionbuffer = cvm_framebuffer->cptr;

			// But for the final ssao shading values, we need an extra buffer
			cvm_ssaoShadebuffer->commit(cvm_framebuffer->comitted / 2);

			// Multiscale SSAO parameters: per-level radius (as fraction of view-space depth)
			// and per-level bias factor. When enableMultiscaleSSAO is off we still use the
			// kernel's level-loop, but with one level at the original 0.2025 radius.
			// All radii are multiplied by the global ssaoRadiusScale so the user can retune
			// for any scene with a single slider.
			float r0, r1, r2, r3, b0, b1, b2, b3;
			int   s0, s1, s2, s3;
			int   numLevels;
			float scl = CuRastSettings::ssaoRadiusScale;
			if(CuRastSettings::enableMultiscaleSSAO){
				r0 = CuRastSettings::ssaoLevelRadius[0] * scl;
				r1 = CuRastSettings::ssaoLevelRadius[1] * scl;
				r2 = CuRastSettings::ssaoLevelRadius[2] * scl;
				r3 = CuRastSettings::ssaoLevelRadius[3] * scl;
				b0 = CuRastSettings::ssaoLevelBias[0];
				b1 = CuRastSettings::ssaoLevelBias[1];
				b2 = CuRastSettings::ssaoLevelBias[2];
				b3 = CuRastSettings::ssaoLevelBias[3];
				s0 = CuRastSettings::ssaoSamplesPerLevel[0];
				s1 = CuRastSettings::ssaoSamplesPerLevel[1];
				s2 = CuRastSettings::ssaoSamplesPerLevel[2];
				s3 = CuRastSettings::ssaoSamplesPerLevel[3];
				numLevels = CuRastSettings::ssaoLevels;
			} else {
				// Single-scale: take the user's level-0 radius so the tuning slider works.
				r0 = CuRastSettings::ssaoLevelRadius[0] * scl;
				r1 = r2 = r3 = 0.0f;
				b0 = CuRastSettings::ssaoLevelBias[0];
				b1 = b2 = b3 = 0.0f;
				s0 = CuRastSettings::ssaoSamplesPerLevel[0];
				s1 = s2 = s3 = 0;
				numLevels = 1;
			}
			float intensity       = CuRastSettings::ssaoIntensity;

			// One-shot report of what actually reached the kernel, plus the camera
			// framing. SSAO cost scales with covered pixels, so a timing is only
			// comparable between runs that share a framing.
			static bool ssaoConfigReported = false;
			if(!ssaoConfigReported){
				ssaoConfigReported = true;
				println("SSAO: multiscale={} levels={} taps/pixel={} intensity={:.2f}",
					CuRastSettings::enableMultiscaleSSAO, numLevels,
					(numLevels > 0 ? s0 : 0) + (numLevels > 1 ? s1 : 0)
					+ (numLevels > 2 ? s2 : 0) + (numLevels > 3 ? s3 : 0),
					intensity);
				println("SSAO: radii=[{:.4f} {:.4f} {:.4f} {:.4f}] samples=[{} {} {} {}] scale={:.2f}",
					r0, r1, r2, r3, s0, s1, s2, s3, scl);
				println("SSAO: camera radius={:.1f} yaw={:.3f} pitch={:.3f} viewport={}x{}",
					Runtime::controls->radius, Runtime::controls->yaw,
					Runtime::controls->pitch, target.width, target.height);
			}

			void* argsSSAO[] = {
				&cvm_framebuffer->cptr,
				&cvm_ssaoShadebuffer->cptr,
				&cvm_normalbuffer->cptr,
				&r0, &r1, &r2, &r3,
				&b0, &b1, &b2, &b3,
				&numLevels,
				&s0, &s1, &s2, &s3,
				&intensity,
			};
			void* argsBlur[] = {
				&cvm_framebuffer->cptr,
				&cvm_ssaoShadebuffer->cptr
			};

			float gtaoRadius    = CuRastSettings::gtaoRadius;
			int   gtaoSlices    = CuRastSettings::gtaoSlices;
			int   gtaoSteps     = CuRastSettings::gtaoSteps;
			float gtaoIntensity = CuRastSettings::gtaoIntensity;
			float gtaoThickness = CuRastSettings::gtaoThickness;

			void* argsGTAO[] = {
				&cvm_framebuffer->cptr,
				&cvm_ssaoShadebuffer->cptr,
				&cvm_normalbuffer->cptr,
				&gtaoRadius,
				&gtaoSlices,
				&gtaoSteps,
				&gtaoIntensity,
				&gtaoThickness,
			};

			// Both paths write the same packed depth+AO into the scratch framebuffer,
			// so the bilateral blur downstream is shared.
			if(CuRastSettings::aoMode == 1){
				prog->launch2D("kernel_gtaoOcclusion", argsGTAO, target.width, target.height);
			}else{
				prog->launch2D("kernel_ssaoOcclusion", argsSSAO, target.width, target.height);
			}
			prog->launch2D("kernel_ssaoBlur", argsBlur, target.width, target.height);
		}

		// static CUdeviceptr cptr_enlarged = CURuntime::alloc("enlarge", 4096 * 4096 * 8);
		// prog->launchCooperative("kernel_enlarge", {
		// 	&mappings.surfaces[0],
		// 	&cptr_ssaoShadebuffer,
		// 	&cptr_enlarged,
		// 	&view.framebuffer->width, 
		// 	&view.framebuffer->height,
		// 	&mouse_X,
		// 	&mouse_Y,
		// 	&cptr_state,
		// 	&CuRastSettings::enableEDL,
		// 	&CuRastSettings::enableSSAO,
		// });

		{ // RESOLVE COLOR BUFFER (write to graphics API framebuffer)
			int viewWidth = view.framebuffer->width;
			int viewHeight = view.framebuffer->height;

			u32 backgroundColor = 0;
			uint8_t* bgRgba = (uint8_t*)&backgroundColor;
			bgRgba[0] = clamp(CuRastSettings::background.x * 256.0f, 0.0f, 255.0f);
			bgRgba[1] = clamp(CuRastSettings::background.y * 256.0f, 0.0f, 255.0f);
			bgRgba[2] = clamp(CuRastSettings::background.z * 256.0f, 0.0f, 255.0f);

			void* args[] = {
				&mappings.surfaces[0],
				&cvm_ssaoShadebuffer->cptr,
				&viewWidth, 
				&viewHeight,
				&mouse_X,
				&mouse_Y,
				&cptr_state,
				&CuRastSettings::enableEDL,
				&CuRastSettings::enableSSAO,
				&CuRastSettings::showInset,
				&backgroundColor,
				&cvm_normalbuffer->cptr
			};
			prog->launch2D("kernel_resolve_colorbuffer_to_opengl_2D", args, target.width, target.height);
		}

		// Deterministic capture for rendering work. Set CURAST_AUTOSHOT=<path> and
		// optionally CURAST_AUTOSHOT_FRAME=<n> (default 150) to save one screenshot at
		// a fixed frame and print the matching per-kernel timings.
		//
		// Both halves matter for A/B-ing shading changes. Fixed framing is required
		// because comparing two runs framed by hand compares two different views, and
		// the per-kernel mean is the only usable speed signal here -- whole-frame FPS
		// on this scene swung 86-158 within a single run, which is far wider than any
		// effect worth measuring.
		{
			static int  autoshotFrame = 0;
			static bool autoshotDone  = false;
			const char* autoshotPath  = getenv("CURAST_AUTOSHOT");

			if(autoshotPath != nullptr && !autoshotDone){
				const char* strWhen = getenv("CURAST_AUTOSHOT_FRAME");
				int when = (strWhen != nullptr) ? atoi(strWhen) : 150;

				// Optional fixed camera: CURAST_AUTOSHOT_VIEW="yaw,pitch,radius,tx,ty,tz".
				// Frame count alone is NOT a stable framing anchor -- the camera is
				// auto-framed when the model finishes loading, and load time varies
				// between runs, so the same frame number gave a close-up in one run and
				// a wide shot in the next. Re-applied every frame until the capture so
				// it wins over auto-framing regardless of when loading completes.
				const char* strView = getenv("CURAST_AUTOSHOT_VIEW");
				if(strView != nullptr){
					float yaw, pitch, radius, tx, ty, tz;
					if(sscanf(strView, "%f,%f,%f,%f,%f,%f", &yaw, &pitch, &radius, &tx, &ty, &tz) == 6){
						Runtime::controls->yaw    = yaw;
						Runtime::controls->pitch  = pitch;
						Runtime::controls->radius = radius;
						Runtime::controls->target = {tx, ty, tz};
					}
				}

				if(++autoshotFrame >= when){
					CuRastSettings::requestScreenshot = make_shared<string>(string(autoshotPath));
					autoshotDone = true;

					println("AUTOSHOT: frame={} path={}", autoshotFrame, autoshotPath);
					for(string label : {
						"kernel_ssaoOcclusion",
						"kernel_gtaoOcclusion",
						"kernel_ssaoBlur",
						"kernel_resolve_colorbuffer_to_opengl_2D",
					}){
						float mean = Runtime::timings.getMean(label);
						if(mean > 0.0f) println("AUTOSHOT: {} = {:.3f} ms", label, mean);
					}
				}
			}
		}

		if(CuRastSettings::requestScreenshot){
			bool wasAutoshot = (getenv("CURAST_AUTOSHOT") != nullptr)
			                && (*CuRastSettings::requestScreenshot == string(getenv("CURAST_AUTOSHOT")));

			saveScreenshot(target, view, cvm_ssaoShadebuffer->cptr, prog);

			// Quit after a scripted capture so a sweep can run captures back to back.
			// Whether the app exits on its own after a screenshot is not reliable, and
			// a run that stays open blocks the next one in the batch.
			if(wasAutoshot && getenv("CURAST_AUTOSHOT_EXIT") != nullptr){
				println("AUTOSHOT: saved, exiting");
				fflush(stdout);
				exit(0);
			}
		}

		unmapCudaVk(mappings);
		
		cuMemcpyDtoHAsync((void*)deviceState, cptr_state, sizeof(DeviceState), 0);

		if(deviceState->dbg_fragcount > 0){
			dvlist.push_back({"fragcounter", format("{:L}", deviceState->dbg_fragcount)});
		}
	}

	CuRastSettings::requestScreenshot = nullptr;
}

void initialize(){
	if(initialized) return;

	int defaultPixels = 1920 * 1080;
	int64_t virtualCapacity = 2'147'483'648; // sufficient for up to 4096 x 4096 pixels with 16x supersampling
	// int max_SuperSamples = 16;
	cvm_framebuffer = MemoryManager::allocVirtualCuda(virtualCapacity, "framebuffer");
	cvm_framebuffer->commit(8 * defaultPixels);

	cvm_colorbuffer = MemoryManager::allocVirtualCuda(virtualCapacity, "colorbuffer");
	cvm_colorbuffer->commit(8 * defaultPixels);

	cvm_sphere_framebuffer = MemoryManager::allocVirtualCuda(virtualCapacity, "sphere_framebuffer");
	cvm_sphere_framebuffer->commit(8 * defaultPixels);

	cvm_normalbuffer = MemoryManager::allocVirtualCuda(virtualCapacity, "normalbuffer");
	cvm_normalbuffer->commit(4 * defaultPixels);

	sphereProg = new CudaModularProgram(CudaModularProgram::CudaModularProgramArgs{
		.modules = {"./src/kernels/spheres_visbuffer.cu"},
		.useLTO  = false,
	});

	jpegTextures = new JpegTextures();

	initialized = true;
}

void CuRast::render(){

	if(VKRenderer::width * VKRenderer::height == 0){
		return;
	}

	initialize();

	VKRenderer::view.framebuffer->setSize(VKRenderer::width, VKRenderer::height);
	
	// RENDER DESKTOP
	VKRenderer::view.proj =  VKRenderer::camera->proj;
	VKRenderer::view.view =  mat4(VKRenderer::camera->view);
	
	draw(&scene, {VKRenderer::view});

	Runtime::debugValues["small"]   = format(getSaneLocale(), "{:L}", deviceState->numSmall);
	Runtime::debugValues["large"]   = format(getSaneLocale(), "{:L}", deviceState->numLarge);
	Runtime::debugValues["massive"] = format(getSaneLocale(), "{:L}", deviceState->numMassive);

	{ // DRAW GUI
		ImGui::NewFrame();
		// ImGuizmo::BeginFrame();

		drawGUI();

		ImGui::Render();
	}

	Runtime::mouseEvents.clear();
}