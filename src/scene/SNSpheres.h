
#pragma once

#include "../scene/SceneNode.h"

struct SNSpheres : public SceneNode {
	uint64_t cptr_positions = 0;  // vec3* GPU, world space
	uint64_t cptr_radii     = 0;  // float* GPU
	uint64_t cptr_colors    = 0;  // uint32_t* RGBA8 GPU — only allocated for CHAIN/ENTITY themes
	uint64_t cptr_atomTypes = 0;  // uint8_t* per-atom palette index (16-byte saving vs cptr_colors)
	uint64_t cptr_palette   = 0;  // uint32_t[256] RGBA8 LUT indexed by atom type
	// QuteMol-style baked per-atom ambient occlusion, one byte per atom. 0 = not baked.
	// View-independent, so it survives orbiting and costs nothing per frame.
	uint64_t cptr_atomAO    = 0;  // uint8_t* GPU
	uint32_t numSpheres     = 0;

	SNSpheres() : SceneNode() {}
	SNSpheres(string name) : SceneNode(name) {}
};
