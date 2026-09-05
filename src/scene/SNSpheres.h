
#pragma once

#include "../scene/SceneNode.h"

struct SNSpheres : public SceneNode {
	uint64_t cptr_positions = 0;  // vec3* GPU, world space
	uint64_t cptr_radii     = 0;  // float* GPU
	uint64_t cptr_colors    = 0;  // uint32_t* RGBA8 GPU — only allocated for CHAIN/ENTITY themes
	uint64_t cptr_atomTypes = 0;  // uint8_t* per-atom palette index (16-byte saving vs cptr_colors)
	uint64_t cptr_palette   = 0;  // uint32_t[256] RGBA8 LUT indexed by atom type
	uint32_t numSpheres     = 0;

	SNSpheres() : SceneNode() {}
	SNSpheres(string name) : SceneNode(name) {}
};
