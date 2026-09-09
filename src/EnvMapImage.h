#pragma once

// Decoding half of the environment-map support, deliberately free of any dependency on
// kernels/HostDeviceInterface.h.
//
// That header is written to be included from CUDA translation units and from host code
// that has already pulled in the project preamble (it uses min/max and the CUDA
// __constant__ keyword unqualified). EnvMap.cpp is a standalone TU that only needs to
// decode pixels, so it includes this header and never the device-facing one.

#include <string>
#include <vector>
#include <cctype>
#include <print>

#include "stb/stb_image.h"

namespace envmap {

struct Image {
	int width  = 0;
	int height = 0;
	std::vector<float> rgb;   // width * height * 3, linear radiance
	bool ok() const {
		return width > 0 && height > 0 && rgb.size() == size_t(width) * size_t(height) * 3;
	}
};

// Implemented in EnvMap.cpp so tinyexr's ~370 kB implementation compiles exactly once.
Image loadEXR(const std::string& path);

inline Image loadHDR(const std::string& path){
	Image img;
	int n = 0;
	float* data = stbi_loadf(path.c_str(), &img.width, &img.height, &n, 3);
	if(data == nullptr){
		std::println("EnvMap: stb_image could not read '{}': {}", path, stbi_failure_reason());
		img.width = img.height = 0;
		return img;
	}
	img.rgb.assign(data, data + size_t(img.width) * size_t(img.height) * 3);
	stbi_image_free(data);
	return img;
}

inline bool iEndsWithLower(const std::string& s, const std::string& suffix){
	if(s.size() < suffix.size()) return false;
	for(size_t i = 0; i < suffix.size(); i++){
		char a = (char)std::tolower((unsigned char)s[s.size() - suffix.size() + i]);
		if(a != suffix[i]) return false;
	}
	return true;
}

inline Image load(const std::string& path){
	if(iEndsWithLower(path, ".exr")) return loadEXR(path);
	if(iEndsWithLower(path, ".hdr")) return loadHDR(path);
	std::println("EnvMap: unsupported extension for '{}' (expected .exr or .hdr)", path);
	return Image{};
}

} // namespace envmap
