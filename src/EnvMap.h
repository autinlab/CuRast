#pragma once

// Projects an HDR environment map onto the 9 spherical-harmonic irradiance
// coefficients used by the sphere/triangle shading in resolve.cu.
//
// Formats are handled in EnvMapImage.h:
//   .exr  - via tinyexr (handles PIZ, a wavelet+Huffman codec, which is what
//           PolyHaven ships and is not worth hand-rolling)
//   .hdr  - via stb_image's Radiance RGBE reader, already vendored
//
// This is the runtime counterpart of src/tools/sh9.py. The Python tool bakes the
// default coefficients compiled into resolve.cu; this loads a map chosen at runtime
// and overrides them.
//
// Convention: world up is +Z, matching OrbitControls. The map's top row maps to +Z and
// the horizontal axis sweeps phi about Z.
//
// Include only from translation units that already have the project preamble, since
// kernels/HostDeviceInterface.h needs it. EnvMap.cpp deliberately does not.

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <print>

#include "glm/glm.hpp"
#include "kernels/HostDeviceInterface.h"
#include "EnvMapImage.h"

namespace envmap {

// MSVC does not define M_PI without _USE_MATH_DEFINES, and defining that macro from a
// header would leak into every translation unit that includes this one.
constexpr double ENVMAP_PI = 3.14159265358979323846;

// Project an equirectangular map onto SH9 irradiance and pick a directional key.
//
// The key light matters because SH up to l=2 is far too low-frequency to represent a
// light source, so diffuse SH alone can never produce a specular highlight. Taking the
// brightest 0.5% of the map's energy gives a direction and colour to hang one on.
inline EnvLighting project(const Image& img, float exposure){
	EnvLighting env = {};
	env.exposure = exposure;
	env.enabled  = 0;

	if(!img.ok()) return env;

	const int W = img.width;
	const int H = img.height;

	// A_l for the 3 bands (Ramamoorthi & Hanrahan convolution coefficients).
	const double A[9] = {
		3.141593,
		2.094395, 2.094395, 2.094395,
		0.785398, 0.785398, 0.785398, 0.785398, 0.785398
	};

	double L[9][3] = {};
	double keyAccum[3] = {};
	double keyDir[3]   = {};
	double keyWeight   = 0.0;

	// Luminance threshold for the key: the top 0.5% of pixels by luminance.
	std::vector<float> lum(size_t(W) * size_t(H));
	for(size_t i = 0; i < lum.size(); i++){
		lum[i] = 0.2126f * img.rgb[i * 3 + 0]
		       + 0.7152f * img.rgb[i * 3 + 1]
		       + 0.0722f * img.rgb[i * 3 + 2];
	}
	std::vector<float> sorted = lum;
	size_t kth = (size_t)(double(sorted.size()) * 0.995);
	if(kth >= sorted.size()) kth = sorted.size() - 1;
	std::nth_element(sorted.begin(), sorted.begin() + kth, sorted.end());
	float keyThreshold = sorted[kth];

	for(int y = 0; y < H; y++){
		double theta = (y + 0.5) / H * ENVMAP_PI;      // 0 at the top row -> +Z
		double st = sin(theta);
		double ct = cos(theta);
		double dw = (2.0 * ENVMAP_PI / W) * (ENVMAP_PI / H) * st;

		for(int x = 0; x < W; x++){
			double phi = (x + 0.5) / W * 2.0 * ENVMAP_PI - ENVMAP_PI;
			double dx = st * cos(phi);
			double dy = st * sin(phi);
			double dz = ct;

			double Y[9];
			Y[0] = 0.282095;
			Y[1] = 0.488603 * dy;
			Y[2] = 0.488603 * dz;
			Y[3] = 0.488603 * dx;
			Y[4] = 1.092548 * dx * dy;
			Y[5] = 1.092548 * dy * dz;
			Y[6] = 0.315392 * (3.0 * dz * dz - 1.0);
			Y[7] = 1.092548 * dx * dz;
			Y[8] = 0.546274 * (dx * dx - dy * dy);

			size_t idx = size_t(y) * size_t(W) + size_t(x);
			double r = img.rgb[idx * 3 + 0];
			double g = img.rgb[idx * 3 + 1];
			double b = img.rgb[idx * 3 + 2];

			for(int i = 0; i < 9; i++){
				double w = Y[i] * dw;
				L[i][0] += w * r;
				L[i][1] += w * g;
				L[i][2] += w * b;
			}

			if(lum[idx] >= keyThreshold){
				double w = lum[idx] * dw;
				keyDir[0]   += w * dx;
				keyDir[1]   += w * dy;
				keyDir[2]   += w * dz;
				keyAccum[0] += w * r;
				keyAccum[1] += w * g;
				keyAccum[2] += w * b;
				keyWeight   += w;
			}
		}
	}

	for(int i = 0; i < 9; i++){
		env.sh[i] = glm::vec4(float(A[i] * L[i][0]), float(A[i] * L[i][1]), float(A[i] * L[i][2]), 0.0f);
	}

	double dlen = sqrt(keyDir[0]*keyDir[0] + keyDir[1]*keyDir[1] + keyDir[2]*keyDir[2]);
	if(dlen > 1e-12 && keyWeight > 0.0){
		env.keyDir = glm::vec4(float(keyDir[0]/dlen), float(keyDir[1]/dlen), float(keyDir[2]/dlen), 0.0f);
		double cr = keyAccum[0]/keyWeight, cg = keyAccum[1]/keyWeight, cb = keyAccum[2]/keyWeight;
		double peak = std::max(cr, std::max(cg, cb));
		if(peak < 1e-9) peak = 1.0;
		env.keyColor = glm::vec4(float(cr/peak), float(cg/peak), float(cb/peak), 0.0f);
	}else{
		// Degenerate map (uniform or black): fall back to a neutral overhead key rather
		// than leaving a zero direction that would normalize to NaN on the device.
		env.keyDir   = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
		env.keyColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
	}

	env.enabled = 1;
	return env;
}

inline EnvLighting loadAndProject(const std::string& path, float exposure){
	Image img = load(path);
	if(!img.ok()){
		EnvLighting env = {};
		env.exposure = exposure;
		env.enabled  = 0;
		return env;
	}

	EnvLighting env = project(img, exposure);

	std::println("EnvMap: '{}' {}x{}", path, img.width, img.height);
	std::println("EnvMap: SH Y00 = ({:.3f}, {:.3f}, {:.3f})  key dir = ({:.3f}, {:.3f}, {:.3f})",
		env.sh[0].x, env.sh[0].y, env.sh[0].z,
		env.keyDir.x, env.keyDir.y, env.keyDir.z);

	return env;
}

} // namespace envmap
