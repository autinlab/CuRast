// tinyexr's implementation is large (~380 kB of header) and pulls in miniz, so it is
// compiled here exactly once rather than in every translation unit that wants to read
// an environment map.

#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_MINIZ (1)
#define TINYEXR_USE_STB_ZLIB (0)
#include "tinyexr/tinyexr.h"

#include "EnvMapImage.h"

namespace envmap {

using std::string;
using std::println;

Image loadEXR(const string& path){
	Image img;

	float* rgba = nullptr;
	int width = 0, height = 0;
	const char* err = nullptr;

	// LoadEXR always returns 4 channels regardless of what the file stores.
	int ret = ::LoadEXR(&rgba, &width, &height, path.c_str(), &err);

	if(ret != TINYEXR_SUCCESS){
		println("EnvMap: could not read EXR '{}': {}", path, err ? err : "(no message)");
		if(err) FreeEXRErrorMessage(err);
		return img;
	}

	img.width  = width;
	img.height = height;
	img.rgb.resize(size_t(width) * height * 3);
	for(size_t i = 0; i < size_t(width) * height; i++){
		img.rgb[i * 3 + 0] = rgba[i * 4 + 0];
		img.rgb[i * 3 + 1] = rgba[i * 4 + 1];
		img.rgb[i * 3 + 2] = rgba[i * 4 + 2];
	}

	free(rgba);
	return img;
}

} // namespace envmap
