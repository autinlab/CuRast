
#include "mmcif/MMCIFLoader.h"

void makeSphereLodControls(){
	if(!ImGui::CollapsingHeader("Sphere LOD")) return;

	ImGui::Checkbox("Enable Sphere LOD", &CuRastSettings::enableSphereLOD);
	if(ImGui::IsItemHovered()) ImGui::SetTooltip(
		"Skip atoms by index stride and grow radius as camera distance increases.\n"
		"Stored values are normalised to the orbit radius and rescaled per frame.");

	float orbit = (float)Runtime::controls->radius;
	ImGui::Text("Orbit radius: %.1f world units (current scene scale multiplier)", orbit);

	SphereLodConfig& cfg = CuRastSettings::sphereLodConfig;
	int n = cfg.numLevels;
	if(ImGui::SliderInt("Levels", &n, 0, MAX_SPHERE_LOD_LEVELS)) cfg.numLevels = n;

	for(int k = 0; k < cfg.numLevels; k++){
		ImGui::PushID(k);
		float effMin = cfg.levels[k].minDist * orbit;
		float effMax = cfg.levels[k].maxDist * orbit;
		bool  isActive = (orbit >= effMin && orbit <= effMax);
		ImVec4 color = isActive ? ImVec4(0.5f, 1.0f, 0.5f, 1.0f) : ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
		ImGui::TextColored(color,
			"Level %d  stride=%d  effective band = [%.0f .. %.0f]  %s",
			k, cfg.levels[k].stride, effMin, effMax, isActive ? "ACTIVE" : "");
		ImGui::SliderInt  ("stride",   &cfg.levels[k].stride,   1,    1024);
		ImGui::SliderFloat("scale",    &cfg.levels[k].scale,    0.1f, 16.0f);
		ImGui::SliderFloat("minDist",  &cfg.levels[k].minDist,  0.0f, 50.0f,  "%.2f x orbit");
		ImGui::SliderFloat("maxDist",  &cfg.levels[k].maxDist,  0.0f, 200.0f, "%.2f x orbit");
		ImGui::SliderFloat("overlap",  &cfg.levels[k].overlap,  0.0f, 20.0f,  "%.2f x orbit");
		ImGui::PopID();
	}
}

// The lighting and shading controls used to live inside one "SSAO" header, which
// both buried them and made them unreachable whenever SSAO was switched off --
// the camera and the environment have nothing to do with occlusion. One header
// per effect instead.

void makeCameraControls(){
	if(!ImGui::CollapsingHeader("Camera")) return;

	int projMode = CuRastSettings::orthographic ? 1 : 0;
	if(ImGui::Combo("Projection", &projMode, "Perspective\0Orthographic\0\0")){
		CuRastSettings::orthographic = (projMode == 1);
	}
	ImGui::SetItemTooltip(
		"Orthographic removes foreshortening, so the far side of an assembly is "
		"drawn at the same scale as the near side. The view height is derived from "
		"the orbit distance, so switching modes keeps the framing.");
	if(CuRastSettings::orthographic){
		ImGui::SliderFloat("Ortho zoom", &CuRastSettings::orthoZoom, 0.05f, 4.0f,
			"%.3f", ImGuiSliderFlags_Logarithmic);
	}
	ImGui::SliderInt("Supersampling", &CuRastSettings::supersamplingFactor, 1, 4);
	ImGui::SetItemTooltip(
		"Renders at N x resolution and downsamples. At whole-cell framing ~90 atoms "
		"fall in each pixel and only one is kept, so aliasing is the dominant artefact "
		"there -- often a bigger visual win than further AO tuning. Cost scales with N^2.");
}

void makeEnvironmentControls(){
	if(!ImGui::CollapsingHeader("Environment lighting")) return;

	static char envPathBuf[512] = {};
	static bool envPathInit = false;
	if(!envPathInit){
		snprintf(envPathBuf, sizeof(envPathBuf), "%s", CuRastSettings::envMapPath.c_str());
		envPathInit = true;
	}

	ImGui::Checkbox("Enable##env", &CuRastSettings::envEnabled);
	ImGui::SetItemTooltip(
		"Off gives a uniform ambient, so the toggle shows what the environment "
		"actually contributes. The loaded map is kept, so switching back on is free.");
	ImGui::SameLine();
	ImGui::Checkbox("Show background", &CuRastSettings::envShowBackground);
	ImGui::SetItemTooltip(
		"Draw the map behind the model. Off keeps the environment lighting the scene "
		"but leaves the background the solid colour -- usually what you want for a figure.");

	ImGui::InputText("Map (.exr/.hdr)", envPathBuf, sizeof(envPathBuf));
	ImGui::SameLine();
	if(ImGui::Button("Load##env")){
		CuRastSettings::envMapPath   = envPathBuf;
		CuRastSettings::envMapReload = true;
	}
	ImGui::SetItemTooltip(
		"Equirectangular HDR map, projected to 9 SH irradiance coefficients on the "
		"host. Empty uses the studio coefficients baked into resolve.cu. World up is +Z.");

	ImGui::SliderFloat("Exposure", &CuRastSettings::envExposure, 0.05f, 4.0f,
		"%.2f", ImGuiSliderFlags_Logarithmic);
	ImGui::SliderFloat("Rotation", &CuRastSettings::envRotation, 0.0f, 360.0f, "%.0f deg");
	ImGui::SetItemTooltip("Rotates the lighting and the background together.");
	ImGui::SliderFloat("Background widen", &CuRastSettings::envBgWiden, 1.0f, 8.0f, "%.2f");
	ImGui::SetItemTooltip(
		"An environment map sits at infinity, so at a 60 degree fov you see only about "
		"a sixth of the panorama and whatever is behind the model looks enormous. "
		"Widening shows more of the map so its features read smaller. Framing only.");
}

void makeShadingControls(){
	if(!ImGui::CollapsingHeader("Shading")) return;

	ImGui::Checkbox("Flat spheres", &CuRastSettings::flatSpheres);
	ImGui::SetItemTooltip(
		"No directional term: uniform irradiance only. AO and the halo still apply, so "
		"shape comes from occlusion and outlines rather than from a light -- the "
		"illustrative look.");
	if(CuRastSettings::flatSpheres){
		ImGui::SliderFloat("Flat brightness", &CuRastSettings::flatBrightness, 0.5f, 5.0f);
		ImGui::SetItemTooltip(
			"Flat loses the directional term, which carries much of the average "
			"brightness, so it is lifted back to roughly the lit mode's level.");
	}

	ImGui::Checkbox("EDL", &CuRastSettings::enableEDL);
	ImGui::SetItemTooltip(
		"Eye-dome lighting: darkens depth discontinuities. Overlaps with the halo, and "
		"on dense atomic scenes it darkens nearly every pixel, so it is worth switching "
		"off while comparing anything else.");

	ImGui::Combo("Debug view", &CuRastSettings::debugView,
		"Off\0AO buffer\0Normals\0Impostor hit / fallback\0\0");
	ImGui::SetItemTooltip(
		"Normals: a correct impostor buffer looks like a field of tiny shaded spheres; "
		"a flat wash of one colour means the sub-pixel fallback dominates.\n"
		"Impostor: green = analytic ray-sphere hit, red = fallback.");
}

void makeBakedAOControls(){
	if(!ImGui::CollapsingHeader("Baked atom AO (QuteMol)")) return;

	ImGui::Checkbox("Enable##bakedao", &CuRastSettings::atomAOEnabled);
	ImGui::SetItemTooltip(
		"Object-space AO baked once at load, one byte per atom. View-independent, so it "
		"supplies the large-scale enclosure that screen-space AO cannot see when ~90 "
		"atoms share a pixel. Multiplies with GTAO. First enable triggers the bake.");

	if(!CuRastSettings::atomAOEnabled) return;

	ImGui::SliderInt("Bake directions", &CuRastSettings::atomAODirections, 8, 256);
	ImGui::SliderInt("Bake resolution", &CuRastSettings::atomAOResolution, 512, 8192);
	ImGui::SetItemTooltip(
		"Sweep buffer is resolution^2 and spans the whole bounding diagonal, so "
		"replicating the model spreads the same grid over a larger extent. Raise this "
		"after replicating, or surface atoms compete for pixels and come out dark.");
	ImGui::SliderFloat("Intensity##bakedao", &CuRastSettings::atomAOIntensity, 0.25f, 4.0f);
	ImGui::SliderFloat("Floor##bakedao", &CuRastSettings::atomAOFloor, 0.0f, 1.0f);
	ImGui::SetItemTooltip(
		"Buried atoms bake to near zero, which is correct but turns the view black once "
		"the camera is inside the structure, where everything visible is buried. Applied "
		"at shading time, so it retunes without a re-bake.");
	ImGui::TextDisabled("Directions/resolution changes need a reload to re-bake.");
}

void makeHaloControls(){
	if(!ImGui::CollapsingHeader("Halo / edge cueing (QuteMol)")) return;

	ImGui::Checkbox("Enable##halo", &CuRastSettings::haloEnabled);
	ImGui::SetItemTooltip(
		"Dark glow where a silhouette stands in front of something far behind it. "
		"QuteMol draws an enlarged billboard per atom; this is the screen-space "
		"equivalent over the depth buffer, which is the only version that scales here.");

	if(!CuRastSettings::haloEnabled) return;

	ImGui::SliderFloat("Size##halo", &CuRastSettings::haloSize, 0.001f, 0.08f, "%.4f");
	ImGui::SliderFloat("Strength##halo", &CuRastSettings::haloStrength, 0.0f, 1.0f);
	ImGui::SliderFloat("Colour##halo", &CuRastSettings::haloColor, 0.0f, 1.0f);
	ImGui::SetItemTooltip("0 = black (QuteMol default), 1 = white for a glow on dark backgrounds.");
	ImGui::SliderFloat("Depth for full halo", &CuRastSettings::haloDepthFull,
		0.001f, 0.5f, "%.4f", ImGuiSliderFlags_Logarithmic);
	ImGui::SetItemTooltip(
		"Depth gap that produces a fully opaque halo, as a fraction of pixel depth. "
		"Larger values restrict the halo to big silhouette jumps.");
	ImGui::SliderInt("Dirs##halo",  &CuRastSettings::haloDirs,  4, 32);
	ImGui::SliderInt("Steps##halo", &CuRastSettings::haloSteps, 1, 12);
	ImGui::Text("Halo taps / pixel: %d", CuRastSettings::haloDirs * CuRastSettings::haloSteps);
}

void makeMultiscaleSSAOControls(){
	if(!ImGui::CollapsingHeader("Ambient occlusion (screen space)")) return;

	ImGui::Checkbox("Enable SSAO", &CuRastSettings::enableSSAO);
	ImGui::SameLine();
	ImGui::Checkbox("Multiscale", &CuRastSettings::enableMultiscaleSSAO);

	if(!CuRastSettings::enableSSAO){
		ImGui::TextDisabled("(SSAO disabled - enable it above to tune)");
		return;
	}

	ImGui::Combo("AO algorithm", &CuRastSettings::aoMode,
		"Hemisphere (legacy)\0GTAO\0\0");
	ImGui::SetItemTooltip(
		"GTAO searches for the horizon in screen space and integrates the visibility "
		"analytically per slice, instead of sampling points in the hemisphere. Much "
		"lower noise per unit cost. It has ONE radius -- the close/far level structure "
		"below applies only to the legacy path.");

	ImGui::SliderFloat("AO floor", &CuRastSettings::aoFloor, 0.0f, 1.0f);
	ImGui::SetItemTooltip("How dark a fully occluded pixel is allowed to get.");
	ImGui::SliderFloat("AO power", &CuRastSettings::aoPower, 0.25f, 3.0f);
	ImGui::SetItemTooltip(
		"shade = floor + (1-floor) * ao^power. Above 1 deepens contact shadows "
		"without darkening open surfaces.");

	if(CuRastSettings::aoMode == 1){
		ImGui::SliderFloat("Radius (frac. of depth)", &CuRastSettings::gtaoRadius,
			0.002f, 0.5f, "%.4f", ImGuiSliderFlags_Logarithmic);
		ImGui::SetItemTooltip(
			"World radius = depth * this, so the screen footprint stays constant with "
			"distance. The large-scale shape cue lives here: small values only find the "
			"cavities between neighbouring atoms.");
		ImGui::SliderInt  ("Slices",    &CuRastSettings::gtaoSlices, 1, 8);
		ImGui::SliderInt  ("Steps/side",&CuRastSettings::gtaoSteps,  2, 24);
		ImGui::SliderFloat("Intensity##gtao", &CuRastSettings::gtaoIntensity, 0.0f, 3.0f);
		ImGui::SliderFloat("Thickness", &CuRastSettings::gtaoThickness, 0.25f, 3.0f);
		ImGui::Text("Depth taps / pixel: %d",
			CuRastSettings::gtaoSlices * CuRastSettings::gtaoSteps * 2);
		ImGui::Separator();
		ImGui::TextDisabled("Legacy hemisphere controls below are inactive.");
	}

	// Radius scale: the single most important knob — multiplies all per-level radii.
	// Use a logarithmic slider since useful values span 0.05× .. 20× across scenes.
	ImGui::SliderFloat("Radius scale (global)", &CuRastSettings::ssaoRadiusScale,
		0.05f, 20.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
	if(ImGui::IsItemHovered()) ImGui::SetTooltip(
		"Multiplier on every level's radius. Increase for big assemblies, decrease for tight cavities.\n"
		"Effective sample radius = ssaoLevelRadius[k] * scale * pixelDepth");
	ImGui::SliderFloat("Intensity",        &CuRastSettings::ssaoIntensity,       0.0f, 4.0f);

	// Total taps per pixel is the headline cost number, so show it next to the sliders.
	{
		int totalTaps = 0;
		int active = CuRastSettings::enableMultiscaleSSAO ? CuRastSettings::ssaoLevels : 1;
		for(int k = 0; k < active; k++) totalTaps += CuRastSettings::ssaoSamplesPerLevel[k];
		ImGui::Text("Depth taps / pixel: %d  (sum over %d level%s)",
			totalTaps, active, active == 1 ? "" : "s");
	}

	if(CuRastSettings::enableMultiscaleSSAO){
		ImGui::SliderInt("Levels", &CuRastSettings::ssaoLevels, 1, 4);
		for(int k = 0; k < CuRastSettings::ssaoLevels; k++){
			ImGui::PushID(k);
			const char* role = (k == 0) ? " (close)"
			                 : (k == CuRastSettings::ssaoLevels - 1) ? " (far)" : "";
			ImGui::SeparatorText(("Level " + std::to_string(k) + role).c_str());
			ImGui::SliderFloat("radius (frac of depth)",
				&CuRastSettings::ssaoLevelRadius[k], 0.0001f, 2.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
			ImGui::SliderFloat("bias",
				&CuRastSettings::ssaoLevelBias[k],   0.0f, 4.0f);
			ImGui::SliderInt("samples",
				&CuRastSettings::ssaoSamplesPerLevel[k], 4, 64);
			if(ImGui::IsItemHovered()) ImGui::SetTooltip(
				"Hemisphere samples for THIS level. The close level needs the most (below\n"
				"~12 it goes visibly noisy); a far level is low-frequency and stays smooth\n"
				"on far fewer taps because the bilateral blur cleans it up.");
			ImGui::PopID();
		}

		ImGui::SeparatorText("Presets");
		if(ImGui::Button("Close + far (2 level, cheapest)")){
			CuRastSettings::ssaoLevels = 2;
			CuRastSettings::ssaoLevelRadius[0] = 0.004f;   // close: inter-atom cavities
			CuRastSettings::ssaoLevelRadius[1] = 0.040f;   // far:   overall enclosure
			CuRastSettings::ssaoSamplesPerLevel[0] = 24;
			CuRastSettings::ssaoSamplesPerLevel[1] = 8;
			CuRastSettings::ssaoRadiusScale = 1.0f;
		}
		if(ImGui::IsItemHovered()) ImGui::SetTooltip(
			"32 depth taps per pixel instead of the 4-level default's 64.\n"
			"Two levels leave a gap at intermediate scale: if mid-size cavities look\n"
			"flat, widen the close radius rather than adding levels back.");
		ImGui::SameLine();
		if(ImGui::Button("Atomic (tight cavities)")){
			CuRastSettings::ssaoLevels = 4;
			CuRastSettings::ssaoLevelRadius[0] = 0.0005f;
			CuRastSettings::ssaoLevelRadius[1] = 0.0015f;
			CuRastSettings::ssaoLevelRadius[2] = 0.005f;
			CuRastSettings::ssaoLevelRadius[3] = 0.015f;
			CuRastSettings::ssaoRadiusScale = 1.0f;
		}
		ImGui::SameLine();
		if(ImGui::Button("Protein-scale")){
			CuRastSettings::ssaoLevels = 4;
			CuRastSettings::ssaoLevelRadius[0] = 0.0015f;
			CuRastSettings::ssaoLevelRadius[1] = 0.005f;
			CuRastSettings::ssaoLevelRadius[2] = 0.015f;
			CuRastSettings::ssaoLevelRadius[3] = 0.050f;
			CuRastSettings::ssaoRadiusScale = 1.0f;
		}
		ImGui::SameLine();
		if(ImGui::Button("Assembly-scale")){
			CuRastSettings::ssaoLevels = 4;
			CuRastSettings::ssaoLevelRadius[0] = 0.005f;
			CuRastSettings::ssaoLevelRadius[1] = 0.020f;
			CuRastSettings::ssaoLevelRadius[2] = 0.080f;
			CuRastSettings::ssaoLevelRadius[3] = 0.250f;
			CuRastSettings::ssaoRadiusScale = 1.0f;
		}
	} else {
		// Single-scale: still honor a single radius taken from level 0.
		ImGui::SliderFloat("radius (frac of depth)",
			&CuRastSettings::ssaoLevelRadius[0], 0.0001f, 2.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
	}
}

void makeBenchmarking(){

	auto editor = CuRast::instance;

	if(CuRastSettings::showBenchmarking){

		ImVec2 windowSize = {800, 600};
		ImGui::SetNextWindowPos({
			(VKRenderer::width - windowSize.x) / 2,
			(VKRenderer::height - windowSize.y) / 2, },
			ImGuiCond_Once);
		ImGui::SetNextWindowSize(windowSize, ImGuiCond_Once);

		static bool open = true;
		if(ImGui::Begin("Benchmarking", &open)){

			// ---- mmCIF replication benchmark ----
			if(!mmcif::loadedAll.empty()){
				auto loaded = mmcif::loadedAll[0];
				ImGui::SeparatorText("mmCIF Grid Replication");
				ImGui::Text("Source: %u atoms (1 copy)", loaded->numAtomsOriginal);
				ImGui::Text("Current: %u atoms (%d copies)", loaded->numAtoms, loaded->numCopies);
				vec3 cell = loaded->aabbMax - loaded->aabbMin;
				ImGui::Text("Bounding box (Å): %.1f x %.1f x %.1f", cell.x, cell.y, cell.z);

				static int copies = 1;
				if(loaded->numCopies != copies && copies < 1) copies = loaded->numCopies;

				auto fmtMB = [](uint64_t bytes){ return (double)bytes / (1024.0 * 1024.0); };
				auto fmtGB = [](uint64_t bytes){ return (double)bytes / (1024.0 * 1024.0 * 1024.0); };

				// Recompute the budget every frame — cheap (just a couple of CUDA queries).
				mmcif::ReplicationBudget budget = mmcif::computeBudget(loaded.get(), copies);

				// Hardware capacity readout
				ImGui::Separator();
				ImGui::Text("GPU:  %.2f / %.2f GB free        Host: %.2f / %.2f GB available",
					fmtGB(budget.gpuFree),  fmtGB(budget.gpuTotal),
					fmtGB(budget.hostAvail), fmtGB(budget.hostTotal));

				// Workstation-wide maxima for this loaded structure
				uint64_t maxAtomsByGPU = (mmcif::MMCIF_BYTES_PER_ATOM_GPU > 0)
					? (budget.gpuFree + budget.gpuReclaimable) / mmcif::MMCIF_BYTES_PER_ATOM_GPU
					: 0;
				ImGui::Text("Max atoms loadable (this workstation): %.1f M    "
					"Max copies of this structure: safe=%d  hard=%d",
					(double)maxAtomsByGPU / 1e6,
					budget.maxSafeCopies, budget.maxHardCopies);

				// Slider — cap at the hard budget so you simply can't drag past what fits.
				int sliderMax = budget.maxHardCopies;
				if(sliderMax < 1) sliderMax = 1;
				if(copies > sliderMax) copies = sliderMax;
				ImGui::SliderInt("Copies", &copies, 1, sliderMax);

				// Projected cost for the current slider value, color-coded.
				uint64_t gpuEffective = budget.gpuFree + budget.gpuReclaimable;
				double gpuFrac  = (gpuEffective > 0) ? (double)budget.gpuNeededAtN / (double)gpuEffective : 1.0;
				double hostFrac = (budget.hostAvail > 0) ? (double)budget.cpuPeakAtN  / (double)budget.hostAvail : 1.0;
				double worstFrac = std::max(gpuFrac, hostFrac);

				ImVec4 costColor;
				const char* costLabel;
				if(worstFrac < 0.70)      { costColor = ImVec4(0.5f, 1.0f, 0.5f, 1.0f); costLabel = "OK"; }
				else if(worstFrac < 0.90) { costColor = ImVec4(1.0f, 0.85f, 0.3f, 1.0f); costLabel = "TIGHT"; }
				else                       { costColor = ImVec4(1.0f, 0.4f, 0.4f, 1.0f); costLabel = "RISKY"; }

				ImGui::TextColored(costColor,
					"[%s] Will use:  GPU %.2f GB (%.0f%% of free)   Host peak %.1f MB (%.0f%% of avail)",
					costLabel,
					fmtGB(budget.gpuNeededAtN), gpuFrac * 100.0,
					fmtMB(budget.cpuPeakAtN),   hostFrac * 100.0);

				bool risky = (worstFrac >= 0.70);

				if(risky) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.25f, 0.25f, 1.0f));
				bool applyClicked = ImGui::Button("Apply##copies");
				if(risky) ImGui::PopStyleColor();

				if(applyClicked){
					if(risky) ImGui::OpenPopup("Confirm large allocation");
					else {
						mmcif::replicateGrid(loaded.get(), copies);
						Runtime::controls->target = { loaded->centroid.x, loaded->centroid.y, loaded->centroid.z };
						Runtime::controls->radius = loaded->radius * 2.0f;
					}
				}

				if(ImGui::BeginPopupModal("Confirm large allocation", nullptr,
						ImGuiWindowFlags_AlwaysAutoResize))
				{
					ImGui::TextColored(costColor,
						"This allocation is %s.", costLabel);
					ImGui::Separator();
					ImGui::Text("GPU bytes:   %.2f GB  (%.0f%% of %.2f GB available)",
						fmtGB(budget.gpuNeededAtN), gpuFrac * 100.0, fmtGB(gpuEffective));
					ImGui::Text("Host peak:   %.1f MB  (%.0f%% of %.2f GB available)",
						fmtMB(budget.cpuPeakAtN), hostFrac * 100.0, fmtGB(budget.hostAvail));
					ImGui::Text("Total atoms: %.1f M (%d copies)",
						(double)budget.totalAtomsAtN / 1e6, copies);
					ImGui::Separator();
					ImGui::Text("If unsure, click Cancel and reduce the slider.");
					if(ImGui::Button("Proceed", ImVec2(120, 0))){
						mmcif::replicateGrid(loaded.get(), copies);
						Runtime::controls->target = { loaded->centroid.x, loaded->centroid.y, loaded->centroid.z };
						Runtime::controls->radius = loaded->radius * 2.0f;
						ImGui::CloseCurrentPopup();
					}
					ImGui::SameLine();
					if(ImGui::Button("Cancel", ImVec2(120, 0))){
						ImGui::CloseCurrentPopup();
					}
					ImGui::EndPopup();
				}

				ImGui::Separator();

				// ----- Colour theme -------------------------------------------
				ImGui::TextUnformatted("Colour theme:");
				ImGui::SameLine();
				int themeIdx = (int)loaded->currentTheme;
				bool themeChanged = false;
				themeChanged |= ImGui::RadioButton("element##theme", &themeIdx, (int)mmcif::ColorTheme::ELEMENT);
				ImGui::SameLine();
				themeChanged |= ImGui::RadioButton("chain##theme",   &themeIdx, (int)mmcif::ColorTheme::CHAIN);
				ImGui::SameLine();
				themeChanged |= ImGui::RadioButton("entity##theme",  &themeIdx, (int)mmcif::ColorTheme::ENTITY);
				if(themeChanged){
					mmcif::applyColorTheme(loaded.get(), (mmcif::ColorTheme)themeIdx);
				}
				ImGui::SameLine();
				ImGui::TextDisabled("(%u chains, %u entities)", loaded->numChains, loaded->numEntities);

				ImGui::Separator();
			}

			makeSphereLodControls();
			makeCameraControls();
			makeEnvironmentControls();
			makeShadingControls();
			makeMultiscaleSSAOControls();
			makeBakedAOControls();
			makeHaloControls();

			string strMeasure;
			if(Benchmarking::measurementCountdown >= 0){
				strMeasure = format("Measure 60 frames ({:2})", Benchmarking::measurementCountdown);
			}else{
				strMeasure = "Measure 60 frames";
			}
			if(ImGui::Button(strMeasure.c_str())){
				Benchmarking::measurementCountdown = 60;
			}

			int i = 0;
			for(Benchmarking::Scenario& scenario : Benchmarking::scenarios){
				
				string strC = scenario.compress ? "c" : " ";
				string strJ = scenario.useJpegTextures ? "j" : " ";
				string strR = scenario.imageDivisionFactor > 1 ? "h" : " ";
				string strM = Benchmarking::isMeshoptimized(&scenario) ? "m" : " ";

				string label = format("load {:<40} {} {} {} {}##benchmark_scenario_{}", 
					scenario.label, strC, strJ, strR, strM, i
				);
				if(ImGui::Button(label.c_str())){
				// if(ImGui::Button(label.c_str(), ImVec2(400, 0))){
					Benchmarking::request_scenario = &scenario;
				}
				ImGui::SameLine();
				string strButtonCloseup = format("closeup##benchmark_scenario_{}", i);
				if(ImGui::Button(strButtonCloseup.c_str())){
					Runtime::controls->yaw    = scenario.view_closeup.yaw;
					Runtime::controls->pitch  = scenario.view_closeup.pitch;
					Runtime::controls->radius = scenario.view_closeup.radius;
					Runtime::controls->target = scenario.view_closeup.target;
					Benchmarking::active_view = scenario.view_closeup;
				}
				ImGui::SameLine();
				string strButtonOverview = format("overview##benchmark_scenario_{}", i);
				if(ImGui::Button(strButtonOverview.c_str())){
					Runtime::controls->yaw    = scenario.view_overview.yaw;
					Runtime::controls->pitch  = scenario.view_overview.pitch;
					Runtime::controls->radius = scenario.view_overview.radius;
					Runtime::controls->target = scenario.view_overview.target;
					Benchmarking::active_view = scenario.view_overview;
				}

				i++;
			}
		}

		ImGui::End();
	}

}