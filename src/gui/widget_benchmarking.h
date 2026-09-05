
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

void makeMultiscaleSSAOControls(){
	if(!ImGui::CollapsingHeader("SSAO")) return;

	ImGui::Checkbox("Enable SSAO", &CuRastSettings::enableSSAO);
	ImGui::SameLine();
	ImGui::Checkbox("Multiscale", &CuRastSettings::enableMultiscaleSSAO);

	if(!CuRastSettings::enableSSAO){
		ImGui::TextDisabled("(SSAO disabled — enable it above to tune)");
		return;
	}

	// Radius scale: the single most important knob — multiplies all per-level radii.
	// Use a logarithmic slider since useful values span 0.05× .. 20× across scenes.
	ImGui::SliderFloat("Radius scale (global)", &CuRastSettings::ssaoRadiusScale,
		0.05f, 20.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
	if(ImGui::IsItemHovered()) ImGui::SetTooltip(
		"Multiplier on every level's radius. Increase for big assemblies, decrease for tight cavities.\n"
		"Effective sample radius = ssaoLevelRadius[k] * scale * pixelDepth");
	ImGui::SliderFloat("Intensity",        &CuRastSettings::ssaoIntensity,       0.0f, 4.0f);
	ImGui::SliderInt  ("Samples / level",  &CuRastSettings::ssaoSamplesPerLevel, 8,    64);
	if(ImGui::IsItemHovered()) ImGui::SetTooltip(
		"Hemisphere samples per level. Below ~12 the AO becomes visibly noisy and the\n"
		"bilateral blur cannot fully clean it up.");

	if(CuRastSettings::enableMultiscaleSSAO){
		ImGui::SliderInt("Levels", &CuRastSettings::ssaoLevels, 1, 4);
		for(int k = 0; k < CuRastSettings::ssaoLevels; k++){
			ImGui::PushID(k);
			ImGui::SeparatorText(("Level " + std::to_string(k)).c_str());
			ImGui::SliderFloat("radius (frac of depth)",
				&CuRastSettings::ssaoLevelRadius[k], 0.0001f, 2.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
			ImGui::SliderFloat("bias",
				&CuRastSettings::ssaoLevelBias[k],   0.0f, 4.0f);
			ImGui::PopID();
		}

		ImGui::SeparatorText("Presets");
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
			makeMultiscaleSSAOControls();

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