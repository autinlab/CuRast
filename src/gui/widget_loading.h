#pragma once

// ImGui progress bar shown while a mmCIF file is being loaded asynchronously.
// State lives in `mmcif::activeLoad` (see MMCIFLoader.h); main.cpp owns the
// lifecycle (start in drag-drop callback, finalise in update()).

#include "imgui.h"
#include "../mmcif/MMCIFLoader.h"

void CuRast::makeLoadingProgress(){

	auto progress = mmcif::activeLoad;
	if(!progress) return;

	int stage = progress->stage.load(std::memory_order_acquire);
	if(stage == mmcif::LoadingProgress::IDLE) return;
	if(stage == mmcif::LoadingProgress::DONE) return;     // finalised by update()
	if(stage == mmcif::LoadingProgress::FAILED) return;

	const char* stageLabel = "Loading...";
	float fraction = -1.0f; // <0 -> indeterminate
	std::string detail;

	size_t bytesRead     = progress->bytesRead.load(std::memory_order_relaxed);
	size_t totalBytes    = progress->totalBytes.load(std::memory_order_relaxed);
	size_t atomCount     = progress->atomCount.load(std::memory_order_relaxed);
	size_t expanded      = progress->expanded.load(std::memory_order_relaxed);
	size_t totalExpanded = progress->totalExpanded.load(std::memory_order_relaxed);
	int    upCopies      = progress->uploadCopies.load(std::memory_order_relaxed);
	int    upTotal       = progress->uploadCopiesTotal.load(std::memory_order_relaxed);

	auto fmtMB = [](size_t bytes) {
		double mb = double(bytes) / (1024.0 * 1024.0);
		return std::format("{:.1f} MB", mb);
	};

	switch(stage){
		case mmcif::LoadingProgress::READING:
			stageLabel = "Reading file";
			detail = totalBytes ? fmtMB(totalBytes) : std::string{};
			break;
		case mmcif::LoadingProgress::PARSING:
			stageLabel = "Parsing mmCIF";
			if(totalBytes > 0) fraction = float(double(bytesRead) / double(totalBytes));
			detail = std::format("{} / {}  ({} atoms)",
				fmtMB(bytesRead), fmtMB(totalBytes), atomCount);
			break;
		case mmcif::LoadingProgress::EXPANDING:
			stageLabel = "Expanding assembly";
			if(totalExpanded > 0) fraction = float(double(expanded) / double(totalExpanded));
			detail = std::format("{} / {} atoms", expanded, totalExpanded);
			break;
		case mmcif::LoadingProgress::UPLOADING:
			stageLabel = "Uploading to GPU";
			if(upTotal > 0) fraction = float(upCopies) / float(upTotal);
			break;
		default: break;
	}

	// Centre the window
	ImGuiIO& io = ImGui::GetIO();
	ImVec2 size{460.0f, 0.0f};
	ImVec2 pos{
		(io.DisplaySize.x - size.x) * 0.5f,
		(io.DisplaySize.y - 120.0f) * 0.5f
	};
	ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
	ImGui::SetNextWindowSize(size, ImGuiCond_Always);

	ImGuiWindowFlags flags =
		  ImGuiWindowFlags_NoCollapse
		| ImGuiWindowFlags_NoResize
		| ImGuiWindowFlags_NoMove
		| ImGuiWindowFlags_NoSavedSettings
		| ImGuiWindowFlags_AlwaysAutoResize;

	ImGui::Begin("Loading mmCIF", nullptr, flags);

	// File name (basename only, full path in tooltip)
	std::string path = progress->filepath;
	std::string base = path;
	auto slash = path.find_last_of("/\\");
	if(slash != std::string::npos) base = path.substr(slash + 1);
	ImGui::TextUnformatted(base.c_str());
	if(!path.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", path.c_str());

	ImGui::Separator();
	ImGui::TextUnformatted(stageLabel);

	if(fraction >= 0.0f){
		if(fraction > 1.0f) fraction = 1.0f;
		std::string overlay = std::format("{:.0f}%", fraction * 100.0f);
		ImGui::ProgressBar(fraction, ImVec2(-1.0f, 0.0f), overlay.c_str());
	}else{
		// Indeterminate: animated bar via fmod(time)
		float t = fmodf((float)ImGui::GetTime() * 0.6f, 1.0f);
		ImGui::ProgressBar(t, ImVec2(-1.0f, 0.0f), "...");
	}

	if(!detail.empty()) ImGui::TextUnformatted(detail.c_str());

	ImGui::End();
}
