
#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <set>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <climits>
#include <memory>
#include <charconv>
#include <atomic>
#include <thread>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "cuda.h"
#include "../scene/SNSpheres.h"
#include "../MemoryManager.h"
#include "../CURuntime.h"

using namespace std;
using glm::vec3;
using glm::vec4;
using glm::mat4;

namespace mmcif {

// ---- CPK colors (RGBA8: byte[0]=R, byte[1]=G, byte[2]=B, byte[3]=A = 0xFF__BBGGRR) ----
struct ElemInfo {
	uint32_t color;
	float    radius;
	uint8_t  typeIdx;   // index into the element palette LUT
};

// Stable element table — index = type-byte uploaded to the GPU. Slot 255 is
// reserved for "unknown element" (default grey, 1.50 Å).
struct ElemEntry { const char* name; uint32_t color; float radius; };
static const ElemEntry ELEMENT_TABLE[] = {
	{"H",  0xFFFFFFFF, 1.20f},   // 0
	{"C",  0xFF909090, 1.70f},   // 1
	{"N",  0xFFF85030, 1.55f},   // 2
	{"O",  0xFF0D0DFF, 1.52f},   // 3
	{"S",  0xFF30FFFF, 1.80f},   // 4
	{"P",  0xFF0080FF, 1.80f},   // 5
	{"FE", 0xFF3366E0, 1.40f},   // 6
	{"MG", 0xFF22228B, 1.73f},   // 7
	{"ZN", 0xFFB8807C, 1.22f},   // 8
	{"CA", 0xFF8C8C8C, 1.97f},   // 9
	{"CL", 0xFF1FF01F, 1.75f},   // 10
	{"NA", 0xFFAB4040, 2.27f},   // 11
	{"K",  0xFF6B2E8F, 2.75f},   // 12
	{"MN", 0xFFB5B5B5, 1.61f},   // 13
	{"CU", 0xFF137BC2, 1.40f},   // 14
	{"NI", 0xFF49D147, 1.63f},   // 15
};
static constexpr uint8_t ELEMENT_UNKNOWN_IDX = 255;
static constexpr uint32_t ELEMENT_UNKNOWN_COLOR = 0xFF808080u;
static constexpr float ELEMENT_UNKNOWN_RADIUS = 1.50f;

static ElemInfo getElemInfo(string_view element) {
	char buf[8] = {0};
	size_t n = std::min(element.size(), sizeof(buf) - 1);
	for(size_t i = 0; i < n; i++) buf[i] = (char)toupper((unsigned char)element[i]);
	string_view key(buf, n);
	for(uint8_t i = 0; i < (uint8_t)(sizeof(ELEMENT_TABLE)/sizeof(ELEMENT_TABLE[0])); i++){
		if(key == ELEMENT_TABLE[i].name){
			return {ELEMENT_TABLE[i].color, ELEMENT_TABLE[i].radius, i};
		}
	}
	return {ELEMENT_UNKNOWN_COLOR, ELEMENT_UNKNOWN_RADIUS, ELEMENT_UNKNOWN_IDX};
}

// Build the 256-entry RGBA8 palette uploaded to cptr_palette. Unknown slots
// fall back to the default grey so a stray atom type can never index garbage.
static void buildElementPalette(uint32_t out[256]){
	for(int i = 0; i < 256; i++) out[i] = ELEMENT_UNKNOWN_COLOR;
	for(uint8_t i = 0; i < (uint8_t)(sizeof(ELEMENT_TABLE)/sizeof(ELEMENT_TABLE[0])); i++){
		out[i] = ELEMENT_TABLE[i].color;
	}
}

// ---- Streaming tokenizer (no materialized token vector) ----
struct TokenStream {
	const char* data = nullptr;
	size_t pos = 0;
	size_t size = 0;

	bool has_pushback = false;
	string_view pushback_tok;

	TokenStream(const string& content) : data(content.data()), size(content.size()) {}

	bool atEnd() const { return !has_pushback && pos >= size; }

	void pushback(string_view tok) {
		pushback_tok = tok;
		has_pushback = true;
	}

	string_view next() {
		if(has_pushback) {
			has_pushback = false;
			return pushback_tok;
		}

		// skip whitespace and # comments
		while(pos < size) {
			char c = data[pos];
			if(c == ' ' || c == '\t' || c == '\r' || c == '\n') { pos++; continue; }
			if(c == '#') {
				while(pos < size && data[pos] != '\n') pos++;
				continue;
			}
			break;
		}
		if(pos >= size) return {};

		char c = data[pos];

		// semicolon block (must start at column 0)
		if(c == ';' && (pos == 0 || data[pos-1] == '\n')) {
			pos++;
			if(pos < size && data[pos] == '\n') pos++;
			size_t start = pos;
			while(pos < size) {
				if(data[pos] == '\n' && pos+1 < size && data[pos+1] == ';') {
					string_view tok(data + start, pos - start);
					pos += 2;
					return tok;
				}
				pos++;
			}
			return string_view(data + start, pos - start);
		}

		// single-quoted
		if(c == '\'') {
			pos++;
			size_t start = pos;
			while(pos < size) {
				if(data[pos] == '\'' && (pos+1 >= size || data[pos+1] == ' ' || data[pos+1] == '\t' || data[pos+1] == '\n' || data[pos+1] == '\r')) break;
				pos++;
			}
			string_view tok(data + start, pos - start);
			if(pos < size) pos++;
			return tok;
		}

		// double-quoted
		if(c == '"') {
			pos++;
			size_t start = pos;
			while(pos < size && data[pos] != '"') pos++;
			string_view tok(data + start, pos - start);
			if(pos < size) pos++;
			return tok;
		}

		// regular token
		size_t start = pos;
		while(pos < size && data[pos] != ' ' && data[pos] != '\t' && data[pos] != '\r' && data[pos] != '\n') pos++;
		return string_view(data + start, pos - start);
	}
};

static bool isLoopKeyword(string_view t) {
	// Empty token is NOT a keyword — it's a legitimate empty quoted string ('').
	if(t.empty()) return false;
	if(t == "loop_" || t == "stop_" || t == "save_") return true;
	if(t[0] == '_') return true;
	if(t.size() >= 5 && t.substr(0,5) == "data_") return true;
	return false;
}

// Fast numeric parsing from string_view (avoids string copy)
static float sv_to_float(string_view s, float fallback = 0.0f) {
	if(s.empty() || s == "." || s == "?") return fallback;
	float v = fallback;
	auto res = std::from_chars(s.data(), s.data() + s.size(), v);
	if(res.ec != std::errc()) {
		// fallback to stof for tricky formats (e.g. with "+" or whitespace)
		try { return std::stof(string(s)); } catch(...) { return fallback; }
	}
	return v;
}

static int sv_to_int(string_view s, int fallback = 0) {
	if(s.empty() || s == "." || s == "?") return fallback;
	int v = fallback;
	auto res = std::from_chars(s.data(), s.data() + s.size(), v);
	if(res.ec != std::errc()) return fallback;
	return v;
}

// ---- Parse operation expression like "(1-10)" or "(1,3,5)" or "(1-3)(11-15)" ----
static vector<vector<string>> parseOperExpression(string_view expr) {
	vector<vector<string>> groups;
	size_t i = 0, n = expr.size();
	while(i < n) {
		if(expr[i] == '(') {
			i++;
			size_t start = i;
			while(i < n && expr[i] != ')') i++;
			string_view group_str = expr.substr(start, i - start);
			if(i < n) i++;

			vector<string> ids;
			size_t p = 0;
			while(p < group_str.size()) {
				size_t comma = group_str.find(',', p);
				string_view part = (comma == string_view::npos)
					? group_str.substr(p)
					: group_str.substr(p, comma - p);
				size_t dash = part.find('-');
				if(dash != string_view::npos) {
					int a = sv_to_int(part.substr(0, dash));
					int b = sv_to_int(part.substr(dash + 1));
					for(int k = a; k <= b; k++) ids.push_back(to_string(k));
				} else {
					ids.push_back(string(part));
				}
				if(comma == string_view::npos) break;
				p = comma + 1;
			}
			groups.push_back(std::move(ids));
		} else {
			i++;
		}
	}
	return groups;
}

// ---- Loaded result ----
// Color theme used to fill the per-atom RGBA8 colour buffer.
enum class ColorTheme : int {
	ELEMENT = 0,   // CPK colours by element type (default)
	CHAIN   = 1,   // each label_asym_id gets a distinct hue
	ENTITY  = 2,   // each label_entity_id gets a distinct hue
};

struct LoadedMmcif {
	shared_ptr<SNSpheres> node;
	uint32_t numAtoms = 0;            // current count (after replication)
	uint32_t numAtomsOriginal = 0;    // count of single instance
	vec3 centroid  = {0,0,0};         // current
	float radius   = 10.0f;           // current

	// Host-side original arrays — kept so we can rebuild the GPU buffers
	// for grid replication benchmarks.
	vector<vec3>     hostPositions;
	vector<float>    hostRadii;
	vector<uint8_t>  hostAtomTypes;       // per-atom palette index (CPK theme)
	vector<uint32_t> hostColors;          // current colour theme baked here (CHAIN/ENTITY only)
	vector<uint32_t> hostChainIds;        // per-atom chain id (interned label_asym_id)
	vector<uint32_t> hostEntityIds;       // per-atom entity id (label_entity_id, 0 if absent)
	uint32_t numChains = 0;
	uint32_t numEntities = 0;

	vec3 aabbMin = {0,0,0};
	vec3 aabbMax = {0,0,0};
	int  numCopies = 1;
	ColorTheme currentTheme = ColorTheme::ELEMENT;
};

// Global registry — accessible from the GUI for benchmarking.
inline std::vector<std::shared_ptr<LoadedMmcif>> loadedAll;

// ---- Async loading progress ----
// Updated by the worker thread inside `load()`; read by the main thread to
// drive a progress bar and to integrate the result into the scene.
struct LoadingProgress {
	enum Stage : int {
		IDLE      = 0,
		READING   = 1,
		PARSING   = 2,
		EXPANDING = 3,
		UPLOADING = 4,
		DONE      = 5,
		FAILED    = -1,
	};

	std::atomic<int>    stage{IDLE};
	std::atomic<size_t> bytesRead{0};
	std::atomic<size_t> totalBytes{0};
	std::atomic<size_t> atomCount{0};
	std::atomic<size_t> expanded{0};
	std::atomic<size_t> totalExpanded{0};
	std::atomic<int>    uploadCopies{0};
	std::atomic<int>    uploadCopiesTotal{0};

	std::string filepath;                    // set once before the thread starts
	std::shared_ptr<LoadedMmcif> result;     // populated by worker before stage=DONE
};

// Currently-running load (if any). The worker writes to it; the main thread
// reads atomics to render a progress bar and finalises scene integration when
// stage==DONE.
inline std::shared_ptr<LoadingProgress> activeLoad;

// ---- Main load function ----
static shared_ptr<LoadedMmcif> load(const string& filepath, CUcontext /*ctx*/, LoadingProgress* progress = nullptr) {

	// Read file
	if(progress) progress->stage.store(LoadingProgress::READING, std::memory_order_relaxed);

	ifstream file(filepath, ios::binary | ios::ate);
	if(!file.is_open()){
		println("MMCIFLoader: failed to open '{}'", filepath);
		return nullptr;
	}
	size_t fileSize = file.tellg();
	file.seekg(0);
	string content(fileSize, '\0');
	file.read(content.data(), fileSize);
	file.close();

	if(progress){
		progress->totalBytes.store(fileSize, std::memory_order_relaxed);
		progress->bytesRead.store(0, std::memory_order_relaxed);
		progress->stage.store(LoadingProgress::PARSING, std::memory_order_relaxed);
	}

	println("MMCIFLoader: parsing '{}' ({} MB)", filepath, fileSize / (1024*1024));

	TokenStream stream(content);

	// ---- Atoms stored compactly (no tokens kept around) ----
	// To minimise memory we store positions + element-look-up index in parallel arrays.
	// For a 4M-atom file: ~16B/atom = 64MB.
	struct AtomCompact {
		float x, y, z;
		uint32_t color;
		float radius;
		uint32_t chain_id;   // index into chain table
		uint32_t entity_id;  // interned _atom_site.label_entity_id (0 if absent)
		uint8_t  type_idx;   // index into ELEMENT_TABLE / palette
	};
	vector<AtomCompact> atoms;
	atoms.reserve(1 << 20); // 1M, will grow as needed

	// Chain ID interning: map asym_id string -> uint32_t index
	unordered_map<string, uint32_t> chainTable;
	auto internChain = [&](string_view sv) -> uint32_t {
		string s(sv);
		auto it = chainTable.find(s);
		if(it != chainTable.end()) return it->second;
		uint32_t idx = (uint32_t)chainTable.size();
		chainTable.emplace(std::move(s), idx);
		return idx;
	};

	// Entity ID interning: map label_entity_id string -> uint32_t index. CIF entity
	// IDs are usually small integers as strings ("1", "2", ...) but we intern any string.
	unordered_map<string, uint32_t> entityTable;
	auto internEntity = [&](string_view sv) -> uint32_t {
		string s(sv);
		auto it = entityTable.find(s);
		if(it != entityTable.end()) return it->second;
		uint32_t idx = (uint32_t)entityTable.size();
		entityTable.emplace(std::move(s), idx);
		return idx;
	};

	struct Oper { mat4 matrix; };
	unordered_map<string, Oper> operations;

	struct AssemblyGen {
		string assembly_id;
		string oper_expression;
		vector<uint32_t> chain_ids;  // interned
	};
	vector<AssemblyGen> assemblyGens;

	auto progressTick = [&](size_t lastReport) -> size_t {
		if(progress){
			progress->bytesRead.store(stream.pos, std::memory_order_relaxed);
			progress->atomCount.store(atoms.size(), std::memory_order_relaxed);
		}
		size_t mb = stream.pos / (1024*1024);
		if(mb >= lastReport + 50) {
			println("  ... {}/{} MB parsed, {} atoms so far", mb, fileSize/(1024*1024), atoms.size());
			return mb;
		}
		return lastReport;
	};
	size_t lastProgress = 0;

	while(!stream.atEnd()) {
		string_view tok = stream.next();
		if(stream.atEnd() && tok.empty()) break;

		if(tok != "loop_") continue;

		// Read column names. The first non-`_` token is the first data row's first token.
		vector<string> cols;
		string_view firstData;
		bool hasFirstData = false;

		while(!stream.atEnd()) {
			string_view t = stream.next();
			if(!t.empty() && t[0] == '_') {
				cols.push_back(string(t));
			} else {
				firstData = t;
				hasFirstData = true;
				break;
			}
		}
		if(cols.empty()) {
			if(hasFirstData) stream.pushback(firstData);
			continue;
		}

		// Determine prefix
		string prefix;
		auto dot = cols[0].find('.');
		if(dot != string::npos) prefix = cols[0].substr(0, dot);

		int numCols = (int)cols.size();
		auto colIdx = [&](const string& name) -> int {
			for(int c = 0; c < numCols; c++)
				if(cols[c] == name) return c;
			return -1;
		};

		// Read one row of `numCols` tokens. `row` is reused across iterations.
		auto readRow = [&](vector<string_view>& row) -> bool {
			row.resize(numCols);
			string_view first;
			if(hasFirstData) { first = firstData; hasFirstData = false; }
			else {
				if(stream.atEnd()) return false;
				first = stream.next();
				if(stream.atEnd() && first.empty()) return false; // real EOF
				if(isLoopKeyword(first)) {
					stream.pushback(first);
					return false;
				}
			}
			row[0] = first;
			for(int c = 1; c < numCols; c++) {
				if(stream.atEnd()) return false;
				row[c] = stream.next();
			}
			return true;
		};

		vector<string_view> row;

		if(prefix == "_atom_site") {
			int ix = colIdx("_atom_site.Cartn_x");
			int iy = colIdx("_atom_site.Cartn_y");
			int iz = colIdx("_atom_site.Cartn_z");
			int ie = colIdx("_atom_site.type_symbol");
			int ia = colIdx("_atom_site.label_asym_id");
			int io = colIdx("_atom_site.occupancy");
			int im = colIdx("_atom_site.pdbx_PDB_model_num");
			int ialt = colIdx("_atom_site.label_alt_id");
			int ient = colIdx("_atom_site.label_entity_id");

			while(readRow(row)) {
				int model = (im >= 0) ? sv_to_int(row[im], 1) : 1;
				if(model != 1) continue;

				float occ = (io >= 0) ? sv_to_float(row[io], 1.0f) : 1.0f;
				if(occ <= 0.0f) continue;

				if(ialt >= 0) {
					string_view alt = row[ialt];
					if(!(alt == "." || alt == "?" || alt == "A" || alt == "1")) continue;
				}

				if(ix < 0 || iy < 0 || iz < 0) continue;
				string_view xs = row[ix], ys = row[iy], zs = row[iz];
				if(xs == "." || ys == "." || zs == ".") continue;

				AtomCompact a;
				a.x = sv_to_float(xs);
				a.y = sv_to_float(ys);
				a.z = sv_to_float(zs);

				string_view elem = (ie >= 0) ? row[ie] : string_view("C");
				ElemInfo info = getElemInfo(elem);
				a.color    = info.color;
				a.radius   = info.radius;
				a.type_idx = info.typeIdx;

				string_view asym = (ia >= 0) ? row[ia] : string_view("A");
				a.chain_id = internChain(asym);

				if(ient >= 0){
					string_view ent = row[ient];
					if(ent == "." || ent == "?" || ent.empty()) ent = string_view("0");
					a.entity_id = internEntity(ent);
				} else {
					a.entity_id = 0;
				}

				atoms.push_back(a);

				lastProgress = progressTick(lastProgress);
			}

		} else if(prefix == "_pdbx_struct_oper_list") {
			int iid  = colIdx("_pdbx_struct_oper_list.id");
			int im11 = colIdx("_pdbx_struct_oper_list.matrix[1][1]");
			int im12 = colIdx("_pdbx_struct_oper_list.matrix[1][2]");
			int im13 = colIdx("_pdbx_struct_oper_list.matrix[1][3]");
			int im21 = colIdx("_pdbx_struct_oper_list.matrix[2][1]");
			int im22 = colIdx("_pdbx_struct_oper_list.matrix[2][2]");
			int im23 = colIdx("_pdbx_struct_oper_list.matrix[2][3]");
			int im31 = colIdx("_pdbx_struct_oper_list.matrix[3][1]");
			int im32 = colIdx("_pdbx_struct_oper_list.matrix[3][2]");
			int im33 = colIdx("_pdbx_struct_oper_list.matrix[3][3]");
			int iv1  = colIdx("_pdbx_struct_oper_list.vector[1]");
			int iv2  = colIdx("_pdbx_struct_oper_list.vector[2]");
			int iv3  = colIdx("_pdbx_struct_oper_list.vector[3]");

			while(readRow(row)) {
				string id = (iid >= 0) ? string(row[iid]) : ".";

				float r11 = (im11 >= 0) ? sv_to_float(row[im11]) : 1.0f;
				float r12 = (im12 >= 0) ? sv_to_float(row[im12]) : 0.0f;
				float r13 = (im13 >= 0) ? sv_to_float(row[im13]) : 0.0f;
				float r21 = (im21 >= 0) ? sv_to_float(row[im21]) : 0.0f;
				float r22 = (im22 >= 0) ? sv_to_float(row[im22]) : 1.0f;
				float r23 = (im23 >= 0) ? sv_to_float(row[im23]) : 0.0f;
				float r31 = (im31 >= 0) ? sv_to_float(row[im31]) : 0.0f;
				float r32 = (im32 >= 0) ? sv_to_float(row[im32]) : 0.0f;
				float r33 = (im33 >= 0) ? sv_to_float(row[im33]) : 1.0f;
				float t1  = (iv1  >= 0) ? sv_to_float(row[iv1])  : 0.0f;
				float t2  = (iv2  >= 0) ? sv_to_float(row[iv2])  : 0.0f;
				float t3  = (iv3  >= 0) ? sv_to_float(row[iv3])  : 0.0f;

				Oper op;
				op.matrix = mat4(
					r11, r21, r31, 0.0f,
					r12, r22, r32, 0.0f,
					r13, r23, r33, 0.0f,
					t1,  t2,  t3,  1.0f
				);
				operations.emplace(std::move(id), op);
			}

		} else if(prefix == "_pdbx_struct_assembly_gen") {
			int iaid = colIdx("_pdbx_struct_assembly_gen.assembly_id");
			int iop  = colIdx("_pdbx_struct_assembly_gen.oper_expression");
			int ias  = colIdx("_pdbx_struct_assembly_gen.asym_id_list");

			while(readRow(row)) {
				AssemblyGen ag;
				ag.assembly_id     = (iaid >= 0) ? string(row[iaid]) : "1";
				ag.oper_expression = (iop  >= 0) ? string(row[iop])  : "";
				if(ias >= 0) {
					string_view list = row[ias];
					size_t p = 0;
					while(p < list.size()) {
						size_t comma = list.find(',', p);
						string_view chain = (comma == string_view::npos)
							? list.substr(p)
							: list.substr(p, comma - p);
						// trim spaces
						while(!chain.empty() && (chain.front() == ' ' || chain.front() == '\t')) chain.remove_prefix(1);
						while(!chain.empty() && (chain.back()  == ' ' || chain.back()  == '\t')) chain.remove_suffix(1);
						if(!chain.empty()) {
							// Only add chains we've seen — saves memory on noise
							auto it = chainTable.find(string(chain));
							if(it != chainTable.end()) ag.chain_ids.push_back(it->second);
						}
						if(comma == string_view::npos) break;
						p = comma + 1;
					}
				}
				assemblyGens.push_back(std::move(ag));
			}

		} else {
			// Skip unknown loop data — consume tokens until next keyword.
			while(!stream.atEnd()) {
				string_view t = stream.next();
				if(stream.atEnd() && t.empty()) break; // real EOF
				if(isLoopKeyword(t)) {
					stream.pushback(t);
					break;
				}
			}
			if(hasFirstData) hasFirstData = false;
		}
	}

	println("MMCIFLoader: {} atoms in asymmetric unit ({} chains)", atoms.size(), chainTable.size());
	println("MMCIFLoader: {} symmetry operations, {} assembly_gen entries", operations.size(), assemblyGens.size());

	if(atoms.empty()) {
		println("MMCIFLoader: no atoms found in '{}'", filepath);
		return nullptr;
	}

	// ---- Build per-chain atom index (one pass over atoms) ----
	// chain_id -> sorted list of atom indices
	vector<vector<uint32_t>> chainAtoms(chainTable.size());
	for(uint32_t i = 0; i < atoms.size(); i++) {
		uint32_t c = atoms[i].chain_id;
		if(c < chainAtoms.size()) chainAtoms[c].push_back(i);
	}

	// ---- Compute final operations to apply ----
	vector<pair<mat4, vector<uint32_t>>> finalOps; // (transform, chain ids)
	if(assemblyGens.empty() || operations.empty()) {
		// No assembly info — use all atoms with identity transform
		vector<uint32_t> allChains;
		for(auto& [name, idx] : chainTable) allChains.push_back(idx);
		finalOps.push_back({mat4(1.0f), std::move(allChains)});
	} else {
		// Filter to assembly_id == "1"
		size_t skipped = 0;
		for(auto& ag : assemblyGens) {
			if(ag.assembly_id != "1") { skipped++; continue; }
			auto groups = parseOperExpression(ag.oper_expression);
			if(groups.empty()) {
				finalOps.push_back({mat4(1.0f), ag.chain_ids});
			} else if(groups.size() == 1) {
				for(auto& opId : groups[0]) {
					auto it = operations.find(opId);
					if(it != operations.end()) {
						finalOps.push_back({it->second.matrix, ag.chain_ids});
					}
				}
			} else {
				// Cartesian product
				for(auto& id0 : groups[0]) {
					auto it0 = operations.find(id0);
					if(it0 == operations.end()) continue;
					for(size_t g = 1; g < groups.size(); g++) {
						for(auto& idg : groups[g]) {
							auto itg = operations.find(idg);
							if(itg == operations.end()) continue;
							mat4 combined = itg->second.matrix * it0->second.matrix;
							finalOps.push_back({combined, ag.chain_ids});
						}
					}
				}
			}
		}
		if(skipped > 0) println("MMCIFLoader: skipped {} non-assembly-1 entries", skipped);
	}

	// ---- Count total expanded atoms (sanity check before allocation) ----
	uint64_t totalAtoms = 0;
	for(auto& [M, chains] : finalOps) {
		for(uint32_t cid : chains) {
			if(cid < chainAtoms.size()) totalAtoms += chainAtoms[cid].size();
		}
	}
	println("MMCIFLoader: {} assembly operations -> {} total atoms after expansion", finalOps.size(), totalAtoms);

	if(totalAtoms == 0) {
		println("MMCIFLoader: nothing to render after expansion");
		return nullptr;
	}

	// Sphere index packs into 32 bits in the visbuffer (idx+1). Cap to ~4G.
	if(totalAtoms > 0xFFFFFFFEull) {
		println("MMCIFLoader: WARNING totalAtoms exceeds 32-bit limit, truncating");
		totalAtoms = 0xFFFFFFFEull;
	}

	// ---- Expand atoms into final arrays ----
	println("MMCIFLoader: allocating {} MB for expanded positions/radii/colors",
		(totalAtoms * (sizeof(vec3) + sizeof(float) + sizeof(uint32_t))) / (1024*1024));

	if(progress){
		progress->totalExpanded.store(totalAtoms, std::memory_order_relaxed);
		progress->expanded.store(0, std::memory_order_relaxed);
		progress->stage.store(LoadingProgress::EXPANDING, std::memory_order_relaxed);
	}

	vector<vec3>     positions(totalAtoms);
	vector<float>    radii(totalAtoms);
	vector<uint8_t>  atomTypes(totalAtoms);
	// Per-atom metadata kept post-expansion so we can re-bake colours when the
	// theme switches (element / chain / entity) without re-parsing the file.
	vector<uint32_t> chainIds(totalAtoms);
	vector<uint32_t> entityIds(totalAtoms);

	uint64_t out = 0;
	for(auto& [M, chains] : finalOps) {
		for(uint32_t cid : chains) {
			if(cid >= chainAtoms.size()) continue;
			for(uint32_t ai : chainAtoms[cid]) {
				if(out >= totalAtoms) break;
				const AtomCompact& a = atoms[ai];
				vec4 p_world = M * vec4(a.x, a.y, a.z, 1.0f);
				positions[out] = vec3(p_world);
				radii[out]     = a.radius;
				atomTypes[out] = a.type_idx;
				chainIds[out]  = a.chain_id;
				entityIds[out] = a.entity_id;
				out++;
			}
		}
		if(progress) progress->expanded.store(out, std::memory_order_relaxed);
	}
	totalAtoms = out;

	// Free temporary state
	atoms.clear(); atoms.shrink_to_fit();
	chainAtoms.clear(); chainAtoms.shrink_to_fit();
	operations.clear();
	assemblyGens.clear(); assemblyGens.shrink_to_fit();

	// ---- Upload to GPU ----
	if(progress){
		progress->uploadCopiesTotal.store(1, std::memory_order_relaxed);
		progress->uploadCopies.store(0, std::memory_order_relaxed);
		progress->stage.store(LoadingProgress::UPLOADING, std::memory_order_relaxed);
	}

	uint64_t bytes_pos     = totalAtoms * sizeof(vec3);
	uint64_t bytes_radii   = totalAtoms * sizeof(float);
	uint64_t bytes_atoms   = totalAtoms * sizeof(uint8_t);
	uint64_t bytes_palette = 256ull * sizeof(uint32_t);

	CUdeviceptr cptr_pos     = MemoryManager::alloc(bytes_pos,     "mmcif_positions");
	CUdeviceptr cptr_radii   = MemoryManager::alloc(bytes_radii,   "mmcif_radii");
	CUdeviceptr cptr_atoms   = MemoryManager::alloc(bytes_atoms,   "mmcif_atomTypes");
	CUdeviceptr cptr_palette = MemoryManager::alloc(bytes_palette, "mmcif_palette");

	uint32_t hostPalette[256];
	buildElementPalette(hostPalette);

	cuMemcpyHtoD(cptr_pos,     positions.data(), bytes_pos);
	cuMemcpyHtoD(cptr_radii,   radii.data(),     bytes_radii);
	cuMemcpyHtoD(cptr_atoms,   atomTypes.data(), bytes_atoms);
	cuMemcpyHtoD(cptr_palette, hostPalette,      bytes_palette);

	if(progress) progress->uploadCopies.store(1, std::memory_order_relaxed);

	// ---- Build SNSpheres node ----
	auto sn = make_shared<SNSpheres>(filepath);
	sn->cptr_positions = cptr_pos;
	sn->cptr_radii     = cptr_radii;
	sn->cptr_colors    = 0;             // ELEMENT theme: rendered from atomTypes + palette
	sn->cptr_atomTypes = cptr_atoms;
	sn->cptr_palette   = cptr_palette;
	sn->numSpheres     = (uint32_t)totalAtoms;

	// Compute centroid, AABB and bounding radius
	vec3 centroid = {0.0f, 0.0f, 0.0f};
	vec3 aabbMin = positions[0], aabbMax = positions[0];
	for(uint64_t i = 0; i < totalAtoms; i++) {
		centroid += positions[i];
		aabbMin = glm::min(aabbMin, positions[i]);
		aabbMax = glm::max(aabbMax, positions[i]);
	}
	if(totalAtoms > 0) centroid /= float(totalAtoms);

	float maxDist = 0.0f;
	for(uint64_t i = 0; i < totalAtoms; i++) {
		vec3 d = positions[i] - centroid;
		float dist = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);
		if(dist > maxDist) maxDist = dist;
	}

	auto result = make_shared<LoadedMmcif>();
	result->node             = sn;
	result->numAtoms         = (uint32_t)totalAtoms;
	result->numAtomsOriginal = (uint32_t)totalAtoms;
	result->centroid         = centroid;
	result->radius           = maxDist > 0.0f ? maxDist : 10.0f;
	result->aabbMin          = aabbMin;
	result->aabbMax          = aabbMax;

	// Keep host arrays alive so we can rebuild GPU buffers for replication.
	result->hostPositions     = std::move(positions);
	result->hostRadii         = std::move(radii);
	result->hostAtomTypes     = std::move(atomTypes);
	result->hostChainIds      = std::move(chainIds);
	result->hostEntityIds     = std::move(entityIds);
	result->numChains         = (uint32_t)chainTable.size();
	result->numEntities       = (uint32_t)entityTable.size();
	result->currentTheme      = ColorTheme::ELEMENT;

	println("MMCIFLoader: done. {} atoms, centroid=({:.2f},{:.2f},{:.2f}), radius={:.2f}, aabb=({:.1f},{:.1f},{:.1f})x({:.1f},{:.1f},{:.1f})",
		totalAtoms, centroid.x, centroid.y, centroid.z, result->radius,
		aabbMin.x, aabbMin.y, aabbMin.z, aabbMax.x, aabbMax.y, aabbMax.z);
	return result;
}

// Hash a small integer ID into an RGBA8 colour. Pleasant golden-ratio hue spacing
// + a fixed saturation/value gives high contrast between adjacent IDs without
// needing a fixed palette.
static inline uint32_t idToColor(uint32_t id){
	if(id == 0xFFFFFFFFu) return 0xFF808080u;
	float h = fmodf(float(id) * 0.61803398875f, 1.0f); // golden-ratio hue rotation
	float s = 0.65f;
	float v = 0.95f;
	float c = v * s;
	float hp = h * 6.0f;
	float x = c * (1.0f - fabsf(fmodf(hp, 2.0f) - 1.0f));
	float r = 0, g = 0, b = 0;
	if      (hp < 1) { r = c; g = x; }
	else if (hp < 2) { r = x; g = c; }
	else if (hp < 3) { g = c; b = x; }
	else if (hp < 4) { g = x; b = c; }
	else if (hp < 5) { r = x; b = c; }
	else             { r = c; b = x; }
	float m = v - c;
	uint8_t R = (uint8_t)((r + m) * 255.0f);
	uint8_t G = (uint8_t)((g + m) * 255.0f);
	uint8_t B = (uint8_t)((b + m) * 255.0f);
	// RGBA8 packed as 0xAABBGGRR (matches existing CPK colour format above)
	return 0xFF000000u | (uint32_t)B << 16 | (uint32_t)G << 8 | (uint32_t)R;
}

// Switch to a different colour theme.
//
// ELEMENT theme is the cheap path: each atom's GPU footprint is a 1-byte
// palette index plus a shared 256-entry LUT, so a theme switch is just a
// pointer flip on the rasterizer side — no per-atom upload at all. We free
// the legacy uint32 colour buffer here to reclaim the (potentially large)
// memory it held while CHAIN/ENTITY was active.
//
// CHAIN/ENTITY themes don't fit in 8 bits (chain counts can hit 1000+), so
// they fall back to the legacy per-atom uint32 colour buffer. We allocate it
// lazily on first switch and re-bake from the cached chain/entity IDs each
// time. The atomTypes + palette buffers stay alive in case the user switches
// back to ELEMENT.
static bool applyColorTheme(LoadedMmcif* loaded, ColorTheme theme){
	if(!loaded || !loaded->node) return false;
	uint64_t n = loaded->numAtoms;
	if(n == 0) return false;

	if(theme == ColorTheme::ELEMENT){
		if(loaded->node->cptr_colors){
			MemoryManager::free(loaded->node->cptr_colors);
			loaded->node->cptr_colors = 0;
		}
		loaded->hostColors.clear();
		loaded->hostColors.shrink_to_fit();
		loaded->currentTheme = theme;
		return true;
	}

	// CHAIN / ENTITY: build per-atom uint32 colours for one asymmetric copy
	// and tile across replicas (matches replicateGrid's layout).
	uint64_t origCount = loaded->numAtomsOriginal;
	if(origCount == 0) return false;

	loaded->hostColors.assign(origCount, 0xFFFFFFFFu);
	if(theme == ColorTheme::CHAIN){
		if(loaded->hostChainIds.size() < origCount) return false;
		for(uint64_t i = 0; i < origCount; i++){
			loaded->hostColors[i] = idToColor(loaded->hostChainIds[i]);
		}
	}else if(theme == ColorTheme::ENTITY){
		if(loaded->hostEntityIds.size() < origCount) return false;
		for(uint64_t i = 0; i < origCount; i++){
			loaded->hostColors[i] = idToColor(loaded->hostEntityIds[i]);
		}
	}

	uint64_t bytes_colors = n * sizeof(uint32_t);
	if(loaded->node->cptr_colors == 0){
		loaded->node->cptr_colors = MemoryManager::alloc(bytes_colors, "mmcif_colors");
		if(loaded->node->cptr_colors == 0){
			println("applyColorTheme: failed to allocate {} MB for colors", bytes_colors >> 20);
			return false;
		}
	}

	int ncopies = (loaded->numCopies < 1) ? 1 : loaded->numCopies;
	for(int idx = 0; idx < ncopies; idx++){
		uint64_t base = (uint64_t)idx * origCount;
		cuMemcpyHtoD(
			loaded->node->cptr_colors + base * sizeof(uint32_t),
			loaded->hostColors.data(),
			origCount * sizeof(uint32_t)
		);
	}
	loaded->currentTheme = theme;
	return true;
}

// Per-atom GPU footprint for replicated mmCIF: vec3 position + float radius + uint8 type.
// 4× cheaper than the previous uint32 color path; CHAIN/ENTITY themes optionally
// add a 4-byte legacy colour buffer on top (see replicateGrid).
constexpr uint64_t MMCIF_BYTES_PER_ATOM_GPU =
	sizeof(vec3) + sizeof(float) + sizeof(uint8_t); // = 17

// Estimated worst-case CPU peak per atom during replicateGrid: one vec3 of scratch
// (held only for the duration of one copy upload — see streamed loop in replicateGrid).
constexpr uint64_t MMCIF_BYTES_PER_ATOM_CPU_PEAK = sizeof(vec3); // = 12

struct ReplicationBudget {
	uint64_t origAtoms       = 0; // atoms in one copy
	uint64_t totalAtomsAtN   = 0; // origAtoms * requestedCopies (informational)
	uint64_t gpuNeededAtN    = 0; // bytes for requested N copies
	uint64_t cpuPeakAtN      = 0; // peak host scratch bytes during upload

	uint64_t gpuFree   = 0, gpuTotal   = 0;
	uint64_t hostAvail = 0, hostTotal  = 0;

	// Already-allocated GPU bytes that will be FREED before this replication
	// (current positions/radii/colors of this loaded structure). Adds back
	// to effectively-available VRAM in the budget calculation.
	uint64_t gpuReclaimable = 0;

	int maxSafeCopies = 1;  // largest n with both GPU and host < 70% headroom
	int maxHardCopies = 1;  // largest n that fits at all (with ~5% headroom)
	int max32bitCopies = 1; // hard cap from visbuffer 32-bit index limit
};

static ReplicationBudget computeBudget(const LoadedMmcif* loaded, int requestedN = 1){
	ReplicationBudget b;
	if(!loaded) return b;
	b.origAtoms = loaded->numAtomsOriginal;
	if(b.origAtoms == 0) return b;

	CURuntime::getGPUMemory(b.gpuFree, b.gpuTotal);
	CURuntime::getHostMemory(b.hostAvail, b.hostTotal);

	// Bytes already held by the live grid (positions+radii+colors) — those will be
	// freed at the start of replicateGrid, so they count as effectively available.
	uint64_t liveBytes = (uint64_t)loaded->numAtoms * MMCIF_BYTES_PER_ATOM_GPU;
	b.gpuReclaimable = liveBytes;

	uint64_t gpuEffective = b.gpuFree + b.gpuReclaimable;

	// 32-bit visbuffer cap (matches existing replicateGrid logic)
	uint64_t cap32 = (b.origAtoms == 0) ? 1 : (0xFFFFFFFEull / b.origAtoms);
	b.max32bitCopies = (int)std::min<uint64_t>(cap32, (uint64_t)INT_MAX);
	if(b.max32bitCopies < 1) b.max32bitCopies = 1;

	// Reserve headroom: 5% or 64 MB, whichever is larger, for driver+other allocations.
	uint64_t hardReserveGPU = std::max<uint64_t>(gpuEffective / 20, 64ull << 20);
	uint64_t safeReserveGPU = std::max<uint64_t>(gpuEffective * 3 / 10, 256ull << 20);
	uint64_t hardReserveHost = std::max<uint64_t>(b.hostAvail / 20, 256ull << 20);
	uint64_t safeReserveHost = std::max<uint64_t>(b.hostAvail * 3 / 10, 1ull << 30);

	uint64_t gpuHardBudget = (gpuEffective > hardReserveGPU) ? gpuEffective - hardReserveGPU : 0;
	uint64_t gpuSafeBudget = (gpuEffective > safeReserveGPU) ? gpuEffective - safeReserveGPU : 0;
	uint64_t hostHardBudget = (b.hostAvail > hardReserveHost) ? b.hostAvail - hardReserveHost : 0;
	uint64_t hostSafeBudget = (b.hostAvail > safeReserveHost) ? b.hostAvail - safeReserveHost : 0;

	uint64_t perCopyGPU = (uint64_t)b.origAtoms * MMCIF_BYTES_PER_ATOM_GPU;
	// CPU peak does NOT scale with N (we now stream one copy at a time):
	uint64_t cpuPeakConstant = (uint64_t)b.origAtoms * MMCIF_BYTES_PER_ATOM_CPU_PEAK;

	auto maxCopiesFor = [&](uint64_t gpuBudget, uint64_t hostBudget) -> int {
		if(perCopyGPU == 0) return b.max32bitCopies;
		uint64_t byGPU  = gpuBudget / perCopyGPU;
		uint64_t byHost = (cpuPeakConstant == 0 || cpuPeakConstant <= hostBudget)
			? (uint64_t)b.max32bitCopies : 0;
		uint64_t cap    = std::min<uint64_t>({ byGPU, byHost, (uint64_t)b.max32bitCopies });
		if(cap < 1) cap = 1;
		return (int)std::min<uint64_t>(cap, (uint64_t)INT_MAX);
	};

	b.maxHardCopies = maxCopiesFor(gpuHardBudget, hostHardBudget);
	b.maxSafeCopies = maxCopiesFor(gpuSafeBudget, hostSafeBudget);
	if(b.maxSafeCopies > b.maxHardCopies) b.maxSafeCopies = b.maxHardCopies;

	int n = (requestedN < 1) ? 1 : requestedN;
	b.totalAtomsAtN = (uint64_t)b.origAtoms * (uint64_t)n;
	b.gpuNeededAtN  = perCopyGPU * (uint64_t)n;
	b.cpuPeakAtN    = cpuPeakConstant; // streaming -> independent of N

	return b;
}

// Build N copies of the loaded molecule on a 2D grid (ceil(sqrt(N)) per side),
// each cell sized to the molecule's AABB so neighbors don't overlap. Replaces
// the GPU arrays in `loaded->node` and updates `numAtoms`/`centroid`/`radius`.
//
// Returns false (and leaves the existing GPU buffers intact) if the budget check
// rules out the requested N — so the caller's previous view is preserved.
static bool replicateGrid(LoadedMmcif* loaded, int n) {
	if(!loaded || n < 1) return false;
	if(loaded->hostPositions.empty()) {
		println("replicateGrid: no host data kept (cannot replicate)");
		return false;
	}

	uint64_t origCount = loaded->numAtomsOriginal;
	uint64_t totalCount = origCount * (uint64_t)n;
	if(totalCount > 0xFFFFFFFEull) {
		println("replicateGrid: {} atoms exceeds 32-bit visbuffer index limit, capping to fit",
			totalCount);
		n = (int)(0xFFFFFFFEull / origCount);
		totalCount = origCount * (uint64_t)n;
		if(n < 1) return false;
	}

	// Pre-allocation memory budget guard: refuse before we free or allocate anything.
	{
		ReplicationBudget b = computeBudget(loaded, n);
		if(n > b.maxHardCopies){
			println("replicateGrid: REFUSED — {} copies would need {} MB GPU + {} MB host peak "
				"but workstation can fit at most {} copies (free GPU {} MB, host {} MB).",
				n, b.gpuNeededAtN >> 20, b.cpuPeakAtN >> 20,
				b.maxHardCopies, b.gpuFree >> 20, b.hostAvail >> 20);
			return false;
		}
	}

	int side  = (int)ceilf(sqrtf((float)n));   // 2D grid: side × side
	vec3 cell = loaded->aabbMax - loaded->aabbMin;
	if(cell.x <= 0) cell.x = 1.0f;
	if(cell.y <= 0) cell.y = 1.0f;
	if(cell.z <= 0) cell.z = 1.0f;

	println("replicateGrid: {} copies on a {}x{} grid, cell=({:.1f},{:.1f},{:.1f}) Å, total atoms={}",
		n, side, side, cell.x, cell.y, cell.z, totalCount);

	// Free old GPU arrays. The palette is reusable (256 entries), so we keep
	// it. cptr_colors only exists when the active theme is CHAIN/ENTITY — in
	// that case we re-allocate it at the new replicated size below.
	bool needsLegacyColors = (loaded->node->cptr_colors != 0);
	if(loaded->node->cptr_positions) MemoryManager::free(loaded->node->cptr_positions);
	if(loaded->node->cptr_radii)     MemoryManager::free(loaded->node->cptr_radii);
	if(loaded->node->cptr_atomTypes) MemoryManager::free(loaded->node->cptr_atomTypes);
	if(loaded->node->cptr_colors)    MemoryManager::free(loaded->node->cptr_colors);
	loaded->node->cptr_positions = 0;
	loaded->node->cptr_radii     = 0;
	loaded->node->cptr_atomTypes = 0;
	loaded->node->cptr_colors    = 0;

	uint64_t bytes_pos    = totalCount * sizeof(vec3);
	uint64_t bytes_radii  = totalCount * sizeof(float);
	uint64_t bytes_atoms  = totalCount * sizeof(uint8_t);
	uint64_t bytes_colors = totalCount * sizeof(uint32_t);

	CUdeviceptr cptr_pos    = MemoryManager::alloc(bytes_pos,   "mmcif_positions");
	CUdeviceptr cptr_radii  = MemoryManager::alloc(bytes_radii, "mmcif_radii");
	CUdeviceptr cptr_atoms  = MemoryManager::alloc(bytes_atoms, "mmcif_atomTypes");
	CUdeviceptr cptr_colors = needsLegacyColors
		? MemoryManager::alloc(bytes_colors, "mmcif_colors")
		: 0;
	if(cptr_pos == 0 || cptr_radii == 0 || cptr_atoms == 0
		|| (needsLegacyColors && cptr_colors == 0))
	{
		println("replicateGrid: GPU allocation failed; rolling back.");
		if(cptr_pos)    MemoryManager::free(cptr_pos);
		if(cptr_radii)  MemoryManager::free(cptr_radii);
		if(cptr_atoms)  MemoryManager::free(cptr_atoms);
		if(cptr_colors) MemoryManager::free(cptr_colors);
		loaded->node->numSpheres = 0;
		loaded->numAtoms = 0;
		loaded->numCopies = 0;
		return false;
	}

	// Per-copy scratch (one copy worth of positions, NOT all N concatenated).
	// This keeps the host peak constant in N — the previous version allocated
	// `totalCount * sizeof(vec3)` here, which dwarfed system RAM at large N.
	vector<vec3> scratch(origCount);

	for(int idx = 0; idx < n; idx++) {
		int gx = idx % side;
		int gy = idx / side;
		vec3 offset = vec3{ gx * cell.x, gy * cell.y, 0.0f };

		for(uint64_t i = 0; i < origCount; i++) {
			scratch[i] = loaded->hostPositions[i] + offset;
		}

		uint64_t base = (uint64_t)idx * origCount;
		cuMemcpyHtoD(cptr_pos    + base * sizeof(vec3),
			scratch.data(), origCount * sizeof(vec3));
		cuMemcpyHtoD(cptr_radii  + base * sizeof(float),
			loaded->hostRadii.data(),  origCount * sizeof(float));
		cuMemcpyHtoD(cptr_atoms  + base * sizeof(uint8_t),
			loaded->hostAtomTypes.data(), origCount * sizeof(uint8_t));
		if(needsLegacyColors){
			cuMemcpyHtoD(cptr_colors + base * sizeof(uint32_t),
				loaded->hostColors.data(), origCount * sizeof(uint32_t));
		}
	}

	loaded->node->cptr_positions = cptr_pos;
	loaded->node->cptr_radii     = cptr_radii;
	loaded->node->cptr_atomTypes = cptr_atoms;
	loaded->node->cptr_colors    = cptr_colors;
	loaded->node->numSpheres     = (uint32_t)totalCount;
	loaded->numAtoms             = (uint32_t)totalCount;
	loaded->numCopies            = n;

	// Update centroid + radius for camera framing
	vec3 origCenter = (loaded->aabbMin + loaded->aabbMax) * 0.5f;
	vec3 gridCenter = vec3{
		(side - 1) * 0.5f * cell.x,
		(side - 1) * 0.5f * cell.y,
		0.0f
	};
	loaded->centroid = origCenter + gridCenter;
	loaded->radius   = sqrtf(
		powf(side * cell.x * 0.5f, 2.0f) +
		powf(side * cell.y * 0.5f, 2.0f) +
		powf((loaded->aabbMax.z - loaded->aabbMin.z) * 0.5f, 2.0f));

	return true;
}

// ---- Async wrapper around `load()` ----
// Spawns a worker thread, returns immediately with a LoadingProgress that the
// main thread can poll. The worker pushes the CUDA primary context (same
// pattern as LargeGlbLoader) so cuMemcpyHtoD inside `load()` works on the
// worker. Caller is expected to read `result` once `stage` reaches DONE.
static std::shared_ptr<LoadingProgress> loadAsync(const string& filepath, CUcontext ctx) {
	auto progress = std::make_shared<LoadingProgress>();
	progress->filepath = filepath;
	progress->stage.store(LoadingProgress::READING, std::memory_order_relaxed);

	std::thread([filepath, ctx, progress]() {
		cuCtxSetCurrent(ctx);
		std::shared_ptr<LoadedMmcif> loaded;
		try {
			loaded = load(filepath, ctx, progress.get());
		} catch(...) {
			loaded = nullptr;
		}
		if(loaded) {
			progress->result = loaded;
			progress->stage.store(LoadingProgress::DONE, std::memory_order_release);
		} else {
			progress->stage.store(LoadingProgress::FAILED, std::memory_order_release);
		}
	}).detach();

	return progress;
}

} // namespace mmcif
