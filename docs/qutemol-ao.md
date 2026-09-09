# QuteMol-style baked ambient occlusion — design notes

Status: **exploration only, nothing implemented yet.** This records the design and the
numbers behind it so the implementation choice is made on evidence rather than taste.

Reference scene throughout: `MG_1184_cellpack_atom_instances.cif`, the mycoplasma cell —
**158,922,213 atoms**, centroid `(7.47, -4.84, -0.04)`, radius **1658 Å** (AABB ~3300 Å
across). GPU is a Quadro RTX 8000, 48 GB, of which the loaded scene uses ~4 GB.

## Why this is worth doing at all

Screen-space AO cannot supply the large-scale cue on this scene, and that is a property
of the data, not of the algorithm. At whole-cell framing the model covers ~1.7M pixels
while carrying 158.9M atoms, so roughly 90 atoms fall in each pixel and the visbuffer
keeps one of them. The depth buffer is therefore *atom noise*, not a surface: a horizon
search marching across it finds a different random atom top at every step, and the result
is near-uniform occlusion — a constant dimmer rather than a shape cue. This is exactly
what the whole-cell captures showed, for both the legacy hemisphere sampler and GTAO.

Baked AO is computed in object space against the actual geometry, so it does not care
how many atoms land in a pixel. It is also view-independent, noise-free, and free per
frame. That makes it complementary to GTAO rather than a replacement:

    final AO = bakedAO (large scale, view-independent)
             * gtaoAO  (contact detail, view-dependent)

## What QuteMol actually does

Tarini, Cignoni & Montani 2006, *Ambient Occlusion and Edge Cueing for Enhancing Real
Time Molecular Visualization*:

1. Give every atom a small patch in a texture atlas, parameterising that atom's sphere
   surface (8x8 texels per atom is the usual figure).
2. Pick N directions on the sphere (they use ~128).
3. For each direction, render the whole molecule orthographically into a depth buffer.
4. For each atom-patch texel, project it into that view and compare against the depth
   buffer. If visible, accumulate `max(0, dot(texelNormal, direction))`.
5. Normalise. The atlas now holds smooth, view-independent, directionally varying AO.

The per-texel detail is what gives QuteMol its characteristic look: AO varies *across*
each atom's surface, not just between atoms.

## Why the literal method does not port

QuteMol targets thousands to ~100k atoms. At 158.9M atoms the atlas alone is the
blocker:

| atlas resolution | texels | at 1 byte/texel |
|---|---|---|
| 8x8 per atom (QuteMol) | 10.17 G | **10.2 GB** |
| 4x4 per atom | 2.54 G | 2.5 GB |
| 2x2 per atom | 636 M | 636 MB |
| 1x1 per atom (scalar) | 158.9 M | **159 MB** |

10.2 GB is not impossible on a 48 GB card, but it is a poor trade: per-texel variation
across a single atom is invisible at any framing where you can see the whole cell, which
is precisely the framing that needs the help.

## Three scalable adaptations

### A. Per-atom scalar AO — 159 MB

One byte per atom. Drops the intra-atom directional variation, keeps the part that
matters at mesoscale: buried atoms dark, surface atoms bright.

Bake reuses the existing renderer, which is the appealing part — CuRast already
rasterises 158.9M spheres per frame at interactive rates, so a bake pass is just that
loop run N times against a depth-only target:

- for each of N directions: rasterise spheres, then for each atom project its centre,
  read the depth at that pixel, and count it visible if it is the front-most surface.
- Estimated cost: a sphere pass is ~16 ms at current framerates, so N=64 is roughly
  **1 second**, N=128 roughly 2 s. One-time, at load.

Open questions: the sphere rasteriser is perspective (`c_target.proj`); a bake wants
orthographic. Either add an ortho path or approximate with a distant camera and a narrow
FOV. Float32 linear depth has ample relative precision over a 3300 Å structure either
way.

### B. AO volume — 17 MB

Bake into a 3D grid instead of per atom, sample it trilinearly at the atom position in
the shader. 256³ over 3300 Å is 12.9 Å per cell; 512³ is 6.4 Å for 134 MB.

Cheapest by far in both memory and code, and it captures the enclosure gradient well.
It cannot distinguish an atom on top of a protein bump from one in the groove beside it
at 256³ — but that is ~20-50 Å detail, which is GTAO's job anyway.

### C. Per-atom SH (l=1) — 636 MB

Four coefficients per atom gives directional AO per atom: the side of an atom facing
outward is brighter than the side facing into the cell. Closest to QuteMol's look at a
cost that still fits. Worth considering only after A is working.

## Recommendation

Start with **A**, because it is the honest adaptation of QuteMol's actual method, reuses
the renderer's existing strength, and 159 MB is affordable against 48 GB. Fall back to
**B** if the bake turns out to be awkward to wire up — B is perhaps a fifth of the code
and would still answer the question "does view-independent baked AO fix the whole-cell
flatness?"

Neither replaces GTAO. Baked AO has no contact detail at close framing; GTAO has no
large-scale cue. They multiply.

## Edge cueing / halo — implemented

Read from `pyQuteMol` (`Qutemol/CgUtil.py`, `MakeHaloShader`). QuteMol's halo is a
separate pass: an enlarged billboard per atom rendered into a **reduced-size** halo
texture (`1 << powres`), and per fragment:

    tmp2.x = data.z * (1 - (u^2 + v^2))            # radial falloff inside the disc
    tmp.z  = sceneDepth - fragDepth                # gap to whatever is behind
    tmp.z  = saturate(tmp.z * 1/P_depth_full)
    tmp.z *= tmp2.x
    tmp.z *= tmp2.x                                # "again for smoother edges"
    result = black * tmp.z + haloColour

So the halo is opaque exactly where a silhouette stands in front of something far
behind it, and fades both radially and with the depth gap. Parameters are
`P_halo_size`, `P_halo_str`, `P_halo_col`, `P_halo_aware`, `P_depth_full`.

Reproduced here as a **screen-space pass over the depth buffer** rather than per-atom
billboards, which are not an option at 158.9M atoms. For each pixel, look outward along
`haloDirs` directions x `haloSteps` radii for a surface nearer than this one; take the
max of `saturate(gap / depthFull) * falloff^2`. Sparse sampling is fine because the
result is a max of a smooth function — QuteMol reached the same conclusion from the
other direction by rendering the halo at reduced resolution.

Two details that mattered:

- **The squared falloff is not cosmetic.** A linear falloff leaves a visible hard ring
  at the search radius. QuteMol's second multiply is doing real work.
- **`depthFull` needs to be scale-relative.** Expressed as a fraction of pixel depth,
  not an absolute distance, or it stops working the moment you dolly. At 0.02 on this
  scene (≈52 Å at the test framing) ordinary inter-atom gaps triggered a full-strength
  halo and the whole interior went dark; 0.06 restricts it to genuine silhouette jumps
  while still outlining individual complexes.

Note this overlaps with the existing EDL, which also darkens depth discontinuities.
EDL is symmetric and local; the halo is one-sided (only where a *nearer* surface
exists) and reaches much further, which is what produces the cut-out separation rather
than a crease darkening.

Still missing from the QuteMol look: `P_border_inside` / `P_border_outside`, the
per-atom outline drawn in the atom's own shader. With analytic sphere normals already
available in the resolve pass, that is a cheap addition — darken where `dot(N, V)`
falls below a threshold, with the threshold widened by the local depth gap.

## Plumbing already in place

`SphereRasterArgs` (`src/kernels/HostDeviceInterface.h`) is where a per-atom AO buffer
would hang, next to `positions` / `radii` / `atomTypes`:

    uint8_t* ambientOcclusion;   // null = not baked

The sphere shading path in `resolve.cu` already looks up per-atom data by `sphere_idx`
for colour, so reading a per-atom AO byte at the same site is a one-line addition.
