#!/usr/bin/env python3
"""
cybertron_model.py - model reader for Transformers: Cybertron Adventures (Wii)

Reads the meshes stored in the game's .data files (characters, levels,
cutscenes) using cybertron_tex.py for the container and textures.

    import cybertron_model as cm
    for m in cm.scan_models("trypticon.data"):
        print(m.name, len(m.meshes), m.bounds())

Model chunk group 0xB0 (one model starts at each sub-chunk 0x08):
    0x08  materials. Characters: 0x3C-byte records
            +0x00 f32 x4 (shading params), +0x10 u32 flags (bit 0 = env map),
            +0x14 5 x (u32 texture hash, u32 flags): slot0 diffuse,
            slot1 specular colour, slot3 sphere env map, slot4 overlay
          Levels/props: a pool of entries, each an optional header followed
          by (texture hash, 0xFFFFxxxx) layer pairs (layer 0 = diffuse)
    0x06  vertex data (one 32-byte-aligned sub-buffer per mesh)
    0x05  u32 offset of each mesh's sub-buffer in 0x06
    0x13  GX display list per mesh (one triangle strip, opcode 0x98)
    0x04  0x24 bytes per mesh: +0x0C/+0x10 vertex declaration hash pair,
          +0x1C u16 vertex count
    0x12  per-mesh material reference (s04 field 6 = word index of the
          mesh's record): 0x1C-byte records (characters,
          +0x14 = material offset) or 0x18-byte records (levels:
          [-1,0,0,-1,off,0] single entry, [-1,0,-1,start,0,end] layered)
          or 8-byte records [-1, off] (sky domes)
    0x02  scene-node transforms (4x4 f32, row vectors, translation in the
          last row); 0x04 field 5 = node of each mesh. Applied to static
          meshes; skinned meshes are already in model space. A node whose
          3x3 part is all zero hides its meshes until the game scales them
          in; they are shown at the node's position with unit scale.
    0x0D  per-mesh bone palette (bone hashes), 0x0C skeleton: 0x44-byte
          records, bone hash + 4x4 inverse bind matrix (row vectors).
          Skinned display-list vertices start with GX matrix-index bytes:
          byte 0 = position matrix = 3 x (palette slot + 1); every vertex is
          bound to one bone (rigid skinning).
    group 0x80 sub 3 (same sub-file): bone hierarchy, 0x48-byte records:
          hash, parent index (-1 = root), 5 x u32, translation xyz, u32,
          rotation quaternion xyzw, scale xyz (an animation rest pose, not
          the bind pose - only the hierarchy is used)
          Some characters/vehicles use 0x38-byte records instead: u32 flags,
          f32 x3, RGBA tint, u32, 4 x (hash, flags) - slot0 greyscale diffuse
          + stored colour (not what is shown), slot2 sphere env map.
          Body materials (flags byte 1 = 1, env map on): slot0 greyscale skin,
          slot3 = team-colour region mask (blue plates, green underframe).
          Light/decal materials (byte 1 = 0): drawn unlit as
          slot0 (shape) x (slot3.R x colour2 + slot3.B x colour1); the
          slot3 texture is a channel mask, colour2 is at +0x14
Positions are stored in bind pose (model space, Z up). Character UVs are
1/4096 fixed point, level/prop UVs 1/1024; 16-byte character vertices use
1/256 fixed-point positions.
"""

import hashlib
import os
import struct
import sys

import numpy as np

import cybertron_tex as ct

GROUP_MODEL = 0xB0
MAT_SIZE = 0x3C

# Vertex declarations, keyed by the first hash in the 0x04 record.
#   stride   bytes per vertex in the vertex buffer
#   dlpre    bytes before the attribute indices in each display-list vertex
#            (matrix index bytes) ; nidx: number of u16 indices that follow
#   pos      ("f32", offset) or ("s16", offset, scale)
#   nrm/uv/col   byte offset (None if absent); uvs = UV scale
DECLS = {
    0x889336E4: dict(name="skin24", stride=24, dlpre=5, nidx=3, pos=("f32", 0), nrm=12, uv=16, uvs=1 / 4096, col=None),
    0xB67C0AD2: dict(name="skin16", stride=16, dlpre=5, nidx=3, pos=("s16", 0, 1 / 256), nrm=6, uv=10, uvs=1 / 4096, col=None),
    0x085739A8: dict(name="skin16b", stride=16, dlpre=5, nidx=3, pos=("s16", 0, 1 / 256), nrm=6, uv=10, uvs=1 / 4096, col=None),
    0xF38D4C50: dict(name="skin16c", stride=16, dlpre=4, nidx=3, pos=("s16", 0, 1 / 256), nrm=6, uv=10, uvs=1 / 4096, col=None),
    0xB4653CF8: dict(name="shadow", stride=16, dlpre=1, nidx=2, pos=("f32", 0), nrm=None, uv=None, uvs=0, col=None),
    0xAC5D5B47: dict(name="static24", stride=24, dlpre=0, nidx=4, pos=("f32", 0), nrm=12, uv=16, uvs=1 / 1024, col=20),
    0x92D1AF58: dict(name="static16", stride=16, dlpre=0, nidx=2, pos=("f32", 0), nrm=None, uv=12, uvs=1 / 1024, col=None),
    0x09FC013D: dict(name="static28", stride=28, dlpre=0, nidx=5, pos=("f32", 0), nrm=12, uv=16, uvs=1 / 1024, col=24),
    0x333626D9: dict(name="static20c", stride=20, dlpre=0, nidx=3, pos=("f32", 0), nrm=None, uv=12, uvs=1 / 4096, col=16, colfmt="rgba8"),
    0x0E1D17A3: dict(name="pos12", stride=12, dlpre=0, nidx=1, pos=("f32", 0), nrm=None, uv=None, uvs=0, col=None),
    0xC218E926: dict(name="static20", stride=20, dlpre=0, nidx=3, pos=("f32", 0), nrm=12, uv=16, uvs=1 / 1024, col=None),
    0xD433C383: dict(name="sky16", stride=16, dlpre=0, nidx=2, pos=("f32", 0), nrm=None, uv=12, uvs=1 / 256, col=None),
    0x30B25C32: dict(name="blend24", stride=24, dlpre=0, nidx=4, pos=("f32", 0), nrm=None, uv=12, uvs=1 / 1024, col=None),
}


DECLS_BY_NAME = {d["name"]: d for d in DECLS.values()}


def guess_decls(stride, dl_vsize):
    """Candidate layouts for an unknown declaration, best first: known
    layouts with the same vertex stride (adjusting how many matrix-index
    bytes precede the indices in the display list), then a bare
    float-position layout."""
    out = []
    for d in DECLS.values():
        if d["stride"] != stride:
            continue
        pre = dl_vsize - 2 * d["nidx"]
        if 0 <= pre <= 5:
            out.append(dict(d, dlpre=pre, name=d["name"] + "?"))
    out.sort(key=lambda d: (d["dlpre"] != DECLS_BY_NAME[d["name"][:-1]]["dlpre"], d["nidx"] == 2))
    if stride >= 12 and dl_vsize >= 2:
        pre = dl_vsize % 2
        out.append(dict(name="unknown", stride=stride, dlpre=pre, nidx=(dl_vsize - pre) // 2,
                        pos=("f32", 0), nrm=None, uv=None, uvs=0, col=None))
    return out


class Material:
    """slots: list of (texture hash, flags). Character materials have 5 fixed
    slots (0 diffuse, 1 specular colour, 3 sphere environment map, 4 overlay);
    level materials are a list of texture layers (layer 0 = diffuse)."""

    def __init__(self, raw=None, slots=None, flags=0, params=(1.0, 1.0, 1.0, 32.0)):
        self.tint = (1.0, 1.0, 1.0)
        self.colour2 = (1.0, 1.0, 1.0)
        if raw is not None and len(raw) == 0x38:
            # u32 flags, f32 x3, RGBA tint, u32, 4 x (hash, flags):
            # slot0 diffuse (greyscale, tinted), slot2 sphere env map, slot3 overlay
            self.flags = struct.unpack(">I", raw[0:4])[0]
            self.params = struct.unpack(">3f", raw[4:16]) + (32.0,)
            c = raw[16:20]
            self.tint = (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)   # colour 1, as stored
            c = raw[20:24]
            self.colour2 = (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
            self.slots = [struct.unpack(">II", raw[24 + 8 * i:32 + 8 * i]) for i in range(4)]
            self.kind = "tinted"
        elif raw is not None:
            self.params = struct.unpack(">4f", raw[0:16])
            self.flags = struct.unpack(">I", raw[16:20])[0]
            self.slots = [struct.unpack(">II", raw[20 + 8 * i:28 + 8 * i]) for i in range(5)]
            self.kind = "character"
        else:
            self.params, self.flags, self.slots = params, flags, list(slots or [])
            self.kind = "layered"

    def slot_roles(self):
        """[(slot index, texture hash, role)] for every texture referenced."""
        roles = {"character": {0: "diffuse", 1: "specular", 2: "slot2", 3: "envmap", 4: "overlay"},
                 "tinted": {0: "diffuse", 1: "slot1", 2: "envmap",
                            3: "mask" if self.is_light else "overlay"}}.get(self.kind)
        out = []
        for i, (h, _fl) in enumerate(self.slots):
            if h in (0, 0xFFFFFFFF):
                continue
            role = roles.get(i, f"slot{i}") if roles else ("diffuse" if i == 0 else f"layer{i}")
            out.append((i, h, role))
        return out

    @property
    def is_light(self):
        """Tinted materials whose flags byte 1 (the env-map bit) is clear are
        decals / light strips: drawn unlit, coloured through their slot-3
        mask. Body materials always have it set. (Byte 3 is not a reliable
        indicator: some bodies have it clear.)"""
        return self.kind == "tinted" and not (self.flags >> 16) & 1

    @property
    def colour_map(self):
        """Slot-3 texture of light/decal materials: a channel mask saying
        which colour each light pixel gets (not a colour itself)."""
        return self._slot(3) if self.is_light else 0

    @property
    def mask_colours(self):
        """(colour for the mask's R channel, G channel, B channel).
        R -> colour 2 (main strips/insignia, e.g. Autobot red),
        B -> colour 1 (secondary patches); G is unused in known files and
        is given colour 1."""
        return (self.colour2, self.tint, self.tint)

    @property
    def game_tint(self):
        """Colour multiplier for the game's look. The stored colour of tinted
        materials does not set the on-screen colour (light/decal materials
        take theirs from colour_map; bodies show their greyscale skin), so
        this is white; Material.tint keeps the stored value. Plain colour
        materials (no texture) are shown in their colour."""
        if self.kind == "colour":
            return self.tint
        return (1.0, 1.0, 1.0)

    def flag_notes(self):
        """Human-readable meaning of the known flag bits."""
        if self.kind == "character":
            return [f"env map {'on' if self.flags & 1 else 'off'} (bit 0)"]
        if self.kind == "tinted":
            b = self.flags.to_bytes(4, "big")
            return [f"bytes {b[0]:02x} {b[1]:02x} {b[2]:02x} {b[3]:02x}",
                    "body part: env map on (byte 1 = 1)" if b[1] & 1 else
                    "light/decal: unlit, coloured via slot-3 mask (byte 1 = 0)",
                    f"bytes 0, 2, 3: meaning unknown ({b[0]:02x} {b[2]:02x} {b[3]:02x})"]
        if self.kind == "colour":
            return ["plain colour, no texture"]
        return ["level/prop texture layers (no flag word)"]

    def describe(self):
        """Everything known about the material, for display."""
        d = dict(colour2=None, kind=self.kind, flags=self.flags, notes=self.flag_notes(),
                 params=self.params, stored_colour=None, shown_colour=self.game_tint,
                 light=self.is_light, slots=[])
        if self.kind == "tinted":
            d["stored_colour"] = self.tint
            d["colour2"] = self.colour2
        roles = {slot: role for slot, _h, role in self.slot_roles()}
        for i, (h, fl) in enumerate(self.slots):
            d["slots"].append(dict(slot=i, hash=h, flags=fl,
                                   role=roles.get(i, "empty"),
                                   empty=h in (0, 0xFFFFFFFF)))
        return d

    def _slot(self, i):
        return self.slots[i][0] if i < len(self.slots) else 0

    @property
    def diffuse(self):
        return self._slot(0)

    @property
    def specular(self):
        return self._slot(1) if self.kind == "character" else 0

    @property
    def envmap(self):
        # bit 0 of the flags enables the sphere map on character materials;
        # on tinted materials it is bit 16
        if self.kind == "character":
            return self._slot(3) if self.flags & 1 else 0
        if self.kind == "tinted":
            return self._slot(2) if (self.flags >> 16) & 1 else 0
        return 0

    @property
    def overlay(self):
        if self.kind == "character":
            return self._slot(4)
        if self.kind == "tinted" and not self.is_light:
            return self._slot(3)        # body: blue/green team-colour region mask
        return 0

    # kept for compatibility
    glow = overlay


class Mesh:
    __slots__ = ("pos", "nrm", "uv", "col", "tris", "material", "decl", "node",
                 "index", "stride", "span", "vb_off", "layout", "hidden",
                 "pnmtx", "palette", "pnmtx_values", "pnmtx_conflicts")

    def __init__(self, pos, nrm, uv, col, tris, material, decl):
        self.pos, self.nrm, self.uv, self.col = pos, nrm, uv, col
        self.tris, self.material, self.decl = tris, material, decl
        self.node = -1
        # where the mesh came from (used by cybertron_check.py)
        self.index = self.stride = self.span = self.vb_off = -1
        self.layout = None
        self.hidden = False          # placed by a zero-scale (hidden) node
        self.pnmtx = None            # per-vertex GX position-matrix index (skinned meshes)
        self.palette = None          # bone hashes for matrix slots 0, 1, ... (s0d)
        self.pnmtx_values = None     # every matrix-index value in the display list
        self.pnmtx_conflicts = 0     # vertices referenced with two different matrices


class Model:
    def __init__(self, src, rel, index, sub_index):
        self.src = src            # path of the .data file
        self.rel = rel            # display path
        self.index = index        # model number within the file
        self.sub_index = sub_index
        self.materials = []
        self.meshes = []
        self.chunks = {}          # sub id -> [Chunk]
        self.warnings = []
        self.loaded = False
        self.bones = []           # [(hash, inverse bind 4x4 row-vector matrix)] from s0c
        self.hierarchy = []       # [(hash, parent index, local T, R quat, S)] from 0x80/3

    @property
    def name(self):
        stem = os.path.splitext(os.path.basename(self.rel))[0]
        return f"{stem} #{self.index}"

    @property
    def mesh_count(self):
        c = self.chunks.get(0x04)
        return c[0].size // 0x24 if c else 0

    @property
    def material_count(self):
        c = self.chunks.get(0x08)
        return c[0].size if c else 0

    @property
    def is_proxy(self):
        """Shadow / collision proxies have no materials."""
        return self.material_count == 0

    def texture_hashes(self):
        return {s[0] for m in self.materials for s in m.slots}

    def bounds(self):
        if not self.meshes:
            return np.zeros(3), np.zeros(3)
        lo = np.min([m.pos.min(0) for m in self.meshes], 0)
        hi = np.max([m.pos.max(0) for m in self.meshes], 0)
        return lo, hi

    def material_usage(self):
        """material index -> (mesh count, triangle count)."""
        use = {}
        for m in self.meshes:
            a, b = use.get(m.material, (0, 0))
            use[m.material] = (a + 1, b + len(m.tris))
        return use

    def material_report(self, find_texture=None):
        """Plain-text listing of every material (find_texture: hash -> Texture)."""
        self.load()
        use = self.material_usage()
        out = [f"Materials of {self.name} ({self.rel}) - style: {self.mat_style}", ""]
        for i, mat in enumerate(self.materials):
            d = mat.describe()
            n, t = use.get(i, (0, 0))
            out.append(f"mat {i}: {d['kind']}, used by {n} mesh(es), {t} triangles")
            if d["kind"] != "layered":
                out.append(f"  flags 0x{d['flags']:08x}: " + "; ".join(d["notes"]))
                out.append("  params " + " ".join(f"{p:g}" for p in d["params"]))
            if d["stored_colour"] is not None:
                c = d["stored_colour"]
                hx = lambda c: "#%02x%02x%02x" % tuple(round(v * 255) for v in c)
                out.append(f"  colour 1 {hx(c)}   colour 2 {hx(d['colour2'])}")
                out.append("  shown: " + (f"unlit light; slot-3 mask R -> colour 2 {hx(d['colour2'])}, "
                                          f"mask B -> colour 1 {hx(c)}" if d["light"]
                                          else "greyscale skin, untinted"))
            for sl in d["slots"]:
                if sl["empty"]:
                    out.append(f"  slot {sl['slot']}: (empty)")
                    continue
                tx = find_texture(sl["hash"]) if find_texture else None
                info = f"{tx.width}x{tx.height} {tx.fmt}, {tx.mips} mip(s), in {os.path.basename(tx.src)}" \
                    if tx else "not found in the game folder"
                out.append(f"  slot {sl['slot']} {sl['role']:9s} {sl['hash']:08x}  flags {sl['flags']:08x}  {info}")
            out.append("")
        unused = use.get(-1)
        if unused:
            out.append(f"{unused[0]} mesh(es) without a material")
        return "\n".join(out)

    def triangle_count(self):
        return sum(len(m.tris) for m in self.meshes)

    def vertex_count(self):
        return sum(len(m.pos) for m in self.meshes)

    # ------------------------------------------------------------------
    def load(self, f=None):
        """Decode materials and meshes (reads the file)."""
        if self.loaded:
            return self
        own = f is None
        if own:
            f = ct.open_data(self.src)
        try:
            self._load(f)
        finally:
            if own:
                f.close()
        self.loaded = True
        return self

    @staticmethod
    def _pool_material_keys(s12, n, rec_start=None):
        """Per-mesh material references for pooled (level/prop) materials.
        s12 is a stream of 32-bit words; s04 field 6 (rec_start) gives the
        word index where each mesh's record starts. Within a record, what
        follows the last 0xFFFFFFFF marker is [offset, 0] (single entry) or
        [start, 0, end] (layered entry). Without usable record starts the
        stream is split assuming the same number of markers per mesh."""
        if n <= 0 or len(s12) < 4:
            return []
        words = struct.unpack(f">{len(s12) // 4}I", s12[:len(s12) // 4 * 4])
        nw = len(words)
        spans = None
        if rec_start and len(rec_start) == n and all(0 <= r < nw for r in rec_start) and \
                all(words[r] == 0xFFFFFFFF for r in rec_start):
            order = sorted(set(rec_start))
            nxt = {r: (order[k + 1] if k + 1 < len(order) else nw) for k, r in enumerate(order)}
            spans = [(r, nxt[r]) for r in rec_start]
        else:
            marks = [k for k, x in enumerate(words) if x == 0xFFFFFFFF]
            if not marks or len(marks) % n:
                return [None] * n
            per = len(marks) // n
            spans = [(marks[i * per], marks[(i + 1) * per] if i + 1 < n else nw) for i in range(n)]
        keys = []
        for a, b in spans:
            rec = words[a:b]
            last = max(k for k, x in enumerate(rec) if x == 0xFFFFFFFF)
            tail = rec[last + 1:]
            if len(tail) >= 3:
                keys.append(("range", tail[0], tail[2]))
            elif tail:
                keys.append(("single", tail[0]))
            else:
                keys.append(None)
        return keys

    @staticmethod
    def _layered_material(pool, key):
        """Level/prop materials live in a pool; an entry is an optional header
        (flags, sometimes floats) followed by (texture hash, 0xFFFFxxxx) pairs."""
        if key[0] == "single":
            a, b = key[1], min(key[1] + 0x40, len(pool))
        else:
            a, b = key[1], min(key[2], len(pool))
        slots, p = [], a
        while p + 8 <= b:
            h, fl = struct.unpack(">II", pool[p:p + 8])
            if fl >> 16 == 0xFFFF and h not in (0, 0xFFFFFFFF) and (h >> 16) != 0xFFFF:
                slots.append((h, fl))
                p += 8
            elif slots:
                break
            else:
                p += 4
        if slots:
            return Material(slots=slots)
        # no texture layers: a plain colour entry (RGBA, alpha byte 0xFF)
        if key[0] == "single" and a + 4 <= len(pool):
            c = pool[a:a + 4]
            if c[3] == 0xFF:
                m = Material(slots=[])
                m.kind = "colour"
                m.tint = (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
                return m
        return None

    def _apply_node(self, mesh, node):
        """Place a static mesh with its scene node (s04 field 5). Skinned
        meshes are already stored in model space (bind pose) and are left
        alone."""
        mesh.node = node
        if mesh.decl.startswith("skin") or not 0 <= node < len(self.nodes):
            return
        M = self.nodes[node]
        if not np.isfinite(M).all() or np.allclose(M, np.eye(4), atol=1e-6):
            return
        A, t = M[:3, :3], M[3, :3]
        if not A.any():
            # rotation/scale all zero: the object is hidden until the game
            # scales it in (doors, pickups...). Show it at its position, unit scale.
            mesh.hidden = True
            A = np.eye(3)
        mesh.pos = (mesh.pos.astype(np.float64) @ A + t).astype(np.float32)
        if mesh.nrm is not None:
            try:
                n = mesh.nrm.astype(np.float64) @ np.linalg.inv(A).T
                ln = np.linalg.norm(n, axis=1, keepdims=True)
                mesh.nrm = (n / np.where(ln > 1e-9, ln, 1)).astype(np.float32)
            except np.linalg.LinAlgError:
                pass
        if np.linalg.det(A) < 0:                       # mirrored: keep the winding
            mesh.tris = mesh.tris[:, [0, 2, 1]].copy()

    def _blob(self, f, sub, i=0):
        c = self.chunks[sub][i]
        return ct._read_at(f, c.abs, c.size)

    def _load(self, f):
        g = self.chunks
        mats = self._blob(f, 0x08) if 0x08 in g else b""
        n04 = g[0x04][0].size // 0x24 if 0x04 in g else 0
        s12 = self._blob(f, 0x12) if 0x12 in g else b""
        rec = len(s12) // n04 if n04 else 0
        mesh_mat = {}
        msize = None
        if rec == 0x1C:
            offs_ = [struct.unpack(">I", s12[i * rec + 0x14:i * rec + 0x18])[0] for i in range(n04)]
            # 0x3C records start with a float (1.0), 0x38 records with a flags word
            first_is_float = len(mats) >= 4 and 0x3D000000 <= struct.unpack(">I", mats[:4])[0] <= 0x42000000
            for size in ((MAT_SIZE, 0x38) if first_is_float else (0x38, MAT_SIZE)):
                if len(mats) and len(mats) % size == 0 and all(o % size == 0 and o < len(mats) for o in offs_):
                    msize = size
                    break
        self.mat_style = {MAT_SIZE: "character", 0x38: "tinted"}.get(msize, "layered")
        if msize:
            self.materials = [Material(mats[i:i + msize])
                              for i in range(0, len(mats) - msize + 1, msize)]
            for i, mo in enumerate(offs_):
                mesh_mat[i] = mo // msize
        else:
            self.materials = []
            index = {}
            s04_raw = self._blob(f, 0x04) if 0x04 in g else b""
            rec_start = [struct.unpack(">I", s04_raw[i * 0x24 + 0x18:i * 0x24 + 0x1C])[0]
                         for i in range(n04)] if len(s04_raw) >= n04 * 0x24 else None
            for i, key in enumerate(self._pool_material_keys(s12, n04, rec_start)):
                if key is None:
                    continue
                if key not in index:
                    m = self._layered_material(mats, key)
                    index[key] = len(self.materials) if m else -1
                    if m:
                        self.materials.append(m)
                if index[key] >= 0:
                    mesh_mat[i] = index[key]
        self.mesh_mat = mesh_mat
        if not all(k in g for k in (0x04, 0x05, 0x06, 0x13)):
            self.warnings.append("incomplete model (missing chunks)")
            return
        s04 = self._blob(f, 0x04)
        ndl = len(s04) // 0x24
        s05 = self._blob(f, 0x05)
        offs = struct.unpack(f">{ndl}I", s05[:4 * ndl])
        vb = self._blob(f, 0x06)
        # scene-node transforms (0x02): 4x4, row vectors, translation in the last row
        self.nodes = []
        if 0x02 in g:
            s02 = self._blob(f, 0x02)
            self.nodes = [np.array(struct.unpack(">16f", s02[k:k + 64]), np.float64).reshape(4, 4)
                          for k in range(0, len(s02) - 63, 64)]
        dls = g[0x13]
        if len(dls) != ndl:
            self.warnings.append(f"{len(dls)} display lists for {ndl} meshes")
        palettes = [None] * ndl
        if 0x0C in g:
            s0c = self._blob(f, 0x0C)
            self.bones = [(struct.unpack(">I", s0c[k:k + 4])[0],
                           np.array(struct.unpack(">16f", s0c[k + 4:k + 68]), np.float64).reshape(4, 4))
                          for k in range(0, len(s0c) - 67, 68)]
            if len(g.get(0x0D, [])) == ndl:
                for k in range(ndl):
                    b = self._blob(f, 0x0D, k)
                    palettes[k] = list(struct.unpack(f">{len(b) // 4}I", b[:len(b) // 4 * 4]))
            else:
                self.warnings.append("bone palettes don't match the meshes; no skinning")
            self.hierarchy = self._find_hierarchy(f)
        for i in range(min(ndl, len(dls))):
            e = struct.unpack(">9I", s04[i * 0x24:(i + 1) * 0x24])
            span = (offs[i + 1] if i + 1 < ndl else len(vb)) - offs[i]
            count = e[7] >> 16
            mat = self.mesh_mat.get(i, -1)
            dl = ct._read_at(f, dls[i].abs, dls[i].size)
            try:
                mesh = self._decode_mesh(vb, offs[i], count, span, dl, e[3], mat)
            except Exception as ex:  # keep going with the other meshes
                self.warnings.append(f"mesh {i}: {ex}")
                continue
            if mesh is not None:
                mesh.index, mesh.span, mesh.vb_off = i, span, offs[i]
                if mesh.pnmtx is not None and palettes[i]:
                    mesh.palette = palettes[i]
                else:
                    mesh.pnmtx = None
                self._apply_node(mesh, e[5])
                self.meshes.append(mesh)

    def _find_hierarchy(self, f):
        """The bone hierarchy (group 0x80 sub 3) in this model's sub-file
        that covers the most of its bones."""
        try:
            subs = ct.find_subfiles(f, ct.data_size(f), self.src)
            chunks = subs[self.sub_index][1]
        except Exception:
            return []
        want = {h for h, _ in self.bones}
        best, best_cover = [], 0
        for c in chunks:
            if c.is_meta or c.group != 0x80 or c.sub != 3 or c.size % 72:
                continue
            b = ct._read_at(f, c.abs, c.size)
            recs = []
            for o in range(0, len(b), 72):
                v = struct.unpack(">II5I3fI4f3f", b[o:o + 72])
                par = -1 if v[1] == 0xFFFFFFFF else v[1]
                recs.append((v[0], par, v[7:10], v[11:15], v[15:18]))
            if any(not (-1 <= r[1] < i) for i, r in enumerate(recs)):
                continue                       # parents must come first
            cover = len(want & {r[0] for r in recs})
            if cover > best_cover:
                best, best_cover = recs, cover
        return best

    def skeleton(self):
        """Joints for export: list of dict(hash, parent, world, ibm), parents
        first. world = bind-pose 4x4 (row vectors, model space); ibm = the
        game's inverse bind matrix (None for helper bones that no vertex
        uses). Hierarchy from group 0x80; bones it doesn't list hang off
        the root."""
        if not self.bones:
            return []
        ibm = {h: m for h, m in self.bones}
        # bind poses of bones this model doesn't use, from the other models
        # in the same sub-file (e.g. the full-detail mesh of a low-detail one)
        shared = {}
        try:
            with ct.open_data(self.src) as f:
                for other in scan_models(self.src):
                    if other.sub_index != self.sub_index or 0x0C not in other.chunks:
                        continue
                    b = other._blob(f, 0x0C)
                    for k in range(0, len(b) - 67, 68):
                        h = struct.unpack(">I", b[k:k + 4])[0]
                        shared.setdefault(h, np.array(struct.unpack(">16f", b[k + 4:k + 68]),
                                                      np.float64).reshape(4, 4))
        except Exception:
            pass
        joints, index = [], {}

        def local_from_rec(r):
            x, y, z, w = r[3]
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
                          [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
                          [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)]])
            L = np.eye(4)
            L[:3, :3] = np.diag(r[4]) @ R
            L[3, :3] = r[2]
            return L

        for r in self.hierarchy:
            h, par = r[0], r[1]
            if h in index:
                continue
            pj = index.get(self.hierarchy[par][0], -1) if par >= 0 else -1
            if h in ibm or h in shared:
                try:
                    world = np.linalg.inv(ibm.get(h, shared.get(h)))
                except np.linalg.LinAlgError:
                    world = np.eye(4)
            else:
                world = local_from_rec(r) @ (joints[pj]["world"] if pj >= 0 else np.eye(4))
            index[h] = len(joints)
            joints.append(dict(hash=h, parent=pj, world=world, ibm=ibm.get(h)))
        loose = [h for h, _ in self.bones if h not in index]
        if loose:
            root = 0 if joints and joints[0]["parent"] < 0 else -1
            if root < 0:
                joints.insert(0, dict(hash=0, parent=-1, world=np.eye(4), ibm=None))
                for j in joints[1:]:
                    j["parent"] = j["parent"] + 1 if j["parent"] >= 0 else 0
                index = {j["hash"]: k for k, j in enumerate(joints)}
                root = 0
            for h in loose:
                index[h] = len(joints)
                joints.append(dict(hash=h, parent=root, world=np.linalg.inv(ibm[h]), ibm=ibm[h]))
        return joints

    def _decode_mesh(self, vb, off, count, span, dl, decl_hash, mat):
        if count == 0 or len(dl) < 3:
            return None
        # display list: opcode, u16 count, vertices; then zero padding
        decl = DECLS.get(decl_hash)
        if decl is not None:
            return self._decode_with(decl, vb, off, count, dl, mat)
        n0 = struct.unpack(">H", dl[1:3])[0]
        vsize = (len(dl) - 3) // max(n0, 1)
        while vsize > 1 and 3 + n0 * vsize > len(dl):
            vsize -= 1
        # stride: sub-buffers are count*stride rounded up to 32 bytes
        stride = next((st for st in range(32, 3, -2)
                       if (count * st + 31) // 32 * 32 == span), None) or span // count
        last = None
        for cand in guess_decls(stride, vsize):
            try:
                mesh = self._decode_with(cand, vb, off, count, dl, mat)
            except ValueError as ex:
                last = ex
                continue
            self.warnings.append(f"unknown vertex format {decl_hash:08x}, decoded as {cand['name']}")
            return mesh
        raise ValueError(f"unknown vertex format {decl_hash:08x}"
                         + (f" ({last})" if last else ""))

    def _decode_with(self, decl, vb, off, count, dl, mat):
        st = decl["stride"]
        raw = np.frombuffer(vb, np.uint8, count * st, off).reshape(count, st)

        kind = decl["pos"]
        if kind[0] == "f32":
            pos = raw[:, kind[1]:kind[1] + 12].copy().view(">f4").astype(np.float32)
        else:
            pos = (raw[:, kind[1]:kind[1] + 6].copy().view(">i2").astype(np.float32) * kind[2])
        nrm = uv = col = None
        if decl["nrm"] is not None:
            o = decl["nrm"]
            nrm = raw[:, o:o + 3].view(np.int8).astype(np.float32) / 64.0
        if decl["uv"] is not None:
            o = decl["uv"]
            uv = raw[:, o:o + 4].copy().view(">i2").astype(np.float32) * decl["uvs"]
        if decl["col"] is not None and decl.get("colfmt") == "rgba8":
            o = decl["col"]
            col = raw[:, o:o + 3].astype(np.float32) / 255.0
        elif decl["col"] is not None:
            o = decl["col"]
            c = raw[:, o:o + 2].copy().view(">u2").astype(np.uint32)
            col = np.stack([(c >> 11 & 31) * 255 // 31, (c >> 5 & 63) * 255 // 63,
                            (c & 31) * 255 // 31], -1).astype(np.float32) / 255.0

        # triangle strips -> triangles
        vsize = decl["dlpre"] + 2 * decl["nidx"]
        tris = []
        mtx_parts = []
        p = 0
        n = len(dl)
        while p + 3 <= n:
            op = dl[p]
            if op == 0:
                p += 1
                continue
            prim = op & 0xF8
            cnt = struct.unpack(">H", dl[p + 1:p + 3])[0]
            p += 3
            if p + cnt * vsize > n:
                raise ValueError("display list overruns its chunk")
            arr = np.frombuffer(dl, np.uint8, cnt * vsize, p).reshape(cnt, vsize)
            idx = arr[:, decl["dlpre"]:decl["dlpre"] + 2].copy().view(">u2").ravel().astype(np.int64)
            if decl["dlpre"]:
                mtx_parts.append((idx, arr[:, 0].copy()))
            p += cnt * vsize
            if prim == 0x98 and cnt >= 3:                     # triangle strip
                a, b, c = idx[:-2], idx[1:-1], idx[2:]
                odd = (np.arange(cnt - 2) & 1).astype(bool)
                t = np.stack([np.where(odd, b, a), np.where(odd, a, b), c], 1)
            elif prim == 0x90:                                 # triangles
                t = idx[:cnt // 3 * 3].reshape(-1, 3)
            elif prim == 0xA0 and cnt >= 3:                   # triangle fan
                t = np.stack([np.full(cnt - 2, idx[0]), idx[1:-1], idx[2:]], 1)
            elif prim == 0x80 and cnt >= 4:                   # quads
                q = idx[:cnt // 4 * 4].reshape(-1, 4)
                t = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
            else:
                continue
            tris.append(t)
        if not tris:
            raise ValueError("no triangles")
        t = np.concatenate(tris)
        keep = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
        t = t[keep]
        if not len(t):
            raise ValueError("no triangles")
        if t.max() >= count:
            raise ValueError("index out of range")
        if not np.isfinite(pos).all() or np.abs(pos).max() > 1e6:
            raise ValueError("implausible vertex positions")
        mesh = Mesh(pos, nrm, uv, col, t.astype(np.uint32), mat, decl["name"])
        mesh.stride, mesh.layout = st, decl
        if mtx_parts:
            vi = np.concatenate([a for a, _ in mtx_parts])
            mv = np.concatenate([b for _, b in mtx_parts])
            ok = vi < count
            pn = np.zeros(count, np.uint8)
            pn[vi[ok]] = mv[ok]
            mesh.pnmtx = pn
            mesh.pnmtx_values = np.unique(mv)
            pairs = np.unique(vi[ok] * 256 + mv[ok])
            mesh.pnmtx_conflicts = int(len(pairs) - len(np.unique(vi[ok])))
        return mesh


def scan_models(path, rel=None, warn=None):
    """List the models in a .data file without decoding them."""
    rel = rel or os.path.basename(path)
    out = []
    with ct.open_data(path) as f:
        n = ct.data_size(f)
        subs = ct.find_subfiles(f, n, path)
        for si, (_, chunks) in enumerate(subs):
            cur = None
            for c in chunks:
                if c.is_meta or c.group != GROUP_MODEL:
                    continue
                if c.sub == 0x08:
                    cur = Model(path, rel, len(out), si)
                    out.append(cur)
                if cur is not None:
                    cur.chunks.setdefault(c.sub, []).append(c)
    return out


def _texture_refs_and_masked(blob, mask):
    """Texture references in a material blob: (hash, 0xFFFFxxxx flags) word
    pairs. Returns (set of hashes, blob with the low 16 bits of every such
    flags word cleared if `mask`). Those bits differ between otherwise
    identical copies (e.g. character vs _npc) and are not used for drawing."""
    n4 = len(blob) // 4 * 4
    if n4 < 8:
        return set(), blob
    a = np.frombuffer(blob[:n4], ">u4").copy()
    h, fl = a[:-1], a[1:]
    pair = np.nonzero(((fl >> 16) == 0xFFFF) & (h != 0) & ((h >> 16) != 0xFFFF))[0]
    refs = {int(x) for x in h[pair]}
    if mask and len(pair):
        fi = pair + 1
        fi = fi[a[fi] != 0xFFFFFFFF]
        a[fi] &= np.uint32(0xFFFF0000)
        blob = a.tobytes() + blob[n4:]
    return refs, blob


def _model_digest(model, f, tex_digest):
    """Content fingerprint of everything the viewer draws for `model`:
    every model chunk byte for byte (material slot-flag ID bits excepted)
    plus the pixel data of every texture it references."""
    h = hashlib.sha1()
    refs = set()
    for sub in sorted(model.chunks):
        for c in model.chunks[sub]:
            f.seek(c.abs)
            b = f.read(c.size)
            if len(b) != c.size:
                return None
            if sub in (0x08, 0x12):
                r, b2 = _texture_refs_and_masked(b, sub == 0x08)
                refs |= r
                if sub == 0x08:
                    b = b2
            h.update(struct.pack(">BI", sub, c.size))
            h.update(b)
    for t in sorted(refs):
        h.update(struct.pack(">I", t))
        h.update(tex_digest(t, model.src))
    return h.digest()


def duplicate_rank(model):
    """Lower = preferred copy to keep: uncompressed files, names without
    _npc, then shorter paths."""
    rel = model.rel.lower().replace("\\", "/")
    base = os.path.basename(rel)
    return ("compressed" in base or ct.is_compressed_file(model.src),
            "_npc" in base, len(rel), rel, model.index)


def find_duplicates(models, find_texture=None, stop=None):
    """Find models that are exact copies of another model.
    Returns {index in models: index of the copy that is kept}.

    Candidates must have identical chunk tables (same parts, same sizes);
    they are then compared byte for byte (see _model_digest), including the
    pixel data of the textures they use. `find_texture(hash, prefer_src)`
    resolves a texture the way the viewer does."""
    groups = {}
    for i, m in enumerate(models):
        if not m.chunks:
            continue
        sig = tuple((sub, tuple(c.size for c in m.chunks[sub])) for sub in sorted(m.chunks))
        groups.setdefault(sig, []).append(i)
    cands = [g for g in groups.values() if len(g) > 1]
    if not cands:
        return {}

    tcache = {}

    def tex_digest(hsh, src):
        t = find_texture(hsh, src) if find_texture else None
        if t is None or not t.fmt:
            return b"-"
        key = (t.src, t.file_off)
        if key not in tcache:
            try:
                with ct.open_data(t.src) as tf:
                    tf.seek(t.file_off)
                    raw = tf.read(t.byte_size())
                tcache[key] = hashlib.sha1(struct.pack(">HHI", t.width, t.height, t.fmt_id) + raw).digest()
            except OSError:
                tcache[key] = b"?" + struct.pack(">I", hsh)
        return tcache[key]

    by_file = {}
    for g in cands:
        for i in g:
            by_file.setdefault(models[i].src, []).append(i)
    digest = {}
    for src, idxs in by_file.items():
        if stop is not None and stop():
            return {}
        try:
            with ct.open_data(src) as f:
                for i in idxs:
                    digest[i] = _model_digest(models[i], f, tex_digest)
        except (OSError, ValueError):
            continue
    out = {}
    for g in cands:
        same = {}
        for i in g:
            d = digest.get(i)
            if d is not None:
                same.setdefault(d, []).append(i)
        for members in same.values():
            if len(members) < 2:
                continue
            keep = min(members, key=lambda i: duplicate_rank(models[i]))
            for i in members:
                if i != keep:
                    out[i] = keep
    return out


# ----------------------------------------------------------------- glTF ---
_Y_UP = [-0.7071067811865476, 0.0, 0.0, 0.7071067811865476]   # Z-up game space -> glTF Y-up


def _quat_from_matrix(R):
    """3x3 rotation (column-vector form) -> quaternion [x, y, z, w]."""
    t = np.trace(R)
    if t > 0:
        s_ = np.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s_, (R[0, 2] - R[2, 0]) / s_, (R[1, 0] - R[0, 1]) / s_, 0.25 * s_]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s_ = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s_, (R[0, 1] + R[1, 0]) / s_, (R[0, 2] + R[2, 0]) / s_, (R[2, 1] - R[1, 2]) / s_]
    elif R[1, 1] > R[2, 2]:
        s_ = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s_, 0.25 * s_, (R[1, 2] + R[2, 1]) / s_, (R[0, 2] - R[2, 0]) / s_]
    else:
        s_ = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s_, (R[1, 2] + R[2, 1]) / s_, 0.25 * s_, (R[1, 0] - R[0, 1]) / s_]
    q = np.array(q, np.float64)
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return [float(x) for x in q]


def _node_transform(local_row):
    """Row-vector 4x4 -> glTF node fields (TRS when it decomposes cleanly,
    otherwise a matrix)."""
    M = local_row.T                      # column-vector form
    t = M[:3, 3]
    A = M[:3, :3]
    sc = np.linalg.norm(A, axis=0)
    if (sc < 1e-9).any():
        return {"matrix": [float(x) for x in local_row.ravel()]}
    if np.linalg.det(A) < 0:
        sc[0] = -sc[0]
    R = A / sc
    if np.abs(R.T @ R - np.eye(3)).max() > 1e-4:
        return {"matrix": [float(x) for x in local_row.ravel()]}   # sheared
    out = {}
    if np.abs(t).max() > 1e-9:
        out["translation"] = [float(x) for x in t]
    q = _quat_from_matrix(R)
    if abs(q[3] - 1) > 1e-9:
        out["rotation"] = q
    if np.abs(sc - 1).max() > 1e-6:
        out["scale"] = [float(x) for x in sc]
    return out


def _decompose_frames(L):
    """(F, 4, 4) row-vector local matrices -> translations (F,3),
    quaternions (F,4, xyzw, sign-continuous) and scales (F,3)."""
    M = np.transpose(L, (0, 2, 1))            # column-vector form
    t = M[:, :3, 3]
    A = M[:, :3, :3]
    sc = np.linalg.norm(A, axis=1)            # column norms
    sc = np.where(sc < 1e-9, 1e-9, sc)
    neg = np.linalg.det(A) < 0
    sc[neg, 0] *= -1
    R = A / sc[:, None, :]
    q = np.array([_quat_from_matrix(r) for r in R])
    for k in range(1, len(q)):
        if np.dot(q[k], q[k - 1]) < 0:
            q[k] = -q[k]
    return t, q, sc


class _GlbBuilder:
    def __init__(self):
        self.bin = bytearray()
        self.views, self.accessors = [], []

    def view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        v = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            v["target"] = target
        self.bin += data
        self.views.append(v)
        return len(self.views) - 1

    def accessor(self, arr, kind, target=None, minmax=False, normalized=False):
        arr = np.ascontiguousarray(arr)
        ctype = {np.dtype(np.float32): 5126, np.dtype(np.uint32): 5125,
                 np.dtype(np.uint16): 5123, np.dtype(np.uint8): 5121}[arr.dtype]
        a = {"bufferView": self.view(arr.astype(arr.dtype.newbyteorder("<")).tobytes(), target),
             "componentType": ctype, "count": int(arr.shape[0]), "type": kind}
        if normalized:
            a["normalized"] = True
        if minmax:
            a["min"] = [float(x) for x in np.atleast_1d(arr.min(axis=0))]
            a["max"] = [float(x) for x in np.atleast_1d(arr.max(axis=0))]
        self.accessors.append(a)
        return len(self.accessors) - 1


def export_glb(model, path, images=None, embed=False, skeleton=True, clips=None, progress=None):
    """Write the model as a binary glTF (.glb) and, for characters/vehicles,
    the skeleton and vertex weights.

    images: texture key -> PIL image, keys as from Material (texture hash,
    or ("baked", diffuse, colour map) for light/decal materials).
    Textures are written as PNG files next to the .glb and referenced by
    relative path (embed=True packs them into the .glb instead).
    Game space is Z-up; a root node rotates it to glTF's Y-up. Skinning is
    rigid (one bone per vertex, weight 1), as in the game.

    skeleton=False writes the meshes alone, in the bind pose.
    clips: [(ClipInfo, kind)] from cybertron_anim (kind "same"/"shared") -
    each is baked as a glTF animation on the joints, which Blender imports
    as an action. Needs the skeleton. progress(done, total) is called per
    clip."""
    import io
    import json
    model.load()
    images = images or {}
    B = _GlbBuilder()
    gl = {"asset": {"version": "2.0", "generator": "Cybertron Adventures tools"},
          "scene": 0, "scenes": [{"name": model.name, "nodes": [0]}],
          "nodes": [{"name": model.name, "rotation": _Y_UP, "children": []}],
          "meshes": [], "materials": [], "buffers": [{"byteLength": 0}]}
    nodes = gl["nodes"]
    used_ext = set()

    # ---- textures
    tex_index, img_list, tex_list = {}, [], []
    out_dir = os.path.dirname(os.path.abspath(path))
    role_of = {}
    for mat in model.materials or []:
        for _slot, h, role in mat.slot_roles():
            role_of.setdefault(h, role)
    written = []

    def texture(key):
        if key is None or key not in images:
            return None
        if key not in tex_index:
            im = images[key]
            if isinstance(key, tuple):
                name = f"{key[1]:08x}x{key[2]:08x}_light"
            else:
                name = f"{key:08x}_{role_of.get(key, 'texture')}"
            if embed:
                buf = io.BytesIO()
                im.save(buf, "PNG")
                img_list.append({"bufferView": B.view(buf.getvalue()), "mimeType": "image/png",
                                 "name": name})
            else:
                im.save(os.path.join(out_dir, name + ".png"))
                written.append(name + ".png")
                img_list.append({"uri": name + ".png", "name": name})
            tex_list.append({"sampler": 0, "source": len(img_list) - 1})
            tex_index[key] = len(tex_list) - 1
        return tex_index[key]

    def has_alpha(key):
        im = images.get(key)
        if im is None or im.mode not in ("RGBA", "LA", "P"):
            return False
        return im.convert("RGBA").getchannel("A").getextrema()[0] < 128

    # ---- materials
    for i, mat in enumerate(model.materials or []):
        m = {"name": f"mat{i}", "doubleSided": True,
             "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.8}}
        tint = list(mat.game_tint) if mat.kind != "colour" else list(mat.tint)
        pbr = m["pbrMetallicRoughness"]
        pbr["baseColorFactor"] = [float(x) for x in tint] + [1.0]
        baked_key = ("baked", mat.diffuse, mat.colour_map) if mat.colour_map else None
        main = baked_key if baked_key in images else mat.diffuse
        ti = texture(main)
        if ti is not None:
            pbr["baseColorTexture"] = {"index": ti}
            if has_alpha(main):
                m["alphaMode"] = "MASK"
                m["alphaCutoff"] = 0.5
        if baked_key in images:
            # light/decal material: drawn unlit in the game
            m["extensions"] = {"KHR_materials_unlit": {}}
            used_ext.add("KHR_materials_unlit")
            pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
        extras = {}
        for slot, h, role in mat.slot_roles():
            if role == "diffuse" and h == mat.diffuse:
                continue
            k = texture(h)
            if k is not None:
                extras[f"{role}Texture"] = {"index": k, "slot": slot, "hash": f"{h:08x}"}
        if extras:
            m["extras"] = extras
        gl["materials"].append(m)

    # ---- skeleton
    joints = model.skeleton() if skeleton and any(me.palette for me in model.meshes) else []
    skin_index = None
    joint_of = {}
    if joints:
        first = len(nodes)
        for k, j in enumerate(joints):
            pw = joints[j["parent"]]["world"] if j["parent"] >= 0 else np.eye(4)
            local = j["world"] @ np.linalg.inv(pw)
            nd = {"name": f"bone_{j['hash']:08x}" if j["hash"] else "root"}
            nd.update(_node_transform(local))
            nodes.append(nd)
            joint_of[j["hash"]] = k
        for k, j in enumerate(joints):
            par = nodes[first + j["parent"]] if j["parent"] >= 0 else nodes[0]
            par.setdefault("children", []).append(first + k)
        ibms = np.stack([(j["ibm"] if j["ibm"] is not None else np.linalg.inv(j["world"])).ravel()
                         for j in joints]).astype(np.float32)
        gl["skins"] = [{"name": f"{model.name} skeleton", "joints": list(range(first, first + len(joints))),
                        "skeleton": first, "inverseBindMatrices": B.accessor(ibms, "MAT4")}]
        skin_index = 0

    # ---- meshes
    for k, me in enumerate(model.meshes):
        n = len(me.pos)
        attrs = {"POSITION": B.accessor(me.pos.astype(np.float32), "VEC3", 34962, minmax=True)}
        if me.nrm is not None:
            nr = me.nrm.astype(np.float32)
            ln = np.linalg.norm(nr, axis=1, keepdims=True)
            nr = np.where(ln > 1e-6, nr / np.maximum(ln, 1e-6), np.array([0, 0, 1], np.float32))
            attrs["NORMAL"] = B.accessor(nr.astype(np.float32), "VEC3", 34962)
        if me.uv is not None:
            attrs["TEXCOORD_0"] = B.accessor(me.uv.astype(np.float32), "VEC2", 34962)
        if me.col is not None:
            attrs["COLOR_0"] = B.accessor(np.clip(me.col, 0, 1).astype(np.float32), "VEC3", 34962)
        skinned = skin_index is not None and me.palette and me.pnmtx is not None
        if skinned:
            slot = me.pnmtx.astype(np.int64) // 3 - 1
            lut = np.array([joint_of.get(h, 0) for h in me.palette] + [0], np.int64)
            slot = np.where((slot >= 0) & (slot < len(me.palette)), slot, len(me.palette))
            jt = np.zeros((n, 4), np.uint16 if len(joints) > 255 else np.uint8)
            jt[:, 0] = lut[slot]
            wt = np.zeros((n, 4), np.float32)
            wt[:, 0] = 1.0
            attrs["JOINTS_0"] = B.accessor(jt, "VEC4", 34962)
            attrs["WEIGHTS_0"] = B.accessor(wt, "VEC4", 34962)
        prim = {"attributes": attrs, "mode": 4,
                "indices": B.accessor(me.tris.astype(np.uint32).ravel(), "SCALAR", 34963)}
        if 0 <= me.material < len(gl["materials"]):
            prim["material"] = me.material
        gl["meshes"].append({"name": f"mesh{k:03d}", "primitives": [prim]})
        nd = {"name": f"mesh{k:03d}" + ("_hidden" if me.hidden else ""), "mesh": len(gl["meshes"]) - 1}
        if skinned:
            nd["skin"] = skin_index
            nodes.append(nd)
            gl["scenes"][0]["nodes"].append(len(nodes) - 1)     # skinned meshes at the scene root
        else:
            nodes.append(nd)
            nodes[0]["children"].append(len(nodes) - 1)

    # ---- animations (Blender: one action per clip)
    anim_names, anim_failed = [], []
    if joints and clips:
        import cybertron_anim as ca
        first = gl["skins"][0]["joints"][0]
        seen = {}
        for ci, (info, kind) in enumerate(clips):
            if progress:
                progress(ci, len(clips))
            try:
                L = ca.bake_joint_locals(model, joints, info, kind)
            except Exception as ex:          # noqa: BLE001
                L = None
                print(f"animation {info.name}: {ex}")
            if L is None:
                anim_failed.append(info.name)
                continue
            F = L.shape[0]
            times = np.arange(F, dtype=np.float32) / np.float32(ca.FPS)
            t_all = B.accessor(times, "SCALAR", minmax=True)
            t_one = None
            samplers, channels = [], []

            def channel(node, path, vals, kind_):
                nonlocal t_one
                if np.abs(vals - vals[0]).max() < 1e-6:       # constant: one key
                    if t_one is None:
                        t_one = B.accessor(times[:1], "SCALAR", minmax=True)
                    inp, vals = t_one, vals[:1]
                else:
                    inp = t_all
                samplers.append({"input": inp, "output": B.accessor(vals.astype(np.float32), kind_),
                                 "interpolation": "LINEAR"})
                channels.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": path}})

            for k in range(len(joints)):
                t, q, sc = _decompose_frames(L[:, k])
                channel(first + k, "translation", t, "VEC3")
                channel(first + k, "rotation", q, "VEC4")
                channel(first + k, "scale", sc, "VEC3")
            name = info.name or f"clip{ci}"
            if name in seen:
                stem = os.path.splitext(os.path.basename(info.rel.replace("\\", "/")))[0]
                name = f"{name} ({stem})"
                while name in seen:
                    name += "+"
            seen[name] = 1
            gl.setdefault("animations", []).append({"name": name, "samplers": samplers, "channels": channels})
            anim_names.append(name)
        if progress:
            progress(len(clips), len(clips))
        if anim_names:       # animated nodes must be TRS, not a matrix (glTF rule)
            for k in range(len(joints)):
                nd = nodes[first + k]
                if "matrix" in nd:
                    t, q, sc = _decompose_frames(np.array(nd.pop("matrix")).reshape(1, 4, 4))
                    nd.update(translation=[float(x) for x in t[0]], rotation=[float(x) for x in q[0]],
                              scale=[float(x) for x in sc[0]])

    if img_list:
        gl["images"], gl["textures"] = img_list, tex_list
        gl["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
    if not gl["materials"]:
        del gl["materials"]
    if used_ext:
        gl["extensionsUsed"] = sorted(used_ext)
    if not nodes[0]["children"]:
        del nodes[0]["children"]
    while len(B.bin) % 4:
        B.bin.append(0)
    gl["buffers"][0]["byteLength"] = len(B.bin)
    gl["bufferViews"], gl["accessors"] = B.views, B.accessors
    js = json.dumps(gl, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    with open(path, "wb") as fo:
        fo.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(B.bin)))
        fo.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        fo.write(struct.pack("<II", len(B.bin), 0x004E4942) + bytes(B.bin))
    return dict(joints=len(joints), meshes=len(model.meshes), textures=len(img_list), files=written,
                animations=len(anim_names), animations_failed=anim_failed)


def export_obj(model, path, textures=None):
    """Write a Wavefront OBJ (+ MTL). `textures`: hash -> PNG path."""
    model.load()
    base = os.path.splitext(path)[0]
    mtl = os.path.basename(base) + ".mtl"
    with open(path, "w") as o:
        o.write(f"# {model.name} from {model.rel}\nmtllib {mtl}\n")
        vo = 1
        for k, m in enumerate(model.meshes):
            o.write(f"o mesh{k:03d}\nusemtl mat{max(m.material, 0)}\n")
            for p in m.pos:
                o.write(f"v {p[0]:.6f} {p[2]:.6f} {-p[1]:.6f}\n")   # Z-up -> Y-up
            # always write vt / vn for every vertex so the numbering stays aligned
            uv = m.uv if m.uv is not None else np.zeros((len(m.pos), 2), np.float32)
            for u in uv:
                o.write(f"vt {u[0]:.6f} {1 - u[1]:.6f}\n")
            nr = m.nrm if m.nrm is not None else np.tile([0.0, 0.0, 1.0], (len(m.pos), 1))
            for nn in nr:
                o.write(f"vn {nn[0]:.4f} {nn[2]:.4f} {-nn[1]:.4f}\n")
            for t in m.tris + vo:
                o.write("f " + " ".join(f"{i}/{i}/{i}" for i in t) + "\n")
            vo += len(m.pos)
    with open(base + ".mtl", "w") as o:
        o.write(f"# materials for {model.name}\n"
                "# map_Kd = diffuse, map_Ks = specular colour, refl = sphere env map,\n"
                "# map_Ke = overlay layer (slot 4 on characters / slot 3 on tinted\n"
                "# body materials: hit-flash / team-colour mask), or for light/decal\n"
                "# materials the baked diffuse x colour map (they are drawn unlit)\n\n")
        for i, mat in enumerate(model.materials or [None]):
            kd = mat.game_tint if mat is not None else (1.0, 1.0, 1.0)
            o.write(f"newmtl mat{i}\nKd {kd[0]:.4f} {kd[1]:.4f} {kd[2]:.4f}\n")
            if mat is None or not textures:
                o.write("\n")
                continue
            baked = textures.get(("baked", mat.diffuse, mat.colour_map)) if mat.colour_map else None
            if baked:
                # light/decal material: diffuse mask x colour map, drawn unlit
                o.write(f"# light material: slot0 shape x (mask R x colour 2 + mask B x colour 1), baked\n"
                        f"map_Kd {baked}\nKe 1 1 1\nmap_Ke {baked}\n")
            for slot, h, role in mat.slot_roles():
                fn = textures.get(h)
                o.write(f"# slot {slot} ({role}): {h:08x}" + ("" if fn else "  - not found in the game folder") + "\n")
                if not fn:
                    continue
                if role == "diffuse":
                    if not baked:
                        o.write(f"map_Kd {fn}\n")
                elif role == "specular":
                    o.write(f"map_Ks {fn}\n")
                elif role == "envmap":
                    o.write(f"refl -type sphere {fn}\n")
                elif role == "overlay":
                    o.write(f"map_Ke {fn}\n")
            o.write("\n")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        for m in scan_models(p):
            m.load()
            lo, hi = m.bounds()
            print(f"{m.name}: {len(m.meshes)} meshes, {m.vertex_count()} verts, "
                  f"{m.triangle_count()} tris, {len(m.materials)} materials, "
                  f"bounds {lo.round(1)} .. {hi.round(1)}"
                  + (f"  warnings: {m.warnings}" if m.warnings else ""))
