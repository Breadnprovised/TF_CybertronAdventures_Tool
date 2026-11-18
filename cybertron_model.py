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
    0x0D  per-mesh bone palette (bone hashes), 0x0C skeleton (hash + 4x4)
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
                 "index", "stride", "span", "vb_off", "layout", "hidden")

    def __init__(self, pos, nrm, uv, col, tris, material, decl):
        self.pos, self.nrm, self.uv, self.col = pos, nrm, uv, col
        self.tris, self.material, self.decl = tris, material, decl
        self.node = -1
        # where the mesh came from (used by cybertron_check.py)
        self.index = self.stride = self.span = self.vb_off = -1
        self.layout = None
        self.hidden = False          # placed by a zero-scale (hidden) node


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
                self._apply_node(mesh, e[5])
                self.meshes.append(mesh)

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
