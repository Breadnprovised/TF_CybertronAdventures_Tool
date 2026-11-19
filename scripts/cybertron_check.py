#!/usr/bin/env python3
"""
cybertron_check.py - checks every model and mesh in a Transformers: Cybertron
Adventures (Wii) game folder for problems that would make it render wrongly.

    python3 cybertron_check.py GAME_FOLDER_OR_FILES... [--json report.json]
                               [--out report.txt] [--errors-only | --all] [--verbose]

Run without arguments (e.g. double-clicked) and a folder picker opens. On
Windows, when the script gets its own console window, it waits for Enter
before closing so the report stays readable.

By default errors and warnings are listed; --all adds info-level findings,
--verbose also lists models with no findings.

Needs numpy and Pillow, with cybertron_tex.py and cybertron_model.py in the
same folder. No graphics card or window is needed.

What is checked (severity in brackets):
  file level
    UNREADABLE_FILE [error]  a .data file whose layout could not be read
                             (truncated / damaged / unknown container): its
                             models and textures are invisible to the tools
    PARTIAL_FILE    [warn]   only part of a .data file could be mapped
    COMPRESSED_FILE [info]   zlib-compressed container (read transparently)
  model level
    NO_MESHES       [error]  model has data but no mesh could be decoded
    EMPTY_MODEL     [info]   placeholder model with no geometry at all
  mesh level
    DECODE_FAILED   [error]  a mesh could not be decoded (unknown vertex
                             format the fallback could not handle, bad data)
    GUESSED_FORMAT  [warn]   unknown vertex format, decoded by guessing the
                             layout - check it visually and report the hash
    NO_MATERIAL     [error]  mesh has no material (renders flat grey)
    BAD_POSITIONS   [error]  non-finite / absurd vertex positions
    STRIDE_MISMATCH [error]  the vertex size does not fit the mesh's vertex
                             buffer (count x stride, padded to 32 bytes) - the
                             vertex format is decoded with the wrong layout
    POSITION_TYPE   [error]  positions decoded as 16-bit integers but the data
                             is floats, or the other way round
    UNUSED_VERTS    [info]   the mesh's triangles use less than half of its
                             vertices
    DEGENERATE      [warn]   most triangles have zero area
    COLLAPSED_MESH  [info]   every vertex is stored at one point in the game data
                             (an object removed by the artists; invisible in-game)
    BAD_NORMALS     [warn]   stored normals far from unit length (all-zero
                             normals are allowed: they render flat-shaded)
    UV_RANGE        [warn]   UVs far outside the usual range (wrong UV scale)
    UV_NAN          [error]  UVs are not finite
    BAD_NODE        [error]  the mesh's placement transform is not finite or
                             is singular (but not all-zero, see HIDDEN_NODE)
    HIDDEN_NODE     [info]   placed by a zero-scale node: hidden in the game
                             until scaled in; the tools show it at unit scale
    NO_MATERIAL_NOUV [info]  mesh without texture coordinates has no material
                             (cannot be textured anyway)
    BAD_SKIN        [error]  skinning data doesn't fit together: a vertex uses
                             a bone slot outside its mesh's palette, a palette
                             bone is missing from the skeleton, or the bone
                             palettes don't match the meshes
    NO_BONE_HIERARCHY [info] skeleton without a bone hierarchy (0x80/3):
                             exported with every bone under one root
  material level
    MISSING_DIFFUSE [warn]   diffuse texture(s) not found anywhere in the folder
                             (one line per model; full list at the end)
    MISSING_TEXTURE [info]   other texture slots not in the folder (several
                             hashes, e.g. e3ea32bd / e700dce5, appear in unused
                             slots everywhere and are probably placeholders)
    TEXTURE_DECODE  [error]  texture found but could not be decoded
    UNKNOWN_TEX_FMT [warn]   texture uses an unknown pixel format id
    LIGHT_MASK      [warn]   light/decal material whose slot-3 mask has no
                             red channel (its glow would render black) or
                             body material whose team mask is mostly red -
                             i.e. the body/light classification looks wrong
The exit code is 1 if any error was found, else 0.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np

import cybertron_tex as ct
import cybertron_model as cm

SEVERITY = {
    "NO_MESHES": "error", "EMPTY_MODEL": "info", "DECODE_FAILED": "error",
    "GUESSED_FORMAT": "warn", "NO_MATERIAL": "error", "BAD_POSITIONS": "error",
    "STRIDE_MISMATCH": "error", "POSITION_TYPE": "error", "UNUSED_VERTS": "info",
    "DEGENERATE": "warn", "COLLAPSED_MESH": "info", "BAD_NORMALS": "warn",
    "UV_RANGE": "warn", "UV_NAN": "error", "BAD_NODE": "error",
    "MISSING_DIFFUSE": "warn", "MISSING_TEXTURE": "info", "TEXTURE_DECODE": "error",
    "UNKNOWN_TEX_FMT": "warn", "LIGHT_MASK": "warn", "COMPRESSED_FILE": "info",
    "UNREADABLE_FILE": "error", "PARTIAL_FILE": "warn", "HIDDEN_NODE": "info",
    "NO_MATERIAL_NOUV": "info", "BAD_SKIN": "error", "NO_BONE_HIERARCHY": "info",
}
RANK = {"error": 0, "warn": 1, "info": 2}

# thresholds, calibrated on all known-good samples (1353 meshes) and on
# deliberately wrong decodes of them
DEGENERATE_FRAC = 0.5
S16_LOOKS_FLOAT = 0.30       # good 16-bit position data scores 0.00, float data >= 0.40
F32_LOOKS_FLOAT = 0.20       # good float position data scores >= 0.40
NORMAL_BAD_FRAC = 0.20
UV_LIMIT = 64.0


MISSING_GLOBAL = defaultdict(Counter)   # hash -> Counter(diffuse=, other=, models=)


class Report:
    def __init__(self):
        self.items = []          # dicts: file, model, mesh, material, code, severity, detail

    def add(self, code, detail, file, model=None, mesh=None, material=None):
        self.items.append(dict(file=file, model=model, mesh=mesh, material=material,
                               code=code, severity=SEVERITY[code], detail=detail))


# ----------------------------------------------------------------------------
# texture index (same lookup rules as the viewer)
# ----------------------------------------------------------------------------
class TextureIndex:
    def __init__(self):
        self.by_hash = defaultdict(list)
        self.decoded = {}        # hash -> array or Exception

    def add_file(self, path):
        for t in ct.scan_file(path, allow_full_scan=False, warn=lambda m: None):
            self.by_hash[t.hash].append(t)

    def find(self, h, prefer_src=None):
        lst = self.by_hash.get(h)
        if not lst:
            return None
        for t in lst:
            if t.src == prefer_src and t.fmt:
                return t
        for t in lst:
            if t.fmt:
                return t
        return lst[0]

    def decode(self, t):
        if t.hash not in self.decoded:
            try:
                self.decoded[t.hash] = t.read_mip(0)
            except Exception as e:  # noqa: BLE001 - reported, not raised
                self.decoded[t.hash] = e
        return self.decoded[t.hash]


# ----------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------
def _float_plausibility(raw12):
    """Share of vertices whose first 12 bytes read as 3 sensible floats."""
    with np.errstate(invalid="ignore", over="ignore"):
        v = raw12.copy().view(">f4").astype(np.float64)
        a = np.abs(v)
        ok = np.isfinite(v) & ((a == 0) | ((a > 1e-20) & (a < 1e6)))
    return float(ok.all(1).mean()) if len(v) else 1.0


def check_mesh(rep, f, mname, mesh, model, vb):
    k = mesh.index
    p = mesh.pos
    count = len(p)
    lay = mesh.layout or {}
    if not np.isfinite(p).all() or (count and np.abs(p).max() > 1e6):
        rep.add("BAD_POSITIONS", "non-finite or huge vertex positions", f, mname, k)
        return
    # 1) vertex size must fit the mesh's sub-buffer exactly (32-byte padded)
    if mesh.stride > 0 and mesh.span >= 0 and (count * mesh.stride + 31) // 32 * 32 != mesh.span:
        fits = [st for st in range(8, 64, 2) if (count * st + 31) // 32 * 32 == mesh.span]
        rep.add("STRIDE_MISMATCH", f"format {mesh.decl} uses {mesh.stride}-byte vertices but the buffer "
                f"holds {count} vertices in {mesh.span} bytes"
                + (f" (fits a stride of {', '.join(map(str, fits))})" if fits else ""), f, mname, k)
    raw = None
    if vb is not None and mesh.stride >= 12 and mesh.vb_off >= 0 and \
            mesh.vb_off + count * mesh.stride <= len(vb):
        raw = np.frombuffer(vb, np.uint8, count * mesh.stride, mesh.vb_off).reshape(count, mesh.stride)
    # 2) position type
    if raw is not None and lay.get("pos"):
        fp = _float_plausibility(raw[:, :12])
        if lay["pos"][0] == "s16" and fp > S16_LOOKS_FLOAT:
            rep.add("POSITION_TYPE", f"positions decoded as 16-bit integers, but {fp:.0%} of vertices "
                    f"read as valid floats (format {mesh.decl})", f, mname, k)
        elif lay["pos"][0] == "f32" and fp < F32_LOOKS_FLOAT:
            rep.add("POSITION_TYPE", f"positions decoded as floats, but only {fp:.0%} of vertices are "
                    f"valid floats (format {mesh.decl})", f, mname, k)
    # 3) stored normals (raw bytes, before any placement transform)
    if raw is not None and lay.get("nrm") is not None:
        o = lay["nrm"]
        n = raw[:, o:o + 3].view(np.int8).astype(np.float32) / 64.0
        ln = np.linalg.norm(n, axis=1)
        nz = ln > 0.05
        if nz.any():
            bad = float(((ln[nz] < 0.6) | (ln[nz] > 1.4)).mean())
            if bad > NORMAL_BAD_FRAC:
                rep.add("BAD_NORMALS", f"{bad:.0%} of stored normals are far from unit length "
                        f"(format {mesh.decl}) - suggests a wrong vertex layout", f, mname, k)
    t = mesh.tris
    if not len(t):
        return
    used = len(np.unique(t))
    if used < 0.5 * count:
        rep.add("UNUSED_VERTS", f"triangles use only {used} of {count} vertices", f, mname, k)
    if count > 1 and not np.ptp(p, axis=0).any():
        rep.add("COLLAPSED_MESH", f"all {count} vertices are stored at the same point "
                f"({p[0][0]:.1f}, {p[0][1]:.1f}, {p[0][2]:.1f}) - nothing to draw, "
                "the game shows nothing here either", f, mname, k)
        return
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    diag = float(np.linalg.norm(p.max(0) - p.min(0)))
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    dfrac = float((area <= max(diag, 1e-6) ** 2 * 1e-12).mean())
    if dfrac > DEGENERATE_FRAC:
        rep.add("DEGENERATE", f"{dfrac:.0%} of triangles have zero area", f, mname, k)
    if mesh.uv is not None and len(mesh.uv):
        if not np.isfinite(mesh.uv).all():
            rep.add("UV_NAN", "non-finite UVs", f, mname, k)
        else:
            m = float(np.abs(mesh.uv).max())
            if m > UV_LIMIT:
                rep.add("UV_RANGE", f"UVs reach {m:.0f} (usual range is about -16..16; "
                        f"UV scale for format {mesh.decl} may be wrong)", f, mname, k)
    if model.nodes and 0 <= mesh.node < len(model.nodes):
        M = model.nodes[mesh.node]
        if not np.isfinite(M).all():
            rep.add("BAD_NODE", f"node {mesh.node} transform is not finite", f, mname, k)
        elif mesh.hidden:
            rep.add("HIDDEN_NODE", f"node {mesh.node} has zero scale (object hidden until the game "
                    f"shows it); shown at unit scale", f, mname, k)
        elif not mesh.decl.startswith("skin") and abs(np.linalg.det(M[:3, :3])) < 1e-12:
            rep.add("BAD_NODE", f"node {mesh.node} transform is singular (mesh collapses)",
                    f, mname, k)


def mask_channels(arr):
    """Share of lit mask pixels that are mainly red / green / blue."""
    rgb = arr[..., :3].reshape(-1, 3).astype(np.int32)
    on = rgb[rgb.max(1) > 60]
    if not len(on):
        return 0.0, 0.0, 0.0
    return tuple(float((on[:, i] > 100).mean()) for i in range(3))


def check_material(rep, f, mname, i, mat, model, texidx, missing):
    """missing: dict role-kind -> {hash: material count}, filled here and
    reported once per model by check_model."""
    where = dict(file=f, model=mname, material=i)
    for slot, h, role in mat.slot_roles():
        t = texidx.find(h, model.src)
        if t is None:
            kind = "diffuse" if role == "diffuse" else "other"
            missing[kind][h] = missing[kind].get(h, 0) + 1
            continue
        if not t.fmt:
            rep.add("UNKNOWN_TEX_FMT", f"slot {slot} texture {h:08x} has unknown pixel format id "
                    f"{t.fmt_id}", **where)
            continue
        arr = texidx.decode(t)
        if isinstance(arr, Exception):
            rep.add("TEXTURE_DECODE", f"slot {slot} texture {h:08x} ({t.width}x{t.height} {t.fmt}): "
                    f"{arr}", **where)
    # body / light classification sanity (tinted materials)
    if mat.kind == "tinted" and len(mat.slots) > 3:
        h = mat.slots[3][0]
        t = texidx.find(h, model.src) if h not in (0, 0xFFFFFFFF) else None
        if t is not None and t.fmt:
            arr = texidx.decode(t)
            if not isinstance(arr, Exception):
                r, g, b = mask_channels(arr)
                if mat.is_light and r < 0.05:
                    rep.add("LIGHT_MASK", f"treated as a light/decal part, but its slot-3 mask "
                            f"{h:08x} has no red channel (R {r:.0%}, G {g:.0%}, B {b:.0%}): "
                            f"its glow would render black - flags {mat.flags:08x}", **where)
                elif not mat.is_light and r > 0.5 and g < 0.05 and b < 0.05:
                    rep.add("LIGHT_MASK", f"treated as a body part, but its slot-3 mask {h:08x} is "
                            f"a red light mask - flags {mat.flags:08x}", **where)


def check_skin(rep, f, model):
    if not model.bones:
        return
    known = {h for h, _ in model.bones}
    for mesh in model.meshes:
        if not mesh.palette or mesh.pnmtx is None:
            continue
        pn = (mesh.pnmtx_values if mesh.pnmtx_values is not None else mesh.pnmtx).astype(np.int64)
        slot = pn // 3 - 1
        bad = (pn % 3 != 0) | (slot < 0) | (slot >= len(mesh.palette))
        if bad.any():
            rep.add("BAD_SKIN", f"bone slot(s) {sorted(int(x) for x in slot[bad])[:4]} used outside the "
                    f"{len(mesh.palette)}-bone palette", f, model.name, mesh.index)
        if mesh.pnmtx_conflicts:
            rep.add("BAD_SKIN", f"{mesh.pnmtx_conflicts} vertices are bound to two different bones "
                    "(exported with one of them)", f, model.name, mesh.index)
        missing = [h for h in mesh.palette if h not in known]
        if missing:
            rep.add("BAD_SKIN", f"palette bone(s) {', '.join(f'{h:08x}' for h in missing[:4])} "
                    "not in the skeleton", f, model.name, mesh.index)
    if not model.hierarchy:
        rep.add("NO_BONE_HIERARCHY", f"{len(model.bones)} bones, no hierarchy found", f, model.name)


def check_model(rep, f, model, texidx):
    mname = model.name
    sizes = [c.size for v in model.chunks.values() for c in v]
    if sizes and not any(sizes):
        rep.add("EMPTY_MODEL", "placeholder model with no data", f, mname)
        return 0
    try:
        model.load()
    except Exception as e:  # noqa: BLE001
        rep.add("NO_MESHES", f"model could not be loaded: {e}", f, mname)
        return 0
    expected = model.mesh_count
    for w in model.warnings:
        if w.startswith("mesh "):
            idx, _, msg = w.partition(":")
            rep.add("DECODE_FAILED", msg.strip(), f, mname, idx.replace("mesh ", "").strip())
        elif w.startswith("unknown vertex format") and "decoded as" in w:
            pass  # reported per mesh below
        elif "incomplete" in w:
            rep.add("NO_MESHES", w, f, mname)
        elif "bone palettes" in w:
            rep.add("BAD_SKIN", w, f, mname)
        else:
            rep.add("DECODE_FAILED", w, f, mname)
    if expected and not model.meshes:
        rep.add("NO_MESHES", f"none of {expected} mesh(es) could be decoded", f, mname)
    guessed = Counter(m.decl for m in model.meshes if m.decl.endswith("?") or m.decl == "unknown")
    for decl, n in guessed.items():
        hashes = sorted({w.split()[3].rstrip(",") for w in model.warnings
                         if w.startswith("unknown vertex format")})
        rep.add("GUESSED_FORMAT", f"{n} mesh(es) in unknown vertex format "
                f"{', '.join(hashes) or '?'} decoded as '{decl}' - check visually",
                f, mname)
    check_skin(rep, f, model)
    use = model.material_usage()
    vb = None
    if 0x06 in model.chunks:
        try:
            with ct.open_data(model.src) as fh:
                vb = model._blob(fh, 0x06)
        except OSError:
            pass
    for mesh in model.meshes:
        if mesh.material < 0 and not model.is_proxy:
            code = "NO_MATERIAL" if mesh.uv is not None else "NO_MATERIAL_NOUV"
            rep.add(code, f"mesh has no material ({len(mesh.tris)} triangles, "
                    f"format {mesh.decl}) - renders flat grey", f, mname, mesh.index)
        check_mesh(rep, f, mname, mesh, model, vb)
    missing = {"diffuse": {}, "other": {}}
    for i, mat in enumerate(model.materials):
        check_material(rep, f, mname, i, mat, model, texidx, missing)
    meshes_by_mat = {i: n for i, (n, _t) in use.items()}
    if missing["diffuse"]:
        n_mesh = sum(meshes_by_mat.get(i, 0) for i, mat in enumerate(model.materials)
                     if mat.diffuse in missing["diffuse"])
        rep.add("MISSING_DIFFUSE", f"{n_mesh} mesh(es) use diffuse textures not in the folder: "
                + ", ".join(f"{h:08x} ({n} material{'s' if n > 1 else ''})"
                            for h, n in sorted(missing["diffuse"].items(), key=lambda x: -x[1])), f, mname)
    if missing["other"]:
        rep.add("MISSING_TEXTURE", "other texture slots not in the folder: "
                + ", ".join(f"{h:08x} (x{n})" for h, n in sorted(missing["other"].items(), key=lambda x: -x[1])),
                f, mname)
    for kind in ("diffuse", "other"):
        for h, n in missing[kind].items():
            MISSING_GLOBAL[h][kind] += n
            MISSING_GLOBAL[h]["models"] += 1
    return len(model.meshes)


# ----------------------------------------------------------------------------
class _Tee:
    """Write the report to the console and a file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for st in self.streams:
            st.write(text)

    def flush(self):
        for st in self.streams:
            st.flush()


def _pick_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog
        r = tk.Tk()
        r.withdraw()
        d = filedialog.askdirectory(title="Select the game folder to check")
        r.destroy()
        return d or None
    except Exception:
        return None


def _own_console():
    """True on Windows when the console window belongs to this script alone
    (started by double-click / file association, possibly via the py.exe
    launcher), so it would close as soon as the script exits. False when a
    shell (cmd, PowerShell, Windows Terminal...) shares the console."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        arr = (wintypes.DWORD * 64)()
        n = k32.GetConsoleProcessList(arr, 64)
        if n <= 1:
            return True
        me = os.getpid()
        launchers = {"py.exe", "pyw.exe", "python.exe", "python3.exe", "pythonw.exe"}
        for pid in list(arr)[:min(n, 64)]:
            if pid == me:
                continue
            h = k32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
            if not h:
                return False                                  # unknown process: assume a shell
            try:
                buf = ctypes.create_unicode_buffer(1024)
                size = wintypes.DWORD(1024)
                ok = k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size))
            finally:
                k32.CloseHandle(h)
            name = buf.value.replace("\\", "/").rsplit("/", 1)[-1].lower()
            if not ok or name not in launchers:
                return False
        return True
    except Exception:
        return False


def gather(inputs):
    files = []
    for inp in inputs:
        if os.path.isdir(inp):
            root = os.path.abspath(inp)
            for r, dirs, names in os.walk(root):
                dirs.sort()
                for n in sorted(names):
                    if not n.lower().endswith((".dict", ".png", ".obj", ".mtl", ".txt", ".json")):
                        files.append((os.path.join(r, n), os.path.relpath(os.path.join(r, n), root)))
        else:
            files.append((os.path.abspath(inp), os.path.basename(inp)))
    return files


def main():
    ap = argparse.ArgumentParser(description="Check every model/mesh of Transformers: Cybertron "
                                             "Adventures (Wii) for render problems.")
    ap.add_argument("inputs", nargs="*", help="game folder(s) and/or .data files "
                    "(none: a folder picker opens)")
    ap.add_argument("--out", help="also save the printed report to this text file")
    ap.add_argument("--json", help="also write the full report to this JSON file")
    ap.add_argument("--errors-only", action="store_true", help="list only errors")
    ap.add_argument("--all", action="store_true", help="also list info-level findings")
    ap.add_argument("--verbose", action="store_true", help="also list models with no findings")
    a = ap.parse_args()
    if not a.inputs:
        folder = _pick_folder()
        if not folder:
            ap.print_usage()
            print("No folder given.")
            sys.exit(2)
        a.inputs = [folder]
    if a.out:
        sys.stdout = _Tee(sys.stdout, open(a.out, "w", encoding="utf-8"))

    files = gather(a.inputs)
    rep = Report()
    texidx = TextureIndex()
    print(f"Indexing textures in {len(files)} file(s)...", file=sys.stderr)
    for path, rel in files:
        try:
            texidx.add_file(path)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {rel}: {e}", file=sys.stderr)

    nmodels = nmeshes = 0
    per_model = []
    print("Checking models...", file=sys.stderr)
    for path, rel in files:
        if path.lower().endswith(".data"):
            try:
                with ct.open_data(path) as fh:
                    size = ct.data_size(fh)
                    subs = ct.find_subfiles(fh, size, path)
                if ct.is_compressed_file(path):
                    rep.add("COMPRESSED_FILE", "zlib-compressed container, decompressed "
                            f"to {size:,} bytes", rel)
                    per_model.append((rel, "(file)", rep.items[-1:]))
                if not subs:
                    rep.add("UNREADABLE_FILE", "file layout could not be read (truncated, damaged or "
                            "not a model/texture container)", rel)
                    per_model.append((rel, "(file)", rep.items[-1:]))
                elif subs[0][0] != 0:
                    rep.add("PARTIAL_FILE", f"only the part from offset 0x{subs[0][0]:x} on could be "
                            f"mapped; earlier models/textures may be missing", rel)
                    per_model.append((rel, "(file)", rep.items[-1:]))
            except OSError as e:
                rep.add("UNREADABLE_FILE", f"cannot open: {e}", rel)
                per_model.append((rel, "(file)", rep.items[-1:]))
        try:
            models = cm.scan_models(path, rel)
        except Exception as e:  # noqa: BLE001
            rep.add("NO_MESHES", f"file could not be read: {e}", rel)
            continue
        for m in models:
            before = len(rep.items)
            nmeshes += check_model(rep, rel, m, texidx)
            nmodels += 1
            per_model.append((rel, m.name, rep.items[before:]))
            m.meshes = []          # free memory; the report is all we keep
            m.loaded = False

    # --------------------------------------------------------------- print
    min_rank = 0 if a.errors_only else (2 if a.all else 1)
    shown = 0
    for rel, mname, items in per_model:
        items = [it for it in items if RANK[it["severity"]] <= min_rank]
        if not items and not a.verbose:
            continue
        if not items:
            print(f"OK     {mname}  ({rel})")
            continue
        worst = min(items, key=lambda it: RANK[it["severity"]])["severity"].upper()
        print(f"{worst:6s} {mname if mname != '(file)' else rel}" + (f"  ({rel})" if mname != "(file)" else ""))
        for it in sorted(items, key=lambda it: (RANK[it["severity"]], it["code"])):
            loc = (f"mesh {it['mesh']}" if it["mesh"] is not None else
                   f"mat {it['material']}" if it["material"] is not None else
                   "model" if it["model"] is not None else "file")
            print(f"   {it['severity']:5s} {it['code']:15s} {loc:8s} {it['detail']}")
        shown += 1

    sev = Counter(it["severity"] for it in rep.items)
    codes = Counter(it["code"] for it in rep.items)
    bad_models = {(it["file"], it["model"]) for it in rep.items if it["severity"] == "error"}
    print()
    print(f"Checked {nmodels} model(s), {nmeshes} mesh(es) in {len(files)} file(s); "
          f"{sum(len(v) for v in texidx.by_hash.values())} textures indexed.")
    print(f"Findings: {sev.get('error', 0)} error(s), {sev.get('warn', 0)} warning(s), "
          f"{sev.get('info', 0)} info; {len(bad_models)} model(s) with errors.")
    if codes:
        print("By type: " + ", ".join(f"{c} {n}" for c, n in
                                      sorted(codes.items(), key=lambda x: (RANK[SEVERITY[x[0]]], -x[1]))))
    if MISSING_GLOBAL:
        print("Textures referenced but not found in the folder (most-referenced first):")
        for h, c in sorted(MISSING_GLOBAL.items(), key=lambda x: -(x[1]["diffuse"] + x[1]["other"]))[:15]:
            role = "as diffuse" if c["diffuse"] else "in other slots only"
            print(f"   {h:08x}  {c['diffuse'] + c['other']:4d} material slot(s) in {c['models']} model(s), {role}")
        if len(MISSING_GLOBAL) > 15:
            print(f"   ... and {len(MISSING_GLOBAL) - 15} more (see --json)")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(models=nmodels, meshes=nmeshes, files=len(files), findings=rep.items,
                           missing_textures={f"{h:08x}": dict(c) for h, c in MISSING_GLOBAL.items()}),
                      fh, indent=1)
        print(f"Report written to {a.json}")
    sys.exit(1 if sev.get("error") else 0)


if __name__ == "__main__":
    try:
        main()
    finally:
        if _own_console():
            try:
                sys.stdout.flush()
                input("\nPress Enter to close this window...")
            except EOFError:
                pass
