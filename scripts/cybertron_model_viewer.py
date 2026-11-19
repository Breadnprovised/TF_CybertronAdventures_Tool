#!/usr/bin/env python3
"""
cybertron_model_viewer.py - model viewer for Transformers: Cybertron Adventures (Wii)

Scans the game folder for every model (characters, levels, props, cutscene
objects) and shows them textured in a 3D view.

    python3 cybertron_model_viewer.py [GAME_FOLDER]

Without a folder argument a folder picker opens (the last folder is
remembered). Requires Python 3.8+, numpy, Pillow and pyglet 2.x
(pip install numpy pillow pyglet), with cybertron_tex.py and
cybertron_model.py in the same folder.

Controls
  Up / Down, wheel over list   previous / next model
  PgUp / PgDn, Home / End      jump through the list
  click a row                  select it
  / or Ctrl+F, or click the search box   search the list (Esc clears)
  drag the scroll bar / panel edge       scroll the list / resize the panel
  left drag                    orbit        right/middle drag   pan
  wheel over view              zoom         F                   frame model
  Space                        auto-rotate
  T textures   L lighting   M reflections   V vertex colors   W wireframe
  C invert faces   B background color   G show mask texture   P show proxy models
  K material colours (game / raw / off)   N show duplicate models   I material inspector
  A animations panel ([ and ] step one frame)
  E / Ctrl+E / Export button   export current model (.glb + PNG textures, skeleton + weights)
  F12 screenshot (screenshots folder)   H or the Help button: key list
"""

import json
import math
import os
import queue
import sys
import threading
import time
import ctypes

HERE = os.path.dirname(os.path.abspath(__file__))
# exports/screenshots go next to the launchers when the scripts live in "scripts"
BASE = os.path.dirname(HERE) if os.path.basename(HERE).lower() == "scripts" else HERE
sys.path.insert(0, HERE)

import numpy as np
from PIL import Image, ImageChops

try:
    import pyglet
    from pyglet import gl
    from pyglet.graphics.shader import Shader, ShaderProgram
    from pyglet.window import key, mouse
except ImportError:
    sys.exit("pyglet 2.x is required:  pip install pyglet")

import cybertron_tex as ct
import cybertron_model as cm
import cybertron_anim as ca

CONFIG = os.path.join(os.path.expanduser("~"), ".cybertron_model_viewer.json")
PANEL_W = 330          # default side-panel width (resizable, remembered)
PANEL_MIN, VIEW_MIN = 200, 260
SB_W = 10              # scrollbar width
SEARCH_H = 26
ROW_H = 20
FONT = 10
BGS = [(0.16, 0.17, 0.19), (0.45, 0.47, 0.5), (0.02, 0.02, 0.02), (0.85, 0.86, 0.88)]

MAX_BONES = 128

VERT = """#version 330 core
in vec3 position;
in vec3 normal;
in vec2 uv;
in vec3 color;
in float bone;
uniform mat4 mvp;
uniform mat4 model;
uniform mat4 view;
uniform mat4 bones[128];
uniform int skinned;
out vec3 v_pos;
out vec3 v_vnrm;
out vec3 v_nrm;
out vec2 v_uv;
out vec3 v_col;
void main() {
    vec4 p = vec4(position, 1.0);
    vec3 nn = normal;
    if (skinned == 1) {                 // rigid skinning: one bone per vertex
        mat4 b = bones[int(bone + 0.5)];
        p = b * p;
        nn = mat3(b) * normal;
    }
    vec4 wp = model * p;
    v_pos = wp.xyz;
    v_nrm = mat3(model) * nn;
    v_vnrm = mat3(view) * v_nrm;
    v_uv = uv;
    v_col = color;
    gl_Position = mvp * p;
}
"""

FRAG = """#version 330 core
in vec3 v_pos;
in vec3 v_nrm;
in vec2 v_uv;
in vec3 v_col;
in vec3 v_vnrm;
uniform sampler2D diffuse_tex;
uniform sampler2D glow_tex;
uniform sampler2D spec_tex;
uniform sampler2D env_tex;
uniform sampler2D colour_tex;
uniform int has_colmap;
uniform vec3 mask_r;
uniform vec3 mask_g;
uniform vec3 mask_b;
uniform int unlit;
uniform mat4 view;
uniform int has_tex;
uniform int has_glow;
uniform int has_spec;
uniform int has_env;
uniform int use_env;
uniform int has_nrm;
uniform int has_col;
uniform int use_tex;
uniform int use_light;
uniform int use_glow;
uniform int use_col;
uniform vec3 eye;
uniform vec3 tint;
uniform vec3 mat_tint;
out vec4 frag;
void main() {
    vec4 base = vec4(tint, 1.0);
    if (use_tex == 1 && has_tex == 1) {
        base = texture(diffuse_tex, v_uv);
        if (base.a < 0.5) discard;
    }
    base.rgb *= mat_tint;
    if (has_colmap == 1) {
        vec3 m = texture(colour_tex, v_uv).rgb;            // channel mask
        base.rgb *= m.r * mask_r + m.g * mask_g + m.b * mask_b;
    }
    if (use_col == 1 && has_col == 1) base.rgb *= v_col * 1.15;
    vec3 n;
    if (has_nrm == 1 && length(v_nrm) > 0.01) n = normalize(v_nrm);
    else n = normalize(cross(dFdx(v_pos), dFdy(v_pos)));
    vec3 v = normalize(eye - v_pos);
    if (dot(n, v) < 0.0) n = -n;                    // two-sided lighting
    vec3 rgb = base.rgb;
    if (use_light == 1 && unlit == 0) {
        vec3 l1 = normalize(v + vec3(0.3, 0.6, 0.2));
        vec3 l2 = normalize(vec3(-0.5, 0.3, -0.6));
        float d = 0.38 + 0.62 * max(dot(n, l1), 0.0) + 0.18 * max(dot(n, l2), 0.0);
        float spec = pow(max(dot(n, normalize(l1 + v)), 0.0), 32.0) * 0.18;
        rgb = rgb * d + spec;
    }
    if (use_env == 1 && has_env == 1) {
        // sphere environment map, masked by the specular colour map
        vec3 vn = (has_nrm == 1 && length(v_vnrm) > 0.01) ? normalize(v_vnrm) : mat3(view) * n;
        if (vn.z < 0.0) vn = -vn;
        vec2 euv = vec2(vn.x * 0.5 + 0.5, 0.5 - vn.y * 0.5);
        vec3 mask = (has_spec == 1) ? texture(spec_tex, v_uv).rgb : base.rgb;
        rgb += mask * texture(env_tex, euv).rgb * 0.55;
    }
    if (use_glow == 1 && has_glow == 1) rgb += texture(glow_tex, v_uv).rgb * 0.8;
    frag = vec4(rgb, 1.0);
}
"""


def output_dir(kind):
    """Folder for exports / screenshots: next to this script if writable,
    otherwise in the user's home folder."""
    for base in (BASE, os.path.join(os.path.expanduser("~"), "CybertronTools")):
        d = os.path.join(base, kind)
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, ".write_test")
            open(probe, "w").close()
            os.remove(probe)
            return d
        except OSError:
            continue
    raise OSError(f"no writable folder for {kind}")


# ----------------------------------------------------------------------------
# small matrix helpers (row-major numpy; transposed when uploaded)
# ----------------------------------------------------------------------------
def perspective(fov, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov) / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def look_at(eye, target, up):
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= max(np.linalg.norm(s), 1e-9)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


# The game is Z-up; the viewer is Y-up.
Z_TO_Y = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], np.float32)


def gl_mat(m):
    return tuple(np.asarray(m, np.float32).T.flatten().tolist())


# ----------------------------------------------------------------------------
# Catalogue of everything in the game folder
# ----------------------------------------------------------------------------
def _set_label(lbl, text, color=None):
    """Change a label only if needed: every text change re-lays it out,
    which is the slow part of redrawing the model list."""
    if lbl.text != text:
        lbl.text = text
    if color is not None and tuple(lbl.color) != color:
        lbl.color = color


class Catalogue:
    def __init__(self):
        self.models = []
        self.textures = {}     # hash -> [Texture]
        self.lock = threading.Lock()
        self.files_done = 0
        self.files_total = 0
        self.done = False
        self.stop = False
        self.dups = {}          # model index -> index of the identical copy that is kept
        self.dups_done = False
        self.clips = []         # cybertron_anim.ClipInfo for every animation found

    def find_texture(self, h, prefer_src=None):
        lst = self.textures.get(h)
        if not lst:
            return None
        for t in lst:
            if t.src == prefer_src and t.fmt:
                return t
        for t in lst:
            if t.fmt:
                return t
        return None

    def scan(self, root):
        files = []
        for r, dirs, names in os.walk(root):
            dirs.sort()
            files += [os.path.join(r, n) for n in sorted(names)
                      if not n.lower().endswith((".dict", ".png", ".obj", ".mtl", ".glb"))]
        self.files_total = len(files)
        for path in files:
            if self.stop:
                break
            rel = os.path.relpath(path, root)
            try:
                texs = ct.scan_file(path, allow_full_scan=False, warn=lambda m: None)
                mods = cm.scan_models(path, rel)
            except Exception as e:
                print(f"warning: {rel}: {e}", file=sys.stderr)
                texs, mods = [], []
            try:
                clips = ca.list_clips(path, rel)
            except Exception:
                clips = []
            with self.lock:
                for t in texs:
                    self.textures.setdefault(t.hash, []).append(t)
                self.models.extend(mods)
                self.clips.extend(clips)
            self.files_done += 1
        self.done = True
        if not self.stop:
            try:
                with self.lock:
                    models = list(self.models)
                self.dups = cm.find_duplicates(models, self.find_texture, stop=lambda: self.stop)
            except Exception as e:
                print(f"warning: duplicate check failed: {e}", file=sys.stderr)
                self.dups = {}
        self.dups_done = True


# ----------------------------------------------------------------------------
# GPU resources for one model
# ----------------------------------------------------------------------------
class GpuModel:
    def __init__(self, program, model, tex_arrays):
        """tex_arrays: hash -> HxWx4 uint8 numpy (decoded in a worker)."""
        self.model = model
        self.parts = []
        self.textures = {}
        self.arrays = tex_arrays          # kept for the material panel thumbnails
        for h, arr in tex_arrays.items():
            self.textures[h] = upload_texture(arr)
        node_of = {r[0]: i for i, r in enumerate(model.hierarchy)} if model.hierarchy else {}
        self.skinnable = bool(node_of) and len(node_of) <= MAX_BONES
        for mesh in model.meshes:
            n = len(mesh.pos)
            idx = mesh.tris.ravel().tolist()
            if not idx:
                continue
            bone = np.zeros(n, np.float32)
            skin = False
            if self.skinnable and mesh.palette and mesh.pnmtx is not None:
                slot = mesh.pnmtx.astype(np.int64) // 3 - 1
                lut = np.array([node_of.get(h, -1) for h in mesh.palette] + [-1])
                b = lut[np.where((slot >= 0) & (slot < len(mesh.palette)), slot, len(mesh.palette))]
                skin = bool((b >= 0).any())
                bone = np.maximum(b, 0).astype(np.float32)
            data = dict(
                bone=("f", bone.tolist()),
                position=("f", mesh.pos.ravel().tolist()),
                normal=("f", (mesh.nrm if mesh.nrm is not None else np.zeros((n, 3))).ravel().tolist()),
                uv=("f", (mesh.uv if mesh.uv is not None else np.zeros((n, 2))).ravel().tolist()),
                color=("f", (mesh.col if mesh.col is not None else np.ones((n, 3))).ravel().tolist()),
            )
            vl = program.vertex_list_indexed(n, gl.GL_TRIANGLES, idx, **data)
            mat = model.materials[mesh.material] if 0 <= mesh.material < len(model.materials) else None
            get = lambda h: self.textures.get(h) if (mat and h) else None
            diff = get(mat.diffuse) if mat else None
            spec = get(mat.specular) if mat else None
            env = get(mat.envmap) if mat else None
            glow = get(mat.overlay) if mat and mat.overlay != mat.diffuse else None
            tints = (mat.game_tint, mat.tint) if mat else ((1.0, 1.0, 1.0),) * 2
            colmap = get(mat.colour_map) if mat else None
            mcols = mat.mask_colours if mat else ((1.0, 1.0, 1.0),) * 3
            unlit = bool(mat and mat.is_light)
            self.parts.append((vl, mesh, diff, spec, env, glow, tints, colmap, unlit, mcols, skin))
        lo, hi = model.bounds()
        if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
            lo, hi = np.zeros(3), np.ones(3)
        self.center = (lo + hi) / 2
        self.radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-3)

    def delete(self):
        for vl, *_ in self.parts:
            vl.delete()
        ids = [t for t in self.textures.values() if t]
        if ids:
            arr = (gl.GLuint * len(ids))(*ids)
            gl.glDeleteTextures(len(ids), arr)
        self.parts = []
        self.textures = {}


def upload_texture(arr):
    h, w = arr.shape[:2]
    tid = gl.GLuint()
    gl.glGenTextures(1, ctypes.byref(tid))
    gl.glBindTexture(gl.GL_TEXTURE_2D, tid)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    data = np.ascontiguousarray(arr, np.uint8)
    # row 0 of the image = t 0, which is how the Wii addresses textures
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w, h, 0, gl.GL_RGBA,
                    gl.GL_UNSIGNED_BYTE, data.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
    return tid.value


# ----------------------------------------------------------------------------
# Material inspector (right-hand panel)
# ----------------------------------------------------------------------------
def _thumb_image(arr, size):
    """numpy RGBA -> pyglet image, fitted into size x size over a checkerboard."""
    im = Image.fromarray(np.ascontiguousarray(arr), "RGBA")
    im.thumbnail((size, size), Image.LANCZOS if max(im.size) > size else Image.NEAREST)
    if max(im.size) < size:                       # small textures: scale up crisply
        k = size // max(im.size)
        im = im.resize((im.width * k, im.height * k), Image.NEAREST)
    bg = Image.new("RGBA", im.size, (150, 150, 150, 255))
    sq = max(4, size // 8)
    for yy in range(0, im.height, sq):
        for xx in range(0, im.width, sq):
            if (xx // sq + yy // sq) % 2:
                bg.paste((200, 200, 200, 255), (xx, yy, min(xx + sq, im.width), min(yy + sq, im.height)))
    bg.alpha_composite(im)
    return pyglet.image.ImageData(bg.width, bg.height, "RGBA", bg.tobytes(), pitch=-bg.width * 4)


def _hex(c):
    return "#%02x%02x%02x" % tuple(max(0, min(255, round(v * 255))) for v in c)


class MaterialPanel:
    W = 410
    THUMB = 44
    LINE = 16

    def __init__(self, win):
        self.win = win
        self.visible = False
        self.gpu = None
        self.highlight = None        # material index highlighted in the view
        self.scroll = 0
        self.cards = []              # [(height, material index)]
        self.batch = pyglet.graphics.Batch()
        self.g0, self.g1, self.g2 = (pyglet.graphics.Group(order=i) for i in range(3))
        self.objs = []
        self.hits = []               # (x, y, w, h, action, arg)
        self.thumbs = {}             # (hash, size) -> pyglet image
        self.preview = None          # texture hash shown large
        self.pbatch = pyglet.graphics.Batch()
        self.pobjs = []

    # ------------------------------------------------------------------
    def toggle(self):
        self.visible = not self.visible
        if self.visible:
            self.win.close_panels(self)
        if not self.visible:
            self.highlight = None
            self.preview = None
        self.rebuild()
        self.win.flash("Material inspector " + ("open - click a material to show only its faces" if self.visible else "closed"))

    def set_model(self, gpu):
        self.gpu = gpu
        self.highlight = None
        self.preview = None
        self.scroll = 0
        self.thumbs.clear()
        self.rebuild()

    def on_escape(self):
        if self.preview is not None:
            self.preview = None
            self.rebuild()
            return True
        if self.highlight is not None:
            self.highlight = None
            self.rebuild()
            return True
        return False

    def layout(self):
        self.rebuild()

    @property
    def rect(self):
        w = self.win
        x0 = max(w.panel_w + 260, w.width - self.W)
        return x0, 28, w.width - x0, w.height - 46 - 28

    # ------------------------------------------------------------------
    def _material_lines(self, i, mat, use):
        """Text rows of one card (without the slot rows)."""
        d = mat.describe()
        n, t = use.get(i, (0, 0))
        head = f"mat {i}  ·  {d['kind']}"
        sub = f"{n} mesh{'es' if n != 1 else ''} · {t:,} tris"
        rows = []
        if d["kind"] != "layered":
            rows.append(f"flags 0x{d['flags']:08x}")
            rows += ["   " + note for note in d["notes"] if not note.startswith("bytes")]
            rows.append("params  " + "  ".join(f"{p:g}" for p in d["params"]))
        return head, sub, rows, d

    def _card_height(self, d_rows, d):
        h = 8 + 20 + len(d_rows) * self.LINE
        if d["kind"] != "layered":
            h += 22
        if d["stored_colour"] is not None:
            h += 20
        h += sum(self.THUMB + 6 for sl in d["slots"]) + 8
        return h

    def rebuild(self):
        for o in self.objs + self.pobjs:
            o.delete()
        self.objs, self.pobjs, self.hits = [], [], []
        if not self.visible:
            return
        B, G0, G1, G2 = self.batch, self.g0, self.g1, self.g2
        x0, y0, pw, ph = self.rect
        top = y0 + ph
        add = self.objs.append
        add(pyglet.shapes.Rectangle(x0, y0, pw, ph, color=(22, 24, 28, 238), batch=B, group=G0))
        add(pyglet.shapes.Rectangle(x0, y0, 2, ph, color=(60, 64, 72, 255), batch=B, group=G1))
        gpu = self.gpu
        if gpu is None or not gpu.model.materials:
            add(pyglet.text.Label("No materials" if gpu else "No model loaded", x=x0 + 14, y=top - 24,
                                  font_size=FONT, color=(190, 190, 190, 255), batch=B, group=G2))
            return
        m = gpu.model
        use = m.material_usage()
        cards = []
        for i, mat in enumerate(m.materials):
            head, sub, rows, d = self._material_lines(i, mat, use)
            cards.append((self._card_height(rows, d), i, mat, head, sub, rows, d))
        total = sum(c[0] for c in cards) + 30
        self.scroll = max(0, min(self.scroll, max(0, total - ph)))
        # title
        y = top + self.scroll - 6
        add(pyglet.text.Label(f"Materials ({len(cards)})  ·  {m.mat_style} style", x=x0 + 14, y=y - 14,
                              font_size=FONT + 1, weight="bold", color=(235, 235, 235, 255), batch=B, group=G2))
        y -= 26
        chars = int((pw - 30) / self.win.char_w)
        for ch, i, mat, head, sub, rows, d in cards:
            ctop, cbot = y, y - ch
            y = cbot
            if cbot > top or ctop < y0:       # off screen
                continue
            sel = self.highlight == i
            add(pyglet.shapes.Rectangle(x0 + 8, cbot + 3, pw - 16, ch - 6,
                                        color=(48, 70, 104, 255) if sel else (34, 37, 43, 255), batch=B, group=G0))
            self.hits.append((x0 + 8, cbot + 3, pw - 16, ch - 6, "card", i))
            cy = ctop - 22
            add(pyglet.text.Label(head, x=x0 + 18, y=cy, font_size=FONT, weight="bold",
                                  color=(255, 255, 255, 255), batch=B, group=G2))
            add(pyglet.text.Label(sub, x=x0 + pw - 18, y=cy, font_size=FONT - 1, anchor_x="right",
                                  color=(170, 180, 195, 255), batch=B, group=G2))
            cy -= 4
            for r in rows:
                cy -= self.LINE
                add(pyglet.text.Label(r[:chars], x=x0 + 18, y=cy, font_size=FONT - 1,
                                      color=(205, 205, 210, 255), batch=B, group=G2))
            # colours
            if d["kind"] != "layered":
                cy -= 22
                xx = x0 + 18
                def swatch(xx, c, text):
                    add(pyglet.shapes.Rectangle(xx, cy - 2, 16, 16, color=(*[round(v * 255) for v in c], 255), batch=B, group=G1))
                    add(pyglet.text.Label(text, x=xx + 22, y=cy + 2, font_size=FONT - 1,
                                          color=(220, 220, 220, 255), batch=B, group=G2))
                xx = x0 + 18
                if d["stored_colour"] is not None:
                    swatch(xx, d["stored_colour"], f"colour 1 {_hex(d['stored_colour'])}")
                    swatch(xx + 140, d["colour2"], f"colour 2 {_hex(d['colour2'])}")
                    cy -= 20
                if d["light"]:
                    r, _g, b_ = mat.mask_colours
                    swatch(xx, r, f"mask R → {_hex(r)}")
                    swatch(xx + 140, b_, f"mask B → {_hex(b_)}  (unlit)")
                else:
                    swatch(xx, d["shown_colour"], "shown: skin untinted" if d["kind"] == "tinted" else "shown: untinted")
            # texture slots
            cy -= 6
            for sl in d["slots"]:
                cy -= self.THUMB + 6
                tx_ = x0 + 18
                h = sl["hash"]
                arr = gpu.arrays.get(h) if not sl["empty"] else None
                tex = self.win.cat.find_texture(h, m.src) if not sl["empty"] else None
                if arr is not None:
                    key_ = (h, self.THUMB)
                    if key_ not in self.thumbs:
                        self.thumbs[key_] = _thumb_image(arr, self.THUMB)
                    img = self.thumbs[key_]
                    add(pyglet.sprite.Sprite(img, x=tx_ + (self.THUMB - img.width) // 2,
                                             y=cy + (self.THUMB - img.height) // 2, batch=B, group=G2))
                    self.hits.append((tx_, cy, self.THUMB, self.THUMB, "thumb", h))
                else:
                    add(pyglet.shapes.Rectangle(tx_, cy, self.THUMB, self.THUMB, color=(55, 58, 64, 255), batch=B, group=G1))
                    add(pyglet.text.Label("—" if sl["empty"] else "?", x=tx_ + self.THUMB // 2, y=cy + self.THUMB // 2,
                                          anchor_x="center", anchor_y="center", font_size=FONT + 2,
                                          color=(150, 150, 155, 255), batch=B, group=G2))
                line1 = f"slot {sl['slot']} · {sl['role']}" + ("" if sl["empty"] else f" · {h:08x}")
                if sl["empty"]:
                    line2 = "empty"
                elif tex:
                    line2 = f"{tex.width}×{tex.height} {tex.fmt} · {tex.mips} mip · flags {sl['flags']:08x}"
                else:
                    line2 = f"not in game folder · flags {sl['flags']:08x}"
                lc = int((pw - self.THUMB - 46) / self.win.char_w)
                add(pyglet.text.Label(line1[:lc], x=tx_ + self.THUMB + 10, y=cy + self.THUMB - 16,
                                      font_size=FONT - 1, color=(235, 235, 235, 255), batch=B, group=G2))
                add(pyglet.text.Label(line2[:lc], x=tx_ + self.THUMB + 10, y=cy + 6, font_size=FONT - 1,
                                      color=(165, 170, 180, 255) if tex or sl["empty"] else (220, 150, 120, 255),
                                      batch=B, group=G2))
        # scroll bar
        if total > ph:
            th = max(30, int(ph * ph / total))
            ty = top - th - int((ph - th) * self.scroll / max(1, total - ph))
            add(pyglet.shapes.Rectangle(x0 + pw - 6, ty, 4, th, color=(110, 115, 125, 255), batch=B, group=G2))
        self._build_preview()

    def _build_preview(self):
        h = self.preview
        if h is None or self.gpu is None or h not in self.gpu.arrays:
            return
        w = self.win
        x0, y0, pw, ph = self.rect
        area_l, area_r = w.panel_w, x0
        size = int(min(512, area_r - area_l - 60, w.height - 140))
        if size < 64:
            return
        arr = self.gpu.arrays[h]
        key_ = (h, size)
        if key_ not in self.thumbs:
            self.thumbs[key_] = _thumb_image(arr, size)
        img = self.thumbs[key_]
        cx, cy = (area_l + area_r) // 2, w.height // 2
        B = self.pbatch
        self.pobjs.append(pyglet.shapes.Rectangle(area_l, 0, area_r - area_l, w.height, color=(0, 0, 0, 170), batch=B, group=self.g0))
        self.pobjs.append(pyglet.shapes.Rectangle(cx - img.width // 2 - 8, cy - img.height // 2 - 50,
                                                  img.width + 16, img.height + 66, color=(30, 32, 37, 255), batch=B, group=self.g1))
        self.pobjs.append(pyglet.sprite.Sprite(img, x=cx - img.width // 2, y=cy - img.height // 2, batch=B, group=self.g2))
        tex = w.cat.find_texture(h, self.gpu.model.src)
        cap = f"{h:08x}" + (f"  ·  {tex.width}×{tex.height} {tex.fmt}  ·  {tex.mips} mip(s)  ·  {os.path.basename(tex.src)}" if tex else "")
        self.pobjs.append(pyglet.text.Label(cap, x=cx, y=cy - img.height // 2 - 20,
                                            anchor_x="center", font_size=FONT, color=(230, 230, 230, 255),
                                            batch=B, group=self.g2))
        self.pobjs.append(pyglet.text.Label("click another thumbnail to switch  ·  click elsewhere or Esc to close",
                                            x=cx, y=cy - img.height // 2 - 38, anchor_x="center", font_size=FONT - 1,
                                            color=(160, 165, 175, 255), batch=B, group=self.g2))

    # ------------------------------------------------------------------
    def draw(self, scale):
        if not self.visible:
            return
        x0, y0, pw, ph = self.rect
        if self.preview is not None:
            self.pbatch.draw()
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(int(x0 * scale), int(y0 * scale), int(pw * scale) + 1, int(ph * scale) + 1)
        self.batch.draw()
        gl.glDisable(gl.GL_SCISSOR_TEST)

    def on_press(self, x, y, button):
        if not self.visible:
            return False
        if self.preview is not None:
            # another thumbnail -> show that one; the same one or anywhere else -> close
            hit = next((arg for hx, hy, hw, hh, act, arg in self.hits
                        if act == "thumb" and hx <= x <= hx + hw and hy <= y <= hy + hh), None)
            self.preview = hit if (hit is not None and hit != self.preview) else None
            self.rebuild()
            return True
        x0, y0, pw, ph = self.rect
        if not (x0 <= x <= x0 + pw and y0 <= y <= y0 + ph):
            return False
        if button != mouse.LEFT:
            return True
        for hx, hy, hw, hh, act, arg in reversed(self.hits):   # thumbs before cards
            if hx <= x <= hx + hw and hy <= y <= hy + hh:
                if act == "thumb":
                    self.preview = arg
                elif act == "card":
                    self.highlight = None if self.highlight == arg else arg
                    if self.highlight is not None:
                        n, t = self.gpu.model.material_usage().get(arg, (0, 0))
                        self.win.flash(f"Showing only mat {arg}: {n} mesh(es), {t:,} triangles (click again or Esc to show all)")
                self.rebuild()
                return True
        return True

    def on_scroll(self, x, y, sy):
        if not self.visible:
            return False
        x0, y0, pw, ph = self.rect
        if not (x0 <= x <= x0 + pw and y0 <= y <= y0 + ph):
            return False
        self.scroll = max(0, self.scroll - int(sy * 40))
        self.rebuild()
        return True


class AnimPanel:
    """Side panel listing the animations that fit the current model, with
    playback controls. The pose is applied on the GPU (rigid skinning)."""
    W = 410
    ROW = 20
    CTRL_H = 112

    def __init__(self, win):
        self.win = win
        self.visible = False
        self.gpu = None
        self.items = []              # [(ClipInfo, kind)]
        self.cur = None              # index into items
        self.animator = None
        self.time = 0.0              # seconds
        self.playing = False
        self.loop = True
        self.speed = 1.0
        self.additive = False
        self.layer = None            # (local pose, orientation, clip name) the next clip is layered on
        self.scroll = 0
        self.batch = pyglet.graphics.Batch()
        self.g0, self.g1, self.g2 = (pyglet.graphics.Group(order=i) for i in range(3))
        self.objs, self.hits = [], []
        self.dyn = []                # timeline objects updated every frame
        self.dragging = False
        self.sb_drag = None          # grab offset inside the thumb while dragging the scroll bar
        self._sb = None              # (list_top, avail, total, thumb_h, thumb_y) of the list scroll bar
        self._last_rebuild_frame = None

    @property
    def rect(self):
        w = self.win
        x0 = max(w.panel_w + 260, w.width - self.W)
        return x0, 28, w.width - x0, w.height - 46 - 28

    # ------------------------------------------------------------------
    def toggle(self):
        self.visible = not self.visible
        if self.visible:
            self.win.close_panels(self)
        self.rebuild()
        if self.visible:
            n = len(self.items)
            self.win.flash(f"Animations: {n} fit this model - click one to play" if n else
                           "Animations: none found for this model in the scanned folder")
        else:
            self.win.flash("Animations closed (model shown in its bind pose)")

    def set_model(self, gpu):
        self.gpu = gpu
        self.cur = None
        self.animator = None
        self.layer = None
        self.playing = False
        self.time = 0.0
        self.scroll = 0
        self.items = []
        if gpu is not None and gpu.skinnable:
            m = gpu.model
            with self.win.cat.lock:
                clips = list(self.win.cat.clips)
            for info in clips:
                kind = ca.match(m, info)
                if kind:
                    self.items.append((info, kind))
            src = m.src
            self.items.sort(key=lambda it: (it[0].src != src, it[1] != "same", it[0].rel.lower(), it[0].sub_index, it[0].order))
        self.rebuild()

    def refresh_list(self):
        """New clips were scanned: rebuild the list for the current model."""
        cur = self.items[self.cur][0] if self.cur is not None else None
        t, playing = self.time, self.playing
        anim = self.animator
        self.set_model(self.gpu)
        if cur is not None:
            for i, (info, _k) in enumerate(self.items):
                if info is cur:
                    self.cur, self.animator, self.time, self.playing = i, anim, t, playing
        self.rebuild()

    def on_escape(self):
        return False

    def layout(self):
        self.rebuild()

    # ------------------------------------------------------------------
    @property
    def clip(self):
        return self.animator.clip if self.animator else None

    def frames(self):
        c = self.clip
        return max(c.frames, 1) if c else 1

    def frame(self):
        return min(self.time * ca.FPS, self.frames() - 1)

    def matrices(self):
        """Skinning matrices for the current frame, or None (bind pose)."""
        if not self.visible or self.animator is None or self.animator.clip is None:
            return None
        return self.animator.pose(self.frame())

    def play(self, i):
        info, kind = self.items[i]
        try:
            clip = ca.load_clip(info)
        except Exception as ex:          # noqa: BLE001
            self.win.flash(f"Could not read {info.name}: {ex}", error=True)
            return
        if clip is None:
            self.win.flash(f"Could not read {info.name}", error=True)
            return
        if self.animator is None:
            self.animator = ca.Animator(self.gpu.model)
        A = self.animator
        A.set_base(None)
        A.set_clip(clip, info, kind)
        self.cur, self.time, self.playing = i, 0.0, True
        self._apply_layer()
        how = "this model's own rig" if kind == "same" else "shared rig, matched by bone"
        self.win.flash(f"{info.name}: {clip.frames} frames ({clip.frames / ca.FPS:.1f} s) from {info.rel} - {how}")
        self.rebuild()

    def _apply_layer(self):
        A = self.animator
        if A is None or A.clip is None:
            return
        if self.additive:
            # additive clips are always layered on the bind pose
            A.set_base(A.bind_locals())
            A.G = np.eye(4)
        else:
            A.set_base(None)
            A.G = A._centre_from(A._orientation())

    def stop(self):
        self.cur, self.playing, self.time = None, False, 0.0
        self.layer = None
        if self.animator:
            self.animator.set_base(None)
            self.animator.set_clip(None, None, None)
            self.animator.clip = None
        self.rebuild()

    def tick(self, dt):
        if not (self.visible and self.playing and self.clip):
            return
        length = self.frames() / ca.FPS
        self.time += dt * self.speed
        if self.time >= length - 1e-6:
            if self.loop:
                self.time %= max(length, 1e-6)
            else:
                self.time = (self.frames() - 1) / ca.FPS
                self.playing = False
                self.rebuild()
        self._update_timeline()

    def step(self, n):
        if not self.clip:
            return
        self.playing = False
        f = int(round(self.frame())) + n
        self.time = (f % self.frames()) / ca.FPS
        self.rebuild()

    # ------------------------------------------------------------------
    def _timeline_rect(self):
        x0, y0, pw, ph = self.rect
        top = y0 + ph
        return x0 + 16, top - 84, pw - 32, 14

    def _update_timeline(self):
        for o in self.dyn:
            o.delete()
        self.dyn = []
        if not self.visible:
            return
        tx, ty, tw, th = self._timeline_rect()
        B = self.batch
        n = self.frames()
        f = self.frame()
        frac = f / max(n - 1, 1) if self.clip else 0
        self.dyn.append(pyglet.shapes.Rectangle(tx, ty, max(1, int(tw * frac)), th, color=(70, 120, 190, 255), batch=B, group=self.g1))
        self.dyn.append(pyglet.shapes.Rectangle(tx + int(tw * frac) - 2, ty - 3, 4, th + 6, color=(235, 235, 235, 255), batch=B, group=self.g2))
        if self.clip:
            txt = f"frame {int(f) + 1} / {n}   ·   {self.time:.2f} / {n / ca.FPS:.2f} s"
            if self.additive:
                txt += "   ·   additive on bind pose"
        else:
            txt = "bind pose - pick an animation below"
        self.dyn.append(pyglet.text.Label(txt, x=tx, y=ty - 18, font_size=FONT - 1, color=(190, 195, 205, 255),
                                          batch=B, group=self.g2))

    def rebuild(self):
        for o in self.objs + self.dyn:
            o.delete()
        self.objs, self.hits, self.dyn = [], [], []
        if not self.visible:
            return
        B, G0, G1, G2 = self.batch, self.g0, self.g1, self.g2
        x0, y0, pw, ph = self.rect
        top = y0 + ph
        add = self.objs.append
        add(pyglet.shapes.Rectangle(x0, y0, pw, ph, color=(22, 24, 28, 238), batch=B, group=G0))
        add(pyglet.shapes.Rectangle(x0, y0, 2, ph, color=(60, 64, 72, 255), batch=B, group=G1))
        chars = int((pw - 40) / self.win.char_w)
        gpu = self.gpu
        if gpu is None:
            add(pyglet.text.Label("No model loaded", x=x0 + 14, y=top - 24, font_size=FONT,
                                  color=(190, 190, 190, 255), batch=B, group=G2))
            return
        cur = self.items[self.cur][0] if self.cur is not None else None
        title = f"Animations ({len(self.items)})" + (f"  ·  {cur.name}" if cur else "")
        add(pyglet.text.Label(title[:chars], x=x0 + 14, y=top - 20, font_size=FONT + 1, weight="bold",
                              color=(235, 235, 235, 255), batch=B, group=G2))
        # transport buttons
        bx, by = x0 + 14, top - 54
        for name, text, w_ in (("play", "Pause" if self.playing else "Play", 64), ("prev", "<", 30), ("next", ">", 30),
                               ("loop", "Loop", 50),
                               ("speed", f"{self.speed:g}x", 44), ("add", "Additive", 74), ("bind", "Bind pose", 78)):
            on = (name == "loop" and self.loop) or (name == "play" and self.playing) or (name == "add" and self.additive)
            add(pyglet.shapes.Rectangle(bx, by, w_, 22, color=(52, 80, 120, 255) if on else (48, 52, 60, 255), batch=B, group=G1))
            add(pyglet.text.Label(text, x=bx + w_ // 2, y=by + 11, anchor_x="center", anchor_y="center",
                                  font_size=FONT - 1, color=(235, 235, 235, 255), batch=B, group=G2))
            self.hits.append((bx, by, w_, 22, name, None))
            bx += w_ + 5
        tx, ty, tw, th = self._timeline_rect()
        add(pyglet.shapes.Rectangle(tx, ty, tw, th, color=(45, 48, 55, 255), batch=B, group=G0))
        self.hits.append((tx, ty - 6, tw, th + 12, "timeline", None))
        self._update_timeline()
        # list
        list_top = top - self.CTRL_H
        add(pyglet.shapes.Rectangle(x0 + 8, list_top + 4, pw - 16, 1, color=(60, 64, 72, 255), batch=B, group=G1))
        if not gpu.skinnable:
            msg = ["This model has no skeleton."] if not gpu.model.hierarchy else ["Skeleton too large to animate here."]
            for k, line in enumerate(msg):
                add(pyglet.text.Label(line, x=x0 + 14, y=list_top - 20 - k * 16, font_size=FONT - 1,
                                      color=(190, 190, 190, 255), batch=B, group=G2))
            return
        if not self.items:
            msg = ["No animations for this model in the scanned folder.",
                   "Character animations are in art/characters",
                   "(*_shared, *_player and the character's own file)",
                   "and art/NIS (transformations, cutscenes)."]
            for k, line in enumerate(msg):
                add(pyglet.text.Label(line[:chars + 4], x=x0 + 14, y=list_top - 20 - k * 16, font_size=FONT - 1,
                                      color=(190, 190, 190, 255), batch=B, group=G2))
            return
        rows = []
        last = None
        for i, (info, kind) in enumerate(self.items):
            grp = (info.rel, info.sub_index)
            if grp != last:
                name = info.rel.replace("\\", "/").rsplit("/", 1)[-1]
                label = ("own rig  ·  " if kind == "same" else "shared rig  ·  ") + name
                rows.append(("head", label, None))
                last = grp
            rows.append(("clip", info, i))
        total = len(rows) * self.ROW + 10
        avail = list_top - y0
        self.scroll = max(0, min(self.scroll, max(0, total - avail)))
        y = list_top + self.scroll
        for kind_, a, i in rows:
            y -= self.ROW
            if y > list_top - self.ROW + 2 or y < y0:
                continue
            if kind_ == "head":
                add(pyglet.text.Label(a[:int(chars * 0.92)], x=x0 + 12, y=y + 5, font_size=FONT - 1, weight="bold",
                                      color=(150, 175, 210, 255), batch=B, group=G2))
                continue
            sel = i == self.cur
            if sel:
                add(pyglet.shapes.Rectangle(x0 + 8, y, pw - 16, self.ROW, color=(52, 101, 164, 255), batch=B, group=G1))
            secs = f"{a.frames} f · {a.seconds:.1f} s"
            add(pyglet.text.Label(a.name[:max(8, chars - len(secs) - 2)], x=x0 + 22, y=y + 5, font_size=FONT,
                                  color=(255, 255, 255, 255) if sel else (215, 215, 215, 255), batch=B, group=G2))
            add(pyglet.text.Label(secs, x=x0 + pw - 18, y=y + 5, anchor_x="right", font_size=FONT - 1,
                                  color=(160, 165, 175, 255), batch=B, group=G2))
            self.hits.append((x0 + 8, y, pw - 16, self.ROW, "clip", i))
        self._sb = None
        if total > avail:
            th_ = max(30, int(avail * avail / total))
            ty_ = list_top - th_ - int((avail - th_) * self.scroll / max(1, total - avail))
            sx = x0 + pw - 10
            add(pyglet.shapes.Rectangle(sx, y0, 8, avail, color=(36, 39, 45, 255), batch=B, group=G1))
            col = (150, 156, 168, 255) if self.sb_drag is not None else (110, 115, 125, 255)
            add(pyglet.shapes.Rectangle(sx + 1, ty_, 6, th_, color=col, batch=B, group=G2))
            self._sb = (list_top, avail, total, th_, ty_)
            # checked before the clip rows, which reach under the bar
            self.hits.insert(0, (sx - 4, y0, 14, avail, "sbar", None))

    # ------------------------------------------------------------------
    def draw(self, scale):
        if not self.visible:
            return
        x0, y0, pw, ph = self.rect
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(int(x0 * scale), int(y0 * scale), int(pw * scale) + 1, int(ph * scale) + 1)
        self.batch.draw()
        gl.glDisable(gl.GL_SCISSOR_TEST)

    def _scrub(self, x):
        tx, ty, tw, th = self._timeline_rect()
        if not self.clip:
            return
        frac = max(0.0, min(1.0, (x - tx) / max(tw, 1)))
        self.time = frac * (self.frames() - 1) / ca.FPS
        self.playing = False
        self._update_timeline()

    def on_press(self, x, y, button):
        if not self.visible:
            return False
        x0, y0, pw, ph = self.rect
        if not (x0 <= x <= x0 + pw and y0 <= y <= y0 + ph):
            return False
        if button != mouse.LEFT:
            return True
        for hx, hy, hw, hh, act, arg in self.hits:
            if hx <= x <= hx + hw and hy <= y <= hy + hh:
                if act == "clip":
                    self.play(arg)
                elif act == "play":
                    if self.clip:
                        if not self.playing and self.frame() >= self.frames() - 1 and not self.loop:
                            self.time = 0.0
                        self.playing = not self.playing
                    elif self.items:
                        self.play(self.cur if self.cur is not None else 0)
                        return True
                elif act == "prev":
                    self.step(-1)
                    return True
                elif act == "next":
                    self.step(1)
                    return True
                elif act == "loop":
                    self.loop = not self.loop
                elif act == "add":
                    self.additive = not self.additive
                    self._apply_layer()
                    if self.additive:
                        self.win.flash("Additive: clips are layered on the bind pose - only the bones they "
                                       "animate move, by how much they change from the clip's first frame")
                    else:
                        self.win.flash("Additive off: clips play as full poses")
                elif act == "speed":
                    sp = [0.25, 0.5, 1.0, 2.0]
                    self.speed = sp[(sp.index(self.speed) + 1) % len(sp)] if self.speed in sp else 1.0
                elif act == "bind":
                    self.stop()
                    return True
                elif act == "sbar":
                    if self._sb:
                        _lt, _av, _tot, th_, ty_ = self._sb
                        # grab the thumb where it was clicked; a click on the track centres it there
                        self.sb_drag = (y - ty_) if ty_ <= y <= ty_ + th_ else th_ / 2
                        self._scroll_to_thumb(y)
                    return True
                elif act == "timeline":
                    self.dragging = True
                    self._scrub(x)
                    return True
                self.rebuild()
                return True
        return True

    def _scroll_to_thumb(self, y):
        """Scroll so the thumb's grabbed point sits at mouse height y."""
        list_top, avail, total, th_, _ty = self._sb
        travel = max(1, avail - th_)
        frac = (list_top - th_ - (y - self.sb_drag)) / travel
        self.scroll = int(round(max(0.0, min(1.0, frac)) * (total - avail)))
        self.rebuild()

    def on_drag(self, x, y):
        if self.sb_drag is not None and self._sb:
            self._scroll_to_thumb(y)
            return True
        if self.dragging:
            self._scrub(x)
            return True
        return False

    def on_release(self):
        if self.sb_drag is not None:
            self.sb_drag = None
            self.rebuild()
            return True
        if self.dragging:
            self.dragging = False
            self.rebuild()
            return True
        return False

    def on_scroll(self, x, y, sy):
        if not self.visible:
            return False
        x0, y0, pw, ph = self.rect
        if not (x0 <= x <= x0 + pw and y0 <= y <= y0 + ph):
            return False
        self.scroll = max(0, self.scroll - int(sy * 40))
        self.rebuild()
        return True


class SettingsPanel:
    """Side panel with export options (saved between sessions)."""
    W = 410
    OPTIONS = (("export_skeleton", "Export Skeleton", True,
                "Bones and vertex weights (characters, vehicles). Off: meshes only, in the bind pose."),
               ("export_animations", "Export Animations", False,
                "Every animation in the Animations list, one glTF animation each - Blender imports "
                "them as actions. Needs the skeleton."))

    def __init__(self, win):
        self.win = win
        self.visible = False
        self.batch = pyglet.graphics.Batch()
        self.g0, self.g1, self.g2 = (pyglet.graphics.Group(order=i) for i in range(3))
        self.objs, self.hits = [], []
        self.values = {k: bool(win.cfg.get(k, d)) for k, _t, d, _h in self.OPTIONS}

    def get(self, k):
        return self.values[k]

    @property
    def rect(self):
        w = self.win
        x0 = max(w.panel_w + 260, w.width - self.W)
        return x0, 28, w.width - x0, w.height - 46 - 28

    def toggle(self):
        self.visible = not self.visible
        if self.visible:
            self.win.close_panels(self)
        self.rebuild()
        self.win.flash("Settings open" if self.visible else "Settings closed")

    def layout(self):
        self.rebuild()

    def set_model(self, gpu):
        self.rebuild()

    def _save(self):
        self.win.cfg.update(self.values)
        try:
            cfg = json.load(open(CONFIG)) if os.path.exists(CONFIG) else {}
            cfg.update(self.values)
            json.dump(cfg, open(CONFIG, "w"))
        except Exception:
            pass

    def _wrap(self, text, chars):
        lines, cur = [], ""
        for word in text.split():
            if cur and len(cur) + 1 + len(word) > chars:
                lines.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}".strip()
        if cur:
            lines.append(cur)
        return lines

    def rebuild(self):
        for o in self.objs:
            o.delete()
        self.objs, self.hits = [], []
        if not self.visible:
            return
        B, G0, G1, G2 = self.batch, self.g0, self.g1, self.g2
        x0, y0, pw, ph = self.rect
        top = y0 + ph
        add = self.objs.append
        add(pyglet.shapes.Rectangle(x0, y0, pw, ph, color=(22, 24, 28, 238), batch=B, group=G0))
        add(pyglet.shapes.Rectangle(x0, y0, 2, ph, color=(60, 64, 72, 255), batch=B, group=G1))
        add(pyglet.text.Label("Settings", x=x0 + 14, y=top - 20, font_size=FONT + 1, weight="bold",
                              color=(235, 235, 235, 255), batch=B, group=G2))
        add(pyglet.text.Label("Export (E)", x=x0 + 14, y=top - 48, font_size=FONT - 1, weight="bold",
                              color=(150, 175, 210, 255), batch=B, group=G2))
        chars = max(20, int((pw - 60) / (self.win.char_w * 0.92)))
        y = top - 62
        gpu = self.win.gpu
        for k, title, _d, help_ in self.OPTIONS:
            on = self.values[k]
            enabled = not (k == "export_animations" and not self.values["export_skeleton"])
            y -= 26
            box = (52, 101, 164, 255) if on and enabled else ((70, 74, 82, 255) if enabled else (45, 48, 54, 255))
            add(pyglet.shapes.Rectangle(x0 + 16, y + 3, 16, 16, color=(90, 96, 108, 255), batch=B, group=G1))
            add(pyglet.shapes.Rectangle(x0 + 17, y + 4, 14, 14, color=box, batch=B, group=G1))
            if on:
                add(pyglet.shapes.Rectangle(x0 + 20, y + 7, 8, 8, batch=B, group=G2,
                                            color=(235, 240, 250, 255) if enabled else (120, 122, 128, 255)))
            add(pyglet.text.Label(title, x=x0 + 42, y=y + 6, font_size=FONT,
                                  color=(235, 235, 235, 255) if enabled else (130, 132, 138, 255), batch=B, group=G2))
            self.hits.append((x0 + 8, y, pw - 16, 24, k))
            note = help_
            if k == "export_animations":
                if not enabled:
                    note = "Turn on Export Skeleton to export animations."
                elif gpu is not None:
                    n = len(self.win.animpanel.items) if gpu.skinnable else 0
                    note += f" ({n} for the current model.)" if n else " (None for the current model.)"
            for line in self._wrap(note, chars):
                y -= 16
                add(pyglet.text.Label(line, x=x0 + 42, y=y + 4, font_size=FONT - 2,
                                      color=(160, 165, 175, 255), batch=B, group=G2))
            y -= 8

    def draw(self, scale):
        if not self.visible:
            return
        x0, y0, pw, ph = self.rect
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(int(x0 * scale), int(y0 * scale), int(pw * scale) + 1, int(ph * scale) + 1)
        self.batch.draw()
        gl.glDisable(gl.GL_SCISSOR_TEST)

    def on_press(self, x, y, button):
        if not self.visible:
            return False
        x0, y0, pw, ph = self.rect
        if not (x0 <= x <= x0 + pw and y0 <= y <= y0 + ph):
            return False
        if button == mouse.LEFT:
            for hx, hy, hw, hh, k in self.hits:
                if hx <= x <= hx + hw and hy <= y <= hy + hh:
                    if k == "export_animations" and not self.values["export_skeleton"]:
                        self.win.flash("Export Animations needs Export Skeleton", error=True)
                        return True
                    self.values[k] = not self.values[k]
                    self._save()
                    self.rebuild()
                    break
        return True

    def on_scroll(self, x, y, sy):
        if not self.visible:
            return False
        x0, y0, pw, ph = self.rect
        return x0 <= x <= x0 + pw and y0 <= y <= y0 + ph


# ----------------------------------------------------------------------------
# Viewer window
# ----------------------------------------------------------------------------
class Viewer(pyglet.window.Window):
    def __init__(self, root, auto_shots=None):
        cfg = pyglet.gl.Config(double_buffer=True, depth_size=24, major_version=3, minor_version=3,
                               forward_compatible=True, sample_buffers=1, samples=4)
        try:
            super().__init__(1400, 860, caption="Cybertron Adventures Model Viewer",
                             resizable=True, config=cfg)
        except pyglet.window.NoSuchConfigException:
            super().__init__(1400, 860, caption="Cybertron Adventures Model Viewer",
                             resizable=True)
        self.root = root
        self.cat = Catalogue()
        self.program = ShaderProgram(Shader(VERT, "vertex"), Shader(FRAG, "fragment"))

        self.flags = dict(tex=True, light=True, glow=False, env=True, col=True, wire=False, cull=False,
                          proxies=False, spin=False, help=False, nodup=True)
        self.bg = 0
        self.tint_mode = 0         # 0 game colours, 1 raw stored colours, 2 off
        self.vis_rows = []          # indices into cat.models
        self.sel = -1              # row in visible
        self.scroll = 0
        self.filter = ""

        self.gpu = None            # GpuModel being shown
        self.prefetch = []         # model indices the worker decodes when idle
        self.current_mi = None     # index (in cat.models) of the model shown
        self.loading = None        # model index being loaded
        self.load_gen = 0
        self.results = queue.Queue()
        self.cam = dict(yaw=35.0, pitch=18.0, dist=10.0, target=np.zeros(3, np.float32))
        self.status = "Scanning…"
        self.msg = ""
        self.msg_time = 0

        try:
            self.cfg = json.load(open(CONFIG))
        except Exception:
            self.cfg = {}
        self.panel_w = int(self.cfg.get("panel_width", PANEL_W))
        self.search_focus = False
        self.drag = None           # "resize" | "thumb" | "view" | None
        self.thumb_grab = 0
        self.hover_div = False

        self.ui = pyglet.graphics.Batch()
        self.ui_bg = pyglet.graphics.Group(order=0)
        self.ui_mid = pyglet.graphics.Group(order=1)
        self.ui_fg = pyglet.graphics.Group(order=2)
        B, M, F = self.ui, self.ui_bg, self.ui_fg
        self.panel = pyglet.shapes.Rectangle(0, 0, 10, 10, color=(28, 30, 34, 255), batch=B, group=M)
        self.divider = pyglet.shapes.Rectangle(0, 0, 3, 10, color=(55, 58, 64, 255), batch=B, group=self.ui_mid)
        self.sel_bar = pyglet.shapes.Rectangle(0, 0, 10, ROW_H, color=(52, 101, 164, 255), batch=B, group=self.ui_mid)
        self.head = pyglet.text.Label("", x=10, y=0, font_size=FONT + 1, weight="bold",
                                      color=(235, 235, 235, 255), batch=B, group=F)
        # search box
        self.search_border = pyglet.shapes.Rectangle(0, 0, 10, SEARCH_H, color=(70, 74, 82, 255), batch=B, group=M)
        self.search_bg = pyglet.shapes.Rectangle(0, 0, 10, SEARCH_H - 2, color=(16, 17, 20, 255), batch=B, group=self.ui_mid)
        self.search_lbl = pyglet.text.Label("", x=0, y=0, font_size=FONT, anchor_y="center",
                                            color=(235, 235, 235, 255), batch=B, group=F)
        self.search_clear_bg = pyglet.shapes.Circle(0, 0, 8, color=(60, 63, 70, 255), batch=B, group=self.ui_mid)
        self.search_clear = pyglet.text.Label("×", x=0, y=0, font_size=FONT + 2, anchor_x="center",
                                              anchor_y="center", color=(170, 170, 175, 255), batch=B, group=F)
        # scrollbar
        self.sb_track = pyglet.shapes.Rectangle(0, 0, SB_W, 10, color=(38, 40, 45, 255), batch=B, group=self.ui_mid)
        self.sb_thumb = pyglet.shapes.Rectangle(0, 0, SB_W, 10, color=(95, 100, 110, 255), batch=B, group=F)
        self.rows = []
        self.info = pyglet.text.Label("", x=0, y=0, font_size=FONT, multiline=True,
                                      width=700, color=(230, 230, 230, 255), anchor_y="top", batch=B, group=F)
        self.status_lbl = pyglet.text.Label("", x=0, y=8, font_size=FONT,
                                            color=(200, 200, 200, 255), batch=B, group=F)
        self.help_bg = pyglet.shapes.Rectangle(0, 0, 10, 10, color=(0, 0, 0, 200), batch=B, group=M)
        self.help_lbl = pyglet.text.Label(HELP_TEXT, x=0, y=0, font_size=FONT + 1, multiline=True,
                                          width=260, color=(240, 240, 240, 255), anchor_y="top", batch=B, group=F)
        self.buttons = []          # (name, rect, label)
        for name, text in (("animations", "Animations  (A)"), ("materials", "Materials  (I)"),
                           ("export", "Export  (E)"), ("settings", "Settings  (O)"),
                           ("help", "Help  (H)")):
            r = pyglet.shapes.Rectangle(0, 0, 10, 26, color=(50, 54, 62, 235), batch=B, group=M)
            lb = pyglet.text.Label(text, x=0, y=0, font_size=FONT, anchor_x="center", anchor_y="center",
                                   color=(235, 235, 235, 255), batch=B, group=F)
            self.buttons.append((name, r, lb))
        probe = pyglet.text.Label("abcdefghijklmnopqrstuvwxyz_0123456789 #·/", font_size=FONT)
        self.char_w = max(probe.content_width / 41.0, 4.0)
        probe.delete()
        self.matpanel = MaterialPanel(self)
        self.animpanel = AnimPanel(self)
        self.setpanel = SettingsPanel(self)
        self.export_job = None     # background export: dict(name, done, total, result, error)
        self.bones_loc = gl.glGetUniformLocation(self.program.id, b"bones")
        self._clip_count = 0
        self.cur_resize = self.get_system_mouse_cursor(self.CURSOR_SIZE_LEFT_RIGHT)
        self.cur_text = self.get_system_mouse_cursor(self.CURSOR_TEXT)
        self._layout()

        self.auto_shots = auto_shots
        threading.Thread(target=self.cat.scan, args=(root,), daemon=True).start()
        self.worker_q = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()
        pyglet.clock.schedule_interval(self.update, 1 / 60)
        self._last_model_count = -1
        self._rows_dirty = False

    # ------------------------------------------------------------- layout --
    def _clamp_panel(self):
        self.panel_w = int(max(PANEL_MIN, min(int(self.panel_w), self.width - VIEW_MIN)))

    def _layout(self):
        self._clamp_panel()
        w, h, pw = self.width, self.height, self.panel_w
        self.panel.width, self.panel.height = pw, h
        self.divider.x, self.divider.height = pw - 2, h
        self.divider.color = (90, 130, 190, 255) if (self.hover_div or self.drag == "resize") else (55, 58, 64, 255)
        self.head.y = h - 22
        # search box
        sy = h - 36 - SEARCH_H
        self.search_rect = (8, sy, pw - 16, SEARCH_H)
        self.search_border.x, self.search_border.y = 8, sy
        self.search_border.width = pw - 16
        self.search_border.color = (90, 140, 220, 255) if self.search_focus else (70, 74, 82, 255)
        self.search_bg.x, self.search_bg.y, self.search_bg.width = 9, sy + 1, pw - 18
        self.search_lbl.x, self.search_lbl.y = 16, sy + SEARCH_H // 2
        self.search_clear.x, self.search_clear.y = pw - 22, sy + SEARCH_H // 2 + 1
        self.search_clear_bg.x, self.search_clear_bg.y = pw - 22, sy + SEARCH_H // 2
        # list + scrollbar
        top = sy - 8
        bottom = 6
        nrows = max(1, (top - bottom) // ROW_H)
        while len(self.rows) < nrows:
            self.rows.append(pyglet.text.Label("", x=12, y=0, font_size=FONT,
                                               color=(215, 215, 215, 255), batch=self.ui, group=self.ui_fg))
        for i, r in enumerate(self.rows):
            r.y = top - (i + 1) * ROW_H + 5
            r.visible = i < nrows
        self.nrows = nrows
        self.list_top = top
        self.list_bottom = top - nrows * ROW_H
        self.sb_x = pw - SB_W - 5
        self.sb_track.x, self.sb_track.y = self.sb_x, self.list_bottom
        self.sb_track.height = nrows * ROW_H
        self.sel_bar.x, self.sel_bar.width = 0, self.sb_x - 3
        self.row_chars = max(8, int((self.sb_x - 22) / self.char_w))
        # buttons, top-right of the 3D view
        bx = w - 10
        for name, r, lb in reversed(self.buttons):
            bw = max(110, int(lb.content_width) + 24)
            bx -= bw
            r.x, r.y, r.width, r.height = bx, h - 36, bw, 26
            lb.x, lb.y = bx + bw // 2, h - 23
            bx -= 8
        if hasattr(self, "matpanel"):
            self.matpanel.layout()
        if hasattr(self, "animpanel"):
            self.animpanel.layout()
        if hasattr(self, "setpanel"):
            self.setpanel.layout()
        # right side
        self.info.x, self.info.y = pw + 14, h - 10
        self.status_lbl.x = pw + 14
        self._place_help()
        self._refresh_rows()

    def _place_help(self):
        """Help box under the top-right buttons (left of the material panel if open)."""
        h = self.height
        right = self._view_right() - 10
        hh = int(self.help_lbl.content_height) + 28
        hw = int(self.help_lbl.content_width) + 40
        self.help_bg.width, self.help_bg.height = hw, hh
        self.help_bg.x, self.help_bg.y = right - hw, h - 46 - hh
        self.help_lbl.x, self.help_lbl.y = right - hw + 20, h - 46 - 14

    def _view_right(self):
        for pnl in (getattr(self, "matpanel", None), getattr(self, "animpanel", None),
                    getattr(self, "setpanel", None)):
            if pnl is not None and pnl.visible:
                return pnl.rect[0]
        return self.width

    def close_panels(self, keep):
        """Side panels share the right edge: opening one closes the others."""
        for pnl in (self.matpanel, self.animpanel, getattr(self, "setpanel", None)):
            if pnl is not None and pnl is not keep and pnl.visible:
                pnl.toggle()

    def on_resize(self, w, h):
        super().on_resize(w, h)
        self._layout()

    # ------------------------------------------------------------- list --
    def _rebuild_visible(self):
        with self.cat.lock:
            models = list(self.cat.models)
        dups = self.cat.dups if self.cat.dups_done else {}
        terms = self.filter.lower().split()
        cur = self.current_mi            # model on screen (or loading)
        vis = []
        for i, m in enumerate(models):
            if m.is_proxy and not self.flags["proxies"]:
                continue
            if self.flags["nodup"] and i in dups:
                continue
            if terms:
                hay = f"{m.rel.lower()} {m.name.lower()}"
                if not all(t in hay for t in terms):
                    continue
            vis.append(i)
        self.vis_rows = vis
        if cur is None:
            self.sel = -1
            if vis:
                self.select(0)
                return
        elif cur in vis:
            self.sel = vis.index(cur)
            if not (self.scroll <= self.sel < self.scroll + self.nrows):
                self.scroll = max(0, self.sel - self.nrows // 2)
        elif cur in dups and dups[cur] in vis:
            # the shown model is now hidden as a duplicate: show the kept copy
            self.sel = vis.index(dups[cur])
            self.scroll = max(0, self.sel - self.nrows // 2)
            self.request_load(dups[cur])
        elif vis:
            # the shown model was filtered out: jump to the top match
            # (loaded after a short pause so typing doesn't trigger a load per key)
            self.sel, self.scroll = 0, 0
            pyglet.clock.unschedule(self._load_selected)
            pyglet.clock.schedule_once(self._load_selected, 0.25)
        else:
            self.sel = -1                # no matches
        self._refresh_rows()

    def _refresh_rows(self):
        self._rows_dirty = False
        models = self.cat.models
        n = len(self.vis_rows)
        self.scroll = max(0, min(self.scroll, max(0, n - self.nrows)))
        lim = self.row_chars
        for i, r in enumerate(self.rows[:self.nrows]):
            k = self.scroll + i
            if k < n:
                m = models[self.vis_rows[k]]
                stem = os.path.dirname(m.rel)
                txt = m.name
                if stem:
                    txt += f"  ·  {stem}"
                _set_label(r, txt if len(txt) <= lim else txt[:lim - 1] + "…",
                           (255, 255, 255, 255) if k == self.sel else (205, 205, 210, 255))
            else:
                _set_label(r, "")
        if 0 <= self.sel - self.scroll < self.nrows and self.sel >= 0:
            self.sel_bar.visible = True
            self.sel_bar.y = self.list_top - (self.sel - self.scroll + 1) * ROW_H
        else:
            self.sel_bar.visible = False
        # scrollbar
        track = self.nrows * ROW_H
        if n > self.nrows:
            th = max(24, int(track * self.nrows / n))
            frac = self.scroll / max(1, n - self.nrows)
            self.sb_thumb.height = th
            self.sb_thumb.x = self.sb_x
            self.sb_thumb.y = self.list_top - th - int((track - th) * frac)
            self.sb_thumb.color = (130, 136, 148, 255) if self.drag == "thumb" else (95, 100, 110, 255)
            self.sb_thumb.visible = self.sb_track.visible = True
        else:
            self.sb_thumb.visible = self.sb_track.visible = False
        total = len(models)
        _set_label(self.head, f"Models  {n:,}" + (f" of {total:,}" if n != total else ""))
        # search box text (placeholder when empty)
        max_chars = max(4, int((self.search_rect[2] - 40) / self.char_w))
        if self.filter:
            t = self.filter if len(self.filter) <= max_chars else "…" + self.filter[-(max_chars - 1):]
            caret = "|" if (self.search_focus and int(time.time() * 2) % 2 == 0) else ""
            _set_label(self.search_lbl, t + caret, (235, 235, 235, 255))
        else:
            _set_label(self.search_lbl,
                       ("|" if (self.search_focus and int(time.time() * 2) % 2 == 0) else "") +
                       ("" if self.search_focus else "Search models…  ( / )"),
                       (235, 235, 235, 255) if self.search_focus else (120, 124, 132, 255))
        self.search_clear.visible = self.search_clear_bg.visible = bool(self.filter)

    def set_filter(self, text):
        self.filter = text
        self.scroll = 0
        self._rebuild_visible()

    def _load_selected(self, dt=None):
        pyglet.clock.unschedule(self._load_selected)
        if 0 <= self.sel < len(self.vis_rows) and self.vis_rows[self.sel] != self.current_mi:
            self.request_load(self.vis_rows[self.sel])

    def select(self, row):
        if not self.vis_rows:
            return
        row = int(max(0, min(row, len(self.vis_rows) - 1)))
        changed = row != self.sel
        self.sel = row
        if self.sel < self.scroll:
            self.scroll = self.sel
        elif self.sel >= self.scroll + self.nrows:
            self.scroll = self.sel - self.nrows + 1
        self._refresh_rows()
        mi = self.vis_rows[row]
        if changed or self.gpu is None or self.gpu.model is not self.cat.models[mi]:
            self.request_load(mi)

    # ------------------------------------------------------------- loading --
    def request_load(self, mi):
        self.current_mi = mi
        self.load_gen += 1
        self.loading = mi
        self.worker_q.put((self.load_gen, mi))

    TEX_CACHE_BYTES = 200 * 1024 * 1024

    def _decode(self, m, tex_cache):
        """Parse a model and decode its textures (worker thread only)."""
        m.load()
        arrays, missing = {}, set()
        for mat in m.materials:
            for h in [mat.diffuse, mat.specular, mat.envmap, mat.overlay, mat.colour_map] + \
                    [h for _s, h, _r in mat.slot_roles()]:
                if not h or h in arrays or h in missing:
                    continue
                if h in tex_cache:
                    arrays[h] = tex_cache[h]
                    continue
                t = self.cat.find_texture(h, m.src)
                if t is None:
                    missing.add(h)
                    continue
                try:
                    arrays[h] = t.read_mip(0)
                except Exception:
                    missing.add(h)
                    continue
                tex_cache[h] = arrays[h]
                self._tex_bytes += arrays[h].nbytes
                while self._tex_bytes > self.TEX_CACHE_BYTES and len(tex_cache) > 1:
                    old = tex_cache.pop(next(iter(tex_cache)))
                    self._tex_bytes -= old.nbytes
        return arrays, missing

    def _worker(self):
        """Decode meshes and texture pixels off the UI thread. When idle,
        pre-decode the models next to the selection so Up/Down is quicker."""
        tex_cache = {}
        self._tex_bytes = 0
        while True:
            try:
                gen, mi = self.worker_q.get(timeout=0.15)
            except queue.Empty:
                try:
                    mj = self.prefetch.pop(0)
                except (IndexError, AttributeError):
                    continue
                try:
                    self._decode(self.cat.models[mj], tex_cache)
                except Exception:
                    pass
                continue
            # skip stale requests when the user is scrolling quickly
            while not self.worker_q.empty():
                gen, mi = self.worker_q.get()
            if gen != self.load_gen:
                continue
            m = self.cat.models[mi]
            try:
                arrays, missing = self._decode(m, tex_cache)
                self.results.put((gen, mi, arrays, missing, None))
            except Exception as e:
                self.results.put((gen, mi, {}, set(), e))

    def _finish_load(self, gen, mi, arrays, missing, err):
        if gen != self.load_gen:
            return
        self.loading = None
        m = self.cat.models[mi]
        if self.gpu:
            self.gpu.delete()
            self.gpu = None
        if err:
            self.matpanel.set_model(None)
            self.animpanel.set_model(None)
            self.flash(f"Could not load {m.name}: {err}")
            self._update_info(m, missing, err)
            return
        self.gpu = GpuModel(self.program, m, arrays)
        self.missing = missing
        # neighbours to pre-decode while the user looks at this one
        if 0 <= self.sel < len(self.vis_rows):
            self.prefetch = [self.vis_rows[r] for r in (self.sel + 1, self.sel - 1, self.sel + 2)
                             if 0 <= r < len(self.vis_rows) and not self.cat.models[self.vis_rows[r]].loaded]
        self.matpanel.set_model(self.gpu)
        self.animpanel.set_model(self.gpu)
        self.setpanel.set_model(self.gpu)
        self.frame()
        self._update_info(m, missing)

    def _update_info(self, m, missing=(), err=None):
        lo, hi = m.bounds()
        size = hi - lo
        decls = sorted({x.decl for x in m.meshes})
        lines = [f"{m.name}", f"{m.rel}",
                 f"{len(m.meshes)} meshes · {m.vertex_count():,} vertices · {m.triangle_count():,} triangles",
                 f"{len(m.materials)} materials · size {size[0]:.2f} × {size[1]:.2f} × {size[2]:.2f}",
                 f"vertex formats: {', '.join(decls) or '-'}"]
        need = {h for mat in m.materials for h in (mat.diffuse,)}
        if missing & need:
            lines.append(f"{len(missing & need)} diffuse texture(s) not found in this folder")
        if not m.meshes:
            lines.append("nothing to draw: no mesh in this model could be decoded")
        if m.warnings:
            lines.append("warnings: " + "; ".join(sorted(set(m.warnings))[:3]))
        if err:
            lines.append(f"error: {err}")
        self.info.text = "\n".join(lines)

    # ------------------------------------------------------------- camera --
    def frame(self):
        if not self.gpu:
            return
        c = self.gpu.center
        self.cam["target"] = (Z_TO_Y[:3, :3] @ c).astype(np.float32)
        self.cam["dist"] = self.gpu.radius * 2.6
        self.cam["yaw"], self.cam["pitch"] = 35.0, 15.0

    def _eye(self):
        y, p = math.radians(self.cam["yaw"]), math.radians(self.cam["pitch"])
        d = self.cam["dist"]
        off = np.array([math.cos(p) * math.sin(y), math.sin(p), math.cos(p) * math.cos(y)], np.float32) * d
        return self.cam["target"] + off

    # ------------------------------------------------------------- frame --
    def update(self, dt):
        while not self.results.empty():
            self._finish_load(*self.results.get())
        n = (len(self.cat.models), self.cat.dups_done)
        if n != self._last_model_count:
            self._last_model_count = n
            self._rebuild_visible()
        if self.flags["spin"]:
            self.cam["yaw"] += 20 * dt
        self.animpanel.tick(dt)
        self._check_export()
        nclips = len(self.cat.clips)
        if nclips != self._clip_count and (self.cat.done or nclips - self._clip_count > 200):
            self._clip_count = nclips
            if self.gpu:
                self.animpanel.refresh_list()
                self.setpanel.rebuild()
        if self.search_focus:
            self._refresh_rows()          # caret blink
        st = (f"Scanning {self.cat.files_done:,}/{self.cat.files_total:,} files…   "
              if not self.cat.done else
              f"{len(self.cat.models):,} models · {len(self.cat.textures):,} textures indexed   ")
        if self.cat.done and not self.cat.dups_done:
            st += "Checking for duplicates…   "
        elif self.cat.dups and self.flags["nodup"]:
            st += f"{len(self.cat.dups):,} duplicates hidden (N)   "
        if self.loading is not None:
            st += "Loading…   "
        if self.search_focus:
            st += "Typing in search (Enter or Esc to leave; Ctrl+E exports)   "
        if self.msg and time.time() - self.msg_time < 10:
            st += self.msg
        self.status_lbl.text = st.rstrip()
        recent = self.msg and time.time() - self.msg_time < 10
        self.status_lbl.color = ((255, 120, 110, 255) if getattr(self, "msg_error", False) else (150, 230, 150, 255)) \
            if recent else (200, 200, 200, 255)
        if self.auto_shots is not None:
            self._auto_step()

    def on_draw(self):
        if self._rows_dirty:
            self._refresh_rows()
        bg = BGS[self.bg]
        gl.glClearColor(bg[0], bg[1], bg[2], 1)
        self.clear()
        fbw, fbh = self.get_framebuffer_size()
        sx = fbw / max(self.width, 1)
        right = self._view_right()
        vx, vw = int(self.panel_w * sx), max(1, int(right * sx) - int(self.panel_w * sx))
        if self.gpu:
            gl.glViewport(vx, 0, vw, fbh)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthFunc(gl.GL_LEQUAL)
            gl.glDisable(gl.GL_BLEND)
            if self.flags["cull"]:
                gl.glEnable(gl.GL_CULL_FACE)
                gl.glCullFace(gl.GL_BACK)
                gl.glFrontFace(gl.GL_CW)
            if self.flags["wire"]:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            eye = self._eye()
            view = look_at(eye, self.cam["target"], np.array([0, 1, 0], np.float32))
            r = self.gpu.radius
            near = max(self.cam["dist"] - r * 1.5, self.cam["dist"] * 0.002, 1e-3)
            proj = perspective(45, vw / max(fbh, 1), near, self.cam["dist"] + r * 3)
            model = Z_TO_Y
            p = self.program
            p.use()
            p["mvp"] = gl_mat(proj @ view @ model)
            p["view"] = gl_mat(view)
            p["model"] = gl_mat(model)
            p["eye"] = tuple(float(x) for x in eye)
            p["use_tex"] = int(self.flags["tex"])
            p["use_light"] = int(self.flags["light"])
            p["use_glow"] = int(self.flags["glow"])
            p["use_col"] = int(self.flags["col"])
            p["use_env"] = int(self.flags["env"])
            p["diffuse_tex"] = 0
            p["glow_tex"] = 1
            p["spec_tex"] = 2
            p["env_tex"] = 3
            p["colour_tex"] = 4
            hl = self.matpanel.highlight
            posed = self.animpanel.matrices()
            if posed is not None:
                arr = (gl.GLfloat * posed.size)(*posed.astype(np.float32).ravel())
                gl.glUniformMatrix4fv(self.bones_loc, len(posed), gl.GL_FALSE, arr)
            for vl, mesh, diff, spec, env, glow, tints, colmap, unlit, mcols, skin in self.gpu.parts:
                if hl is not None and mesh.material != hl:
                    continue                  # material selected: show only its faces
                p["skinned"] = int(skin and posed is not None)
                p["mask_r"], p["mask_g"], p["mask_b"] = (tuple(float(v) for v in c) for c in mcols)
                game = self.tint_mode == 0 and self.flags["tex"]
                p["has_colmap"] = int(bool(colmap) and game and mesh.uv is not None)
                p["unlit"] = int(unlit and game)
                mt = tints[0] if self.tint_mode == 0 else tints[1] if self.tint_mode == 1 else (1.0, 1.0, 1.0)
                p["mat_tint"] = tuple(float(c) for c in mt) if self.flags["tex"] else (1.0, 1.0, 1.0)
                has_uv = mesh.uv is not None
                p["has_tex"] = int(bool(diff) and has_uv)
                p["has_glow"] = int(bool(glow) and has_uv)
                p["has_spec"] = int(bool(spec) and has_uv)
                p["has_env"] = int(bool(env))
                p["has_nrm"] = int(mesh.nrm is not None)
                p["has_col"] = int(mesh.col is not None)
                mk = self.gpu.model.materials[mesh.material].kind if mesh.material >= 0 else None
                p["tint"] = (1.0, 1.0, 1.0) if mk == "colour" else \
                    (0.75, 0.75, 0.78) if mesh.material >= 0 else (0.55, 0.6, 0.7)
                gl.glActiveTexture(gl.GL_TEXTURE0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, diff or 0)
                gl.glActiveTexture(gl.GL_TEXTURE1)
                gl.glBindTexture(gl.GL_TEXTURE_2D, glow or 0)
                gl.glActiveTexture(gl.GL_TEXTURE2)
                gl.glBindTexture(gl.GL_TEXTURE_2D, spec or 0)
                gl.glActiveTexture(gl.GL_TEXTURE3)
                gl.glBindTexture(gl.GL_TEXTURE_2D, env or 0)
                gl.glActiveTexture(gl.GL_TEXTURE4)
                gl.glBindTexture(gl.GL_TEXTURE_2D, colmap or 0)
                vl.draw(gl.GL_TRIANGLES)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            p.stop()
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glViewport(0, 0, fbw, fbh)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.help_bg.visible = self.help_lbl.visible = self.flags["help"]
        if self.flags["help"]:
            self._place_help()
        self.ui.draw()
        self.matpanel.draw(fbw / max(self.width, 1))
        self.animpanel.draw(fbw / max(self.width, 1))
        self.setpanel.draw(fbw / max(self.width, 1))

    # ------------------------------------------------------------- input --
    def flash(self, text, error=False):
        self.msg, self.msg_time = text, time.time()
        self.msg_error = error

    def on_key_press(self, sym, mods):
        if sym == key.E and mods & (key.MOD_CTRL | key.MOD_COMMAND):
            self.export_current()
            return pyglet.event.EVENT_HANDLED
        if sym == key.F12:
            self.screenshot()
            return pyglet.event.EVENT_HANDLED
        if self.search_focus:
            if sym in (key.ENTER, key.RETURN):
                self._load_selected()        # open the highlighted match now
                self.set_search_focus(False)
                return pyglet.event.EVENT_HANDLED
            if sym == key.ESCAPE:
                if self.filter:
                    self.set_filter("")
                else:
                    self.set_search_focus(False)
                return pyglet.event.EVENT_HANDLED
            if sym == key.BACKSPACE:
                if mods & key.MOD_CTRL:
                    self.set_filter(self.filter.rstrip().rsplit(" ", 1)[0] if " " in self.filter.rstrip() else "")
                else:
                    self.set_filter(self.filter[:-1])
                return pyglet.event.EVENT_HANDLED
            if sym == key.A and mods & key.MOD_CTRL:
                self.set_filter("")
                return pyglet.event.EVENT_HANDLED
            if sym == key.V and mods & (key.MOD_CTRL | key.MOD_COMMAND):
                try:
                    clip = self.get_clipboard_text() or ""
                except Exception:
                    clip = ""
                if clip:
                    self.set_filter(self.filter + clip.splitlines()[0])
                return pyglet.event.EVENT_HANDLED
            if sym not in (key.UP, key.DOWN, key.PAGEUP, key.PAGEDOWN):
                return pyglet.event.EVENT_HANDLED   # typed characters arrive in on_text
        if sym == key.ESCAPE and self.matpanel.on_escape():
            return pyglet.event.EVENT_HANDLED
        if sym == key.ESCAPE:
            if self.filter:
                self.set_filter("")
                return pyglet.event.EVENT_HANDLED
            if self.flags["help"]:
                self.flags["help"] = False
                return pyglet.event.EVENT_HANDLED
            return pyglet.event.EVENT_HANDLED   # don't quit on Esc
        step = {key.UP: -1, key.DOWN: 1, key.PAGEUP: -self.nrows, key.PAGEDOWN: self.nrows}
        if sym in step:
            self.select(self.sel + step[sym])
        elif sym == key.HOME:
            self.select(0)
        elif sym == key.END:
            self.select(len(self.vis_rows) - 1)
        elif sym == key.SLASH or (sym == key.F and mods & (key.MOD_CTRL | key.MOD_COMMAND)):
            self.set_search_focus(True)
        elif sym == key.F:
            self.frame()
        elif sym == key.SPACE:
            self.flags["spin"] = not self.flags["spin"]
        elif sym == key.B:
            self.bg = (self.bg + 1) % len(BGS)
        elif sym == key.I:
            self.matpanel.toggle()
        elif sym == key.A:
            self.animpanel.toggle()
        elif sym == key.O:
            self.setpanel.toggle()
        elif sym in (key.BRACKETLEFT, key.BRACKETRIGHT) and self.animpanel.visible:
            self.animpanel.step(-1 if sym == key.BRACKETLEFT else 1)
        elif sym == key.K:
            self.tint_mode = (self.tint_mode + 1) % 3
            self.flash("Material colours: " + ("game", "raw (as stored)", "off")[self.tint_mode])
        elif sym == key.F12:
            self.screenshot()
        elif sym == key.E:
            self.export_current()
        elif sym == key.R:
            self.frame()
        else:
            names = {key.T: "tex", key.L: "light", key.G: "glow", key.M: "env", key.V: "col", key.W: "wire",
                     key.C: "cull", key.P: "proxies", key.H: "help", key.N: "nodup"}
            if sym in names:
                f = names[sym]
                self.flags[f] = not self.flags[f]
                if f in ("proxies", "nodup"):
                    self._rebuild_visible()
                if f == "nodup":                 # flag means "hide"; shown as "show"
                    self.flash(f"Show duplicate models: {'off' if self.flags[f] else 'on'}")
                elif f != "help":
                    self.flash(f"{LABELS[f]}: {'on' if self.flags[f] else 'off'}")
        return pyglet.event.EVENT_HANDLED

    def on_text(self, text):
        if self.search_focus and text.isprintable():
            if text == "/" and not self.filter and getattr(self, "_just_focused", False):
                return
            self.set_filter(self.filter + text)

    def set_search_focus(self, on, _orig=None):
        if on != self.search_focus:
            self.search_focus = on
            self._just_focused = on
            pyglet.clock.schedule_once(lambda dt: setattr(self, "_just_focused", False), 0.05)
            self._layout()

    def _on_divider(self, x):
        return abs(x - self.panel_w) <= 5

    def _in_search(self, x, y):
        sx, sy, sw, sh = self.search_rect
        return sx <= x <= sx + sw and sy <= y <= sy + sh

    def on_mouse_motion(self, x, y, dx, dy):
        x, y = int(x), int(y)          # pyglet 2.1 reports float coordinates
        hov = self._on_divider(x)
        if hov != self.hover_div:
            self.hover_div = hov
            self._layout()
        if hov:
            self.set_mouse_cursor(self.cur_resize)
        elif self._in_search(x, y):
            self.set_mouse_cursor(self.cur_text)
        else:
            self.set_mouse_cursor(None)

    def _scroll_to_thumb(self, y):
        n = len(self.vis_rows)
        if n <= self.nrows:
            return
        track = self.nrows * ROW_H
        th = self.sb_thumb.height
        # y of the thumb's top edge, relative to the top of the track
        rel = (self.list_top - (y + self.thumb_grab)) / max(1, track - th)
        self.scroll = int(round(max(0.0, min(1.0, rel)) * (n - self.nrows)))
        self._rows_dirty = True          # redrawn once per frame (see on_draw)

    def on_mouse_press(self, x, y, button, mods):
        x, y = int(x), int(y)
        self.drag = None
        if self.matpanel.on_press(x, y, button):
            return
        if self.animpanel.on_press(x, y, button):
            return
        if self.setpanel.on_press(x, y, button):
            return
        if self._on_divider(x) and button == mouse.LEFT:
            self.drag = "resize"
            self._layout()
            return
        if x < self.panel_w:
            if button != mouse.LEFT:
                return
            if self._in_search(x, y):
                if self.filter and x >= self.search_clear.x - 10:
                    self.set_filter("")
                self.set_search_focus(True)
                return
            self.set_search_focus(False)
            n = len(self.vis_rows)
            if x >= self.sb_x - 2 and n > self.nrows and self.list_bottom <= y <= self.list_top:
                t = self.sb_thumb
                if t.y <= y <= t.y + t.height:          # grab the thumb
                    self.thumb_grab = t.y + t.height - y
                else:                                    # click on the track: jump there
                    self.thumb_grab = t.height // 2
                    self._scroll_to_thumb(y)
                self.drag = "thumb"
                self._refresh_rows()
                return
            row = (self.list_top - y) // ROW_H
            if 0 <= row < self.nrows and self.scroll + row < n:
                self.select(self.scroll + row)
            return
        for name, r, lb in self.buttons:
            if button == mouse.LEFT and r.x <= x <= r.x + r.width and r.y <= y <= r.y + r.height:
                r.color = (80, 110, 160, 255)
                pyglet.clock.schedule_once(lambda dt, r=r: setattr(r, "color", (50, 54, 62, 235)), 0.15)
                if name == "materials":
                    self.matpanel.toggle()
                elif name == "animations":
                    self.animpanel.toggle()
                elif name == "export":
                    self.export_current()
                elif name == "settings":
                    self.setpanel.toggle()
                elif name == "help":
                    self.flags["help"] = not self.flags["help"]
                return
        self.set_search_focus(False)
        self.drag = "view"

    def on_mouse_release(self, x, y, button, mods):
        x, y = int(x), int(y)
        was = self.drag
        self.drag = None
        if self.animpanel.on_release():
            return
        if was == "resize":
            self.cfg["panel_width"] = self.panel_w
            try:
                cfg = json.load(open(CONFIG)) if os.path.exists(CONFIG) else {}
                cfg["panel_width"] = self.panel_w
                json.dump(cfg, open(CONFIG, "w"))
            except Exception:
                pass
            self.hover_div = self._on_divider(x)
            self._layout()
        elif was == "thumb":
            self._refresh_rows()

    def on_mouse_drag(self, x, y, dx, dy, buttons, mods):
        x, y = int(x), int(y)
        if self.animpanel.on_drag(x, y):
            return
        if self.drag == "resize":
            self.panel_w = x
            self._layout()
            return
        if self.drag == "thumb":
            self._scroll_to_thumb(y)
            return
        if self.drag != "view":
            return
        if buttons & mouse.LEFT and not mods & key.MOD_SHIFT:
            self.cam["yaw"] -= dx * 0.4
            self.cam["pitch"] = max(-89, min(89, self.cam["pitch"] - dy * 0.4))
        elif buttons & (mouse.RIGHT | mouse.MIDDLE | mouse.LEFT):
            y_, p_ = math.radians(self.cam["yaw"]), math.radians(self.cam["pitch"])
            right = np.array([math.cos(y_), 0, -math.sin(y_)], np.float32)
            up = np.array([-math.sin(p_) * math.sin(y_), math.cos(p_), -math.sin(p_) * math.cos(y_)], np.float32)
            k = self.cam["dist"] * 0.0015
            self.cam["target"] = self.cam["target"] - right * dx * k - up * dy * k

    def on_mouse_scroll(self, x, y, sx, sy):
        x, y = int(x), int(y)
        if self.matpanel.on_scroll(x, y, sy):
            return
        if self.animpanel.on_scroll(x, y, sy):
            return
        if self.setpanel.on_scroll(x, y, sy):
            return
        if x < self.panel_w:
            step = int(round(sy * 3)) or (1 if sy > 0 else -1)
            self.scroll = max(0, self.scroll - step)
            self._rows_dirty = True
        else:
            self.cam["dist"] *= 0.88 ** sy

    # ------------------------------------------------------------- output --
    def screenshot(self, path=None):
        try:
            if path is None:
                m = self.gpu.model if self.gpu else None
                name = (m.name if m else "viewer").replace(" #", "_").replace(" ", "_")
                path = os.path.join(output_dir("screenshots"), f"{name}_{time.strftime('%H%M%S')}.png")
            pyglet.image.get_buffer_manager().get_color_buffer().save(path)
        except Exception as e:
            self.flash(f"Screenshot failed: {e}", error=True)
            return None
        self.flash(f"Screenshot saved: {os.path.abspath(path)}")
        print("screenshot:", os.path.abspath(path))
        return path

    def export_current(self):
        if self.export_job is not None:
            self.flash("An export is still running…")
            return
        if not self.gpu:
            self.flash("Nothing to export - select a model first", error=True)
            return
        m = self.gpu.model
        name = m.name.replace(" #", "_").replace(" ", "_")
        try:
            out = os.path.join(output_dir("model_exports"), name)     # .glb + its PNG textures
            os.makedirs(out, exist_ok=True)
            images, missing = {}, set()
            for mat in m.materials:
                for _slot, h, role in mat.slot_roles():   # every texture slot
                    if h in images or h in missing:
                        continue
                    t = self.cat.find_texture(h, m.src)
                    if t is None:
                        missing.add(h)
                        continue
                    try:
                        images[h] = t.image(0)
                    except Exception:
                        missing.add(h)
            # light/decal materials: bake diffuse mask x colour map into one texture
            nbaked = 0
            for mat in m.materials:
                d, c = mat.diffuse, mat.colour_map
                if not c or ("baked", d, c) in images:
                    continue
                td, tc = self.cat.find_texture(d, m.src), self.cat.find_texture(c, m.src)
                if td is None or tc is None:
                    continue
                try:
                    a = td.image(0)
                    msk = np.asarray(tc.image(0).resize(a.size, Image.BILINEAR).convert("RGB"), np.float32) / 255
                    cr, cg, cb = (np.array(c, np.float32) for c in mat.mask_colours)
                    col = msk[..., :1] * cr + msk[..., 1:2] * cg + msk[..., 2:3] * cb
                    rgb = np.asarray(a.convert("RGB"), np.float32) / 255 * col
                    baked = Image.fromarray(np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
                    baked.putalpha(a.getchannel("A"))
                    images[("baked", d, c)] = baked
                    nbaked += 1
                except Exception:
                    pass
            glb = os.path.join(out, name + ".glb")
            with open(os.path.join(out, name + "_materials.txt"), "w", encoding="utf-8") as fh:
                fh.write(m.material_report(lambda h: self.cat.find_texture(h, m.src)) + "\n")
        except Exception as e:
            self.flash(f"Export failed: {e}", error=True)
            print("export failed:", e, file=sys.stderr)
            return
        skel = self.setpanel.get("export_skeleton")
        clips = []
        if skel and self.setpanel.get("export_animations") and self.gpu.skinnable:
            clips = list(self.animpanel.items)
        job = dict(name=name, out=out, glb=glb, missing=len(missing), done=0, total=len(clips),
                   result=None, error=None, finished=False)
        self.export_job = job

        def progress(done, total):
            job["done"], job["total"] = done, total

        def run():
            try:
                job["result"] = cm.export_glb(m, glb, images, skeleton=skel, clips=clips, progress=progress)
            except Exception as e:          # noqa: BLE001
                job["error"] = e
            job["finished"] = True

        if clips:
            self.flash(f"Exporting {name}.glb with {len(clips)} animations…")
        threading.Thread(target=run, daemon=True).start()

    def _check_export(self):
        job = self.export_job
        if job is None:
            return
        if not job["finished"]:
            if job["total"]:
                self.flash(f"Exporting {job['name']}.glb - animation {min(job['done'] + 1, job['total'])}"
                           f"/{job['total']}…")
            return
        self.export_job = None
        if job["error"] is not None:
            self.flash(f"Export failed: {job['error']}", error=True)
            print("export failed:", job["error"], file=sys.stderr)
            return
        info = job["result"]
        parts = [f"{info['meshes']} meshes", f"{info['textures']} texture PNGs"]
        if info["joints"]:
            parts.append(f"{info['joints']} bones")
        if info.get("animations"):
            parts.append(f"{info['animations']} animations")
        msg = f"Exported {job['name']}.glb ({', '.join(parts)}) to {os.path.abspath(job['out'])}"
        if info.get("animations_failed"):
            msg += f"  ({len(info['animations_failed'])} animation(s) could not be read)"
        if job["missing"]:
            msg += f"  ({job['missing']} referenced texture(s) not in the game folder)"
        self.flash(msg)
        print("exported:", os.path.abspath(job["glb"]))

    # ------------------------------------------------------------- testing --
    def _auto_step(self):
        """--shots mode: render every model to PNG, then quit."""
        st = self.auto_shots
        if not self.cat.done:
            return
        if "i" not in st:
            self._rebuild_visible()
            st["i"] = 0
            st["state"] = "select"
        if st["i"] >= len(self.vis_rows):
            print(f"saved {st['i']} screenshot(s) to {st['dir']}")
            pyglet.app.exit()
            return
        if st["state"] == "select":
            self.select(st["i"])
            st["state"] = "wait"
            st["t"] = time.time()
        elif st["state"] == "wait" and self.loading is None and self.gpu is not None:
            st["state"] = "shoot"
            st["frames"] = 0
        elif st["state"] == "shoot":
            st["frames"] += 1
            if st["frames"] >= 2:
                m = self.gpu.model
                os.makedirs(st["dir"], exist_ok=True)
                name = m.name.replace(" #", "_").replace(" ", "_")
                self.screenshot(os.path.join(st["dir"], f"{st['i']:03d}_{name}.png"))
                st["i"] += 1
                st["state"] = "select"


LABELS = dict(tex="Textures", light="Lighting", glow="Mask texture", env="Reflections", col="Vertex colors",
              wire="Wireframe", cull="Invert faces", proxies="Proxy models",
              nodup="Show duplicate models")

HELP_TEXT = """T - textures
L - lighting
M - reflections
V - vertex colors
W - wireframe
C - invert faces
B - background color
K - render mode
G - show mask texture
P - show proxy models
N - show duplicate models
A - animations
O - settings (export options)"""


def pick_folder(initial):
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    r = tk.Tk()
    r.withdraw()
    d = filedialog.askdirectory(title="Select the game folder (extracted 'files' folder)",
                                initialdir=initial or os.getcwd())
    r.destroy()
    return d or None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    shots = None
    if "--shots" in sys.argv:
        i = sys.argv.index("--shots")
        shots = {"dir": sys.argv[i + 1] if i + 1 < len(sys.argv) else "shots"}
        args = [a for a in args if a != shots["dir"]]
    cfg = {}
    try:
        cfg = json.load(open(CONFIG))
    except Exception:
        pass
    folder = args[0] if args else pick_folder(cfg.get("last_folder"))
    if not folder:
        if cfg.get("last_folder") and os.path.isdir(cfg["last_folder"]):
            folder = cfg["last_folder"]
        else:
            sys.exit("usage: cybertron_model_viewer.py GAME_FOLDER")
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        sys.exit(f"not a folder: {folder}")
    cfg["last_folder"] = folder
    try:
        json.dump(cfg, open(CONFIG, "w"))
    except Exception:
        pass
    Viewer(folder, shots)
    pyglet.app.run()


if __name__ == "__main__":
    main()
