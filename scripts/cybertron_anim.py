"""
Transformers: Cybertron Adventures (Wii) - skeletal animation reader.

Animations live in the same .data containers as the models (group 0x70 =
clip headers/tables, group 0x71 = key data), one clip per 0x70/01 chunk:

  0x70/01  0x5C bytes: +0x04 name hash, +0x08 u32 frame count,
           +0x0C u32 node count (= the skeleton's hierarchy, group 0x80/3)
  0x70/02  clip name (ASCII, zero padded), e.g. "weapon_trans_primary"
  0x70/03  u32 flags per hierarchy node:
             0x10 / 0x20 / 0x40  node has a rotation track, keys are
                                 8 bytes (4 x s16 quaternion) /
                                 6 bytes (s16 x, s8 y, s16 z, s8 w) /
                                 4 bytes (4 x s8)
             0x01 (with 0x10)    2-byte keys: one s16 (rotation about a
                                 single axis)
             0x02  rotation track is constant (one key)
             0x04  translation track is constant (one key)
             0x200 node also has a scale track (0x71/03)
  0x71/01  rotation track per tracked node (1 key, or one per frame)
  0x71/02  translation track per tracked node: 3 x f32 per key
  0x71/03  uniform scale track per 0x200 node: u16 per key, 2048 = 1.0
           (weapons are kept at 0.1 until drawn, transformation parts
           grow / shrink)

Tracked nodes take their tracks in hierarchy order. Rotations are local
(relative to the parent), quaternion x, y, z, w.
"""

import struct

import numpy as np

import cybertron_tex as ct

ROT_KEY = {0x10: 8, 0x20: 6, 0x40: 4}


class Clip:
    def __init__(self, name, frames, nodes, flags):
        self.name, self.frames, self.nodes, self.flags = name, frames, nodes, flags
        self.rot = {}       # node index -> (frames|1, 4) quaternions (x, y, z, w)
        self.trans = {}     # node index -> (frames|1, 3)
        self.scale = {}     # node index -> (frames|1,) uniform scale
        self.warnings = []
        self.src = None
        self.index = 0

    def local(self, node, frame):
        """(quaternion xyzw, translation) of a node at a frame, or None if
        the clip doesn't drive the node."""
        r = self.rot.get(node)
        t = self.trans.get(node)
        if r is None and t is None:
            return None
        q = r[min(frame, len(r) - 1)] if r is not None else None
        p = t[min(frame, len(t) - 1)] if t is not None else None
        return q, p


def _s(v, bits):
    return v - (1 << bits) if v >= 1 << (bits - 1) else v


def _decode_rot(b, flags, frames):
    ks = 2 if (flags & 0x10 and flags & 0x01) else ROT_KEY[flags & 0x70]
    n = len(b) // ks
    if n == 0:
        return None
    a = np.frombuffer(b[:n * ks], np.uint8).reshape(n, ks)
    if ks == 8:
        q = a.copy().view(">i2").astype(np.float64).reshape(n, 4) / 32767.0
    elif ks == 4:
        q = a.view(np.int8).astype(np.float64) / 127.0
    elif ks == 6:
        x = a[:, 0:2].copy().view(">i2").ravel() / 32768.0
        y = a[:, 2].view(np.int8) / 128.0
        z = a[:, 3:5].copy().view(">i2").ravel() / 32768.0
        w = a[:, 5].view(np.int8) / 128.0
        q = np.stack([x, y, z, w], 1).astype(np.float64)
    else:                     # 2-byte single-axis keys (axis not known yet)
        v = a.copy().view(">i2").ravel() / 32768.0
        q = np.stack([np.zeros_like(v), np.zeros_like(v), v, np.sqrt(np.clip(1 - v * v, 0, 1))], 1)
    nrm = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.where(nrm > 1e-9, nrm, 1)


def read_clips(path, sub_index=None):
    """All animation clips in a .data file (optionally one sub-file)."""
    clips = []
    with ct.open_data(path) as f:
        subs = ct.find_subfiles(f, ct.data_size(f), path)
        for si, (_, chunks) in enumerate(subs):
            if sub_index is not None and si != sub_index:
                continue
            groups, cur = [], None
            for c in chunks:
                if c.is_meta or c.group not in (0x70, 0x71):
                    continue
                if c.group == 0x70 and c.sub == 0x01:
                    cur = {}
                    groups.append(cur)
                if cur is not None:
                    cur.setdefault((c.group, c.sub), []).append(c)
                    cur.setdefault("seq", []).append(c)
            for g in groups:
                try:
                    clip = _read_clip(f, g)
                except Exception as ex:          # noqa: BLE001
                    clip = None
                    print(f"warning: {path}: animation skipped ({ex})")
                if clip is not None:
                    clip.src, clip.index, clip.sub_index = path, len(clips), si
                    clips.append(clip)
    return clips


def _read_clip(f, g):
    rd = lambda c: ct._read_at(f, c.abs, c.size)        # noqa: E731
    head = rd(g[(0x70, 1)][0])
    if len(head) < 16:
        return None
    frames, nodes = struct.unpack(">II", head[8:16])
    name = ""
    if (0x70, 2) in g:
        name = rd(g[(0x70, 2)][0]).split(b"\0")[0].decode("latin1", "replace")
    if (0x70, 3) not in g or not 0 < nodes < 1024:
        return None
    fb = rd(g[(0x70, 3)][0])
    flags = list(struct.unpack(f">{nodes}I", fb[:4 * nodes]))
    clip = Clip(name or f"clip {struct.unpack('>I', head[4:8])[0]:08x}", frames, nodes, flags)
    # Key tracks are stored node by node in file order: a rotation track
    # (71/01), then an optional translation (71/02), then an optional scale
    # (71/03).  Some nodes have no translation track, so the streams can't
    # simply be paired by index.
    seq = [c for c in g.get("seq", []) if c.group == 0x71 and c.sub in (1, 2, 3)]
    tracked = [n for n, fl in enumerate(flags) if fl & 0x70]
    pos, used = 0, 0
    for node in tracked:
        fl = flags[node]
        if pos >= len(seq) or seq[pos].sub != 1:
            break
        q = _decode_rot(rd(seq[pos]), fl, frames)
        if q is not None:
            clip.rot[node] = q
        pos += 1
        used += 1
        if pos < len(seq) and seq[pos].sub == 2:
            b = rd(seq[pos])
            n = len(b) // 12
            if n:
                clip.trans[node] = np.frombuffer(b[:n * 12], ">f4").astype(np.float64).reshape(n, 3)
            pos += 1
        if pos < len(seq) and seq[pos].sub == 3:
            b = rd(seq[pos])
            if len(b) >= 2:
                clip.scale[node] = np.frombuffer(b[:len(b) // 2 * 2], ">u2").astype(np.float64) / 2048.0
            pos += 1
    if used != len(tracked) or pos != len(seq):
        clip.warnings.append(f"{len(tracked)} tracked nodes, {used} decoded, {len(seq) - pos} tracks left over")
    return clip


# --------------------------------------------------------------- catalogue --
FPS = 30.0


class ClipInfo:
    """A clip found in a file, without its key data (cheap to list)."""
    __slots__ = ("src", "rel", "sub_index", "order", "name", "frames", "nodes", "hashes", "rig_rest")

    def __init__(self, src, rel, sub_index, order, name, frames, nodes, hashes, rig_rest):
        self.src, self.rel, self.sub_index, self.order = src, rel, sub_index, order
        self.name, self.frames, self.nodes = name, frames, nodes
        self.hashes = hashes          # bone hash per node (hierarchy in the same sub-file) or None
        self.rig_rest = rig_rest      # [(parent, translation, rotation)] of that hierarchy

    @property
    def seconds(self):
        return self.frames / FPS


def _hierarchies(f, chunks):
    out = []
    for c in chunks:
        if c.is_meta or c.group != 0x80 or c.sub != 3 or c.size % 72:
            continue
        b = ct._read_at(f, c.abs, c.size)
        hs, rest = [], []
        for o in range(0, len(b), 72):
            v = struct.unpack(">II5I3fI4f3f", b[o:o + 72])
            hs.append(v[0])
            rest.append((-1 if v[1] == 0xFFFFFFFF else v[1], v[7:10], v[11:15]))   # parent, t, q
        out.append((hs, rest))
    return out


def list_clips(path, rel=None):
    """ClipInfo for every animation in a file (reads only clip headers)."""
    out = []
    with ct.open_data(path) as f:
        subs = ct.find_subfiles(f, ct.data_size(f), path)
        for si, (_, chunks) in enumerate(subs):
            if not any(c.group == 0x70 and c.sub == 1 and not c.is_meta for c in chunks):
                continue
            hier = _hierarchies(f, chunks)
            order = 0
            name_chunks = [c for c in chunks if c.group == 0x70 and c.sub == 2 and not c.is_meta]
            heads = [c for c in chunks if c.group == 0x70 and c.sub == 1 and not c.is_meta]
            for k, c in enumerate(heads):
                head = ct._read_at(f, c.abs, c.size)
                if len(head) < 16:
                    continue
                frames, nodes = struct.unpack(">II", head[8:16])
                name = ""
                if k < len(name_chunks):
                    name = ct._read_at(f, name_chunks[k].abs, name_chunks[k].size).split(b"\0")[0] \
                        .decode("latin1", "replace")
                hs = next(((h, r) for h, r in hier if len(h) == nodes), (None, None))
                out.append(ClipInfo(path, rel or path, si, order, name or f"clip {k}", frames, nodes, hs[0], hs[1]))
                order += 1
    return out


_CLIP_CACHE = {}


def load_clip(info):
    key = (info.src, info.sub_index)
    if key not in _CLIP_CACHE:
        if len(_CLIP_CACHE) > 8:
            _CLIP_CACHE.pop(next(iter(_CLIP_CACHE)))
        _CLIP_CACHE[key] = read_clips(info.src, info.sub_index)
    clips = _CLIP_CACHE[key]
    return clips[info.order] if info.order < len(clips) else None


def match(model, info):
    """How a clip maps onto a model's skeleton: "same" (this rig), "shared"
    (another rig whose bones the model contains, matched by bone hash),
    or None (not compatible)."""
    H = [r[0] for r in model.hierarchy]
    if not H:
        return None
    if info.hashes is not None:
        if list(info.hashes) == H:
            return "same"
        hs = set(H)
        if sum(h in hs for h in info.hashes) >= 0.9 * len(info.hashes) and len(info.hashes) >= 8:
            return "shared"
        return None
    # no hierarchy next to the clip (e.g. transformation files): same node
    # count and the clip is named after the model's file
    base = model.rel.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
    if info.nodes == len(H) and base.startswith(info.name.lower()):
        return "same"
    return None


def _qrow(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
                     [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
                     [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)]])


class Animator:
    """Poses a model's skeleton with a clip. pose() returns one skinning
    matrix per hierarchy node (row vectors: v' = v @ M), i.e. inverse bind x
    animated world, so bind-pose vertices land in the animated pose."""

    def __init__(self, model):
        self.model = model
        self.H = model.hierarchy
        self.n = len(self.H)
        self.node_of = {r[0]: i for i, r in enumerate(self.H)}
        ibm = {h: m for h, m in model.bones}
        self.ibm = [ibm.get(r[0]) for r in self.H]
        # rest pose (= bind pose up to a turn about the vertical axis)
        self.rest = self._worlds([(r[3], r[2], 1.0) for r in self.H])
        k = next((i for i, m in enumerate(self.ibm) if m is not None), None)
        self.first_bound = k
        self.G0 = np.eye(4)
        if k is not None:
            self.G0 = np.linalg.inv(self.rest[k]) @ np.linalg.inv(self.ibm[k])
        self.clip = self.info = None
        self.mapping = None
        self.G = self.G0
        self.base = None          # additive: local matrices of the pose the clip is layered on

    def _worlds(self, locals_):
        W = [None] * self.n
        for i, loc in enumerate(locals_):
            L = np.eye(4)
            if isinstance(loc[0], str):
                L[:3, :3], L[3, :3] = loc[1], loc[2]
            else:
                q, t, sc = loc
                L[:3, :3] = _qrow(q) * sc
                L[3, :3] = t
            p = self.H[i][1]
            W[i] = L if p < 0 else L @ W[p]
        return W

    def set_clip(self, clip, info, kind):
        self.clip, self.info, self.kind = clip, info, kind
        if clip is None:
            return
        if kind == "same" or info.hashes is None:
            self.mapping = list(range(self.n)) if clip.nodes == self.n else [None] * self.n
        else:
            idx = {h: i for i, h in enumerate(info.hashes)}
            self.mapping = [idx.get(r[0]) for r in self.H]
        # hip height of a shared rig -> this model's proportions
        self.hip_scale = 1.0
        if kind == "shared" and info.rig_rest is not None:
            for i, r in enumerate(self.H[:4]):
                ci = self.mapping[i]
                if ci is not None and r[1] >= 0:
                    a, b = np.linalg.norm(r[2]), np.linalg.norm(info.rig_rest[ci][1])
                    if a > 1e-6 and b > 1e-6:
                        self.hip_scale = a / b
                        break
        self.G = self._orientation()
        self.G = self._centre()

    def _locals(self, frame):
        clip, out = self.clip, []
        f0 = int(np.floor(frame))
        a = frame - f0
        f1 = f0 + 1 if f0 + 1 < clip.frames else f0
        for i, r in enumerate(self.H):
            q, t = np.array(r[3], float), np.array(r[2], float)
            sc = float(np.mean(r[4])) if self.kind == "same" else 1.0
            ci = self.mapping[i] if self.mapping else None
            if ci is not None:
                ss = clip.scale.get(ci)
                if ss is not None and self.kind == "same":
                    sc = ss[min(f0, len(ss) - 1)] * (1 - a) + ss[min(f1, len(ss) - 1)] * a
                rq = clip.rot.get(ci)
                if rq is not None:
                    q0, q1 = rq[min(f0, len(rq) - 1)], rq[min(f1, len(rq) - 1)]
                    if np.dot(q0, q1) < 0:
                        q1 = -q1
                    q = q0 * (1 - a) + q1 * a
                    q /= max(np.linalg.norm(q), 1e-9)
                tt = clip.trans.get(ci)
                if tt is not None:
                    t_ = tt[min(f0, len(tt) - 1)] * (1 - a) + tt[min(f1, len(tt) - 1)] * a
                    if self.kind == "same":
                        t = t_
                    elif i <= 2 or r[1] < 0:       # root / hips of a shared rig: scale to this model
                        t = t_ * self.hip_scale
            out.append((q, t, sc))
        return out

    def _orientation(self):
        """Fixed mapping from the clip's space to the model's bind space.
        Own rig: from the rest pose (rest = bind up to a turn about the
        vertical axis). Shared rig: from that rig's rest pose, mapped onto
        this skeleton. Never derived from the animated pose itself (stances
        differ per clip, e.g. side-on with a heavy weapon)."""
        if self.kind != "shared" or not self.info or not self.info.rig_rest:
            return self.G0
        loc = []
        for i, r in enumerate(self.H):
            ci = self.mapping[i] if self.mapping else None
            if ci is not None and ci < len(self.info.rig_rest):
                _p, t, q = self.info.rig_rest[ci]
                t = np.array(t) * (self.hip_scale if i <= 2 or r[1] < 0 else 1.0)
                if not (i <= 2 or r[1] < 0):
                    t = np.array(r[2], float)
                loc.append((np.array(q, float), t, 1.0))
            else:
                loc.append((np.array(r[3], float), np.array(r[2], float), 1.0))
        W = self._worlds(loc)
        k = self.first_bound
        if k is None:
            return self.G0
        return np.linalg.inv(W[k]) @ np.linalg.inv(self.ibm[k])

    def _centre(self):
        return self._centre_from(self.G)

    def _centre_from(self, G):
        """Shift sideways (not turn) so the first frame stands where the
        model does: some clips carry a horizontal offset on the hips."""
        k = [i for i, m in enumerate(self.ibm) if m is not None]
        if not k or self.clip is None:
            return G
        W = self._worlds(self._locals(0.0))
        A = np.array([(W[i] @ G)[3, :2] for i in k]).mean(0)
        B = np.array([np.linalg.inv(self.ibm[i])[3, :2] for i in k]).mean(0)
        T = np.eye(4)
        T[3, :2] = B - A
        return G @ T

    @staticmethod
    def _mat(loc):
        q, t, sc = loc
        return _qrow(q) * sc, np.asarray(t, float)

    def current_locals(self, frame):
        """Local pose (matrices) shown at a frame - the base for an
        additive clip played next."""
        if self.clip is None:
            return [self._mat((np.array(r[3], float), np.array(r[2], float), 1.0)) for r in self.H]
        if self.base is not None:
            return [(R, t) for _m, R, t in self._additive_locals(frame)]
        return [self._mat(loc) for loc in self._locals(frame)]

    def bind_locals(self):
        """Local pose (matrices) that reproduces the bind pose exactly with
        orientation G = identity: bound bones from their bind matrices,
        helper bones from the rest pose."""
        W = []
        for i in range(self.n):
            W.append(np.linalg.inv(self.ibm[i]) if self.ibm[i] is not None else self.rest[i] @ self.G0)
        out = []
        for i, r in enumerate(self.H):
            L = W[i] @ np.linalg.inv(W[r[1]]) if r[1] >= 0 else W[i]
            out.append((L[:3, :3].copy(), L[3, :3].copy()))
        return out

    def set_base(self, base):
        """Play the current clip additively on top of `base` (from
        current_locals), or normally (None). Additive: each bone moves by
        how the clip changes it relative to the clip's first frame, so held
        bones keep the base pose and animated bones add their motion."""
        self.base = base
        self._first = [self._mat(loc) for loc in self._locals(0.0)] if base is not None else None

    def _additive_locals(self, frame):
        out = []
        for (Rb, tb), (R0, t0), loc in zip(self.base, self._first, self._locals(frame)):
            Rf, tf = self._mat(loc)
            try:
                R = Rf @ np.linalg.inv(R0) @ Rb       # bone-space change since frame 0
            except np.linalg.LinAlgError:
                R = Rb
            out.append(("m", R, tb + (tf - t0)))
        return out

    def pose(self, frame):
        """Skinning matrices (n, 4, 4) for a (fractional) frame."""
        out = np.tile(np.eye(4), (self.n, 1, 1))
        if self.clip is None:
            if self.base is not None:
                W = self._worlds([("m", R, t) for R, t in self.base])
                for i, m in enumerate(self.ibm):
                    if m is not None:
                        out[i] = m @ W[i] @ self.G
            return out
        if self.base is not None:
            W = self._worlds(self._additive_locals(frame))
        else:
            W = self._worlds(self._locals(frame))
        for i, m in enumerate(self.ibm):
            if m is not None:
                out[i] = m @ W[i] @ self.G
        return out


# ------------------------------------------------------------------ export --
def bake_joint_locals(model, joints, info, kind, clip=None):
    """Sample a clip for glTF export, exactly as the viewer plays it (full
    pose, not additive). joints: model.skeleton(). Returns an array
    (frames, len(joints), 4, 4) of each joint's local matrix (row vectors,
    relative to its parent joint), so that joint world x the exported
    inverse bind matrix gives the viewer's skinning matrix."""
    if clip is None:
        clip = load_clip(info)
    if clip is None or not clip.frames:
        return None
    A = Animator(model)
    A.set_clip(clip, info, kind)
    A.set_base(None)
    A.G = A._centre_from(A._orientation())
    node = [A.node_of.get(j["hash"]) if j["hash"] else None for j in joints]
    bind = [j["world"] for j in joints]
    out = np.empty((clip.frames, len(joints), 4, 4))
    for f in range(clip.frames):
        W = A._worlds(A._locals(float(f)))
        J = [W[n] @ A.G if n is not None else bind[k] for k, n in enumerate(node)]
        for k, j in enumerate(joints):
            p = j["parent"]
            out[f, k] = J[k] if p < 0 else J[k] @ np.linalg.inv(J[p])
    return out
