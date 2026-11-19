#!/usr/bin/env python3
"""
cybertron_tex.py - texture extractor for Transformers: Cybertron Adventures (Wii)

Works on the game's .data resource files (e.g. FEGroupMainUI_Res.data,
zonefile.data, <character>.data) and writes every texture as PNG.

    python3 cybertron_tex.py FILE_OR_DIR [...] [-o OUTDIR] [--mips] [--list]

Requires: Python 3.8+, numpy, Pillow   (pip install numpy pillow)

----------------------------------------------------------------------------
File format notes (all values big-endian)
----------------------------------------------------------------------------
A .data file holds one or more sub-files laid end to end. Each sub-file is
a set of "sections" followed by its own chunk table:

    [section 0][pad][section 1][pad] ... [chunk table][pad to 0x800]

Simple files (UI, characters, levels) have a single sub-file; cutscene files
(e.g. D8_Cin02_shot01.data) have many, and come with a .dict companion file
listing each sub-file's offset, table size and section sizes. The .dict is
used when present; otherwise the sub-files are found by walking backwards
from the end of the file (each table tells us its sub-file's size).

Every section is padded to a multiple of 0x800. A chunk table starts on a
0x800 boundary and is a list of 12-byte records:

    u8  kind      bit 7 = metadata record (no data); bits 4-6 = section index
    u8  ?
    u8  group     0xB6 = textures
    u8  sub       for group 0xB6: 1 = texture headers, 3 = texture pixel data
    u32 size
    u32 offset    relative to the start of the record's section

Section sizes are not stored; a section's extent is the furthest
(offset+size) of any record in it, so the section bases are worked out by
laying the sections out in order and padding each one to 0x800.

Texture header chunks (group 0xB6, sub 1) hold one 0x70-byte header per
texture; each is followed by its pixel-data chunk (group 0xB6, sub 3):

    0x00 u32   name hash
    0x04 u32   1
    0x08 char  "WII\\0"
    0x0C u16   0xFFFF
    0x0E u16   width
    0x10 u16   height
    0x14 char4 encoder tag ("VQ", "VP" = DXT; "USH" = 16-bit)
    0x18 u8    mip level count
    0x20 u32   pixel format: 1 = RGB5A3, 2 = CMPR (GX DXT1)
    0x24 u32   offset of the image (all mips, largest first) in the data chunk

Pixel data is standard GameCube/Wii GX tiled texture data.
"""

import argparse
import os
import struct
import sys

import numpy as np
from PIL import Image

HEADER_SIZE = 0x70
SECTION_ALIGN = 0x800
MAX_TABLE = 0x100000      # how far back to look for a chunk table start
GROUP_TEXTURE = 0xB6
SUB_HEADERS = 1
SUB_DATA = 3

# ---------------------------------------------------------------------------
# GX texture formats: name -> (bits per pixel, block width, block height)
# ---------------------------------------------------------------------------
GX = {
    "I4": (4, 8, 8), "I8": (8, 8, 4), "IA4": (8, 8, 4), "IA8": (16, 4, 4),
    "RGB565": (16, 4, 4), "RGB5A3": (16, 4, 4), "RGBA8": (32, 4, 4),
    "CMPR": (4, 8, 8),
}
# format field at header+0x20 -> GX format (observed in the game's files)
FORMAT_IDS = {1: "RGB5A3", 2: "CMPR", 3: "RGBA8", 7: "IA8"}
# fallback by bits-per-pixel for an unknown format id
BPP_GUESS = {4: "CMPR", 8: "I8", 16: "RGB5A3", 32: "RGBA8"}


def level_size(fmt, w, h):
    bpp, bw, bh = GX[fmt]
    return ((w + bw - 1) // bw * bw) * ((h + bh - 1) // bh * bh) * bpp // 8


def image_size(fmt, w, h, mips):
    return sum(level_size(fmt, max(w >> i, 1), max(h >> i, 1)) for i in range(mips))


# ---------------------------------------------------------------------------
# Decoders: bytes -> HxWx4 uint8 RGBA (returned at padded size, cropped later)
# ---------------------------------------------------------------------------
def _untile(px, w, h, bw, bh):
    """px: (nblocks, bh*bw, C) in block order -> (H, W, C) image."""
    tw, th = (w + bw - 1) // bw, (h + bh - 1) // bh
    c = px.shape[-1]
    img = px.reshape(th, tw, bh, bw, c).transpose(0, 2, 1, 3, 4)
    return img.reshape(th * bh, tw * bw, c)


def _expand(v, bits):
    v = v.astype(np.uint32)
    if bits == 3:
        return ((v << 5) | (v << 2) | (v >> 1)).astype(np.uint8)
    if bits == 4:
        return ((v << 4) | v).astype(np.uint8)
    if bits == 5:
        return ((v << 3) | (v >> 2)).astype(np.uint8)
    if bits == 6:
        return ((v << 2) | (v >> 4)).astype(np.uint8)
    raise ValueError(bits)


def _rgb565(v):
    return np.stack([_expand(v >> 11 & 31, 5), _expand(v >> 5 & 63, 6),
                     _expand(v & 31, 5)], -1)


def decode_rgb5a3(buf, w, h):
    v = np.frombuffer(buf, ">u2").astype(np.uint32)
    opaque = (v & 0x8000) != 0
    r = np.where(opaque, _expand(v >> 10 & 31, 5), _expand(v >> 8 & 15, 4))
    g = np.where(opaque, _expand(v >> 5 & 31, 5), _expand(v >> 4 & 15, 4))
    b = np.where(opaque, _expand(v & 31, 5), _expand(v & 15, 4))
    a = np.where(opaque, 255, _expand(v >> 12 & 7, 3))
    px = np.stack([r, g, b, a], -1).astype(np.uint8).reshape(-1, 16, 4)
    return _untile(px, w, h, 4, 4)


def decode_rgb565(buf, w, h):
    v = np.frombuffer(buf, ">u2")
    rgb = _rgb565(v)
    px = np.concatenate([rgb, np.full(rgb.shape[:-1] + (1,), 255, np.uint8)], -1)
    return _untile(px.reshape(-1, 16, 4), w, h, 4, 4)


def decode_rgba8(buf, w, h):
    b = np.frombuffer(buf, np.uint8).reshape(-1, 2, 16, 2)  # [AR x16][GB x16]
    a, r = b[:, 0, :, 0], b[:, 0, :, 1]
    g, bl = b[:, 1, :, 0], b[:, 1, :, 1]
    return _untile(np.stack([r, g, bl, a], -1), w, h, 4, 4)


def decode_i4(buf, w, h):
    b = np.frombuffer(buf, np.uint8)
    i = np.stack([b >> 4, b & 15], -1).reshape(-1)
    i = _expand(i, 4)
    px = np.stack([i, i, i, i], -1).reshape(-1, 64, 4)
    return _untile(px, w, h, 8, 8)


def decode_i8(buf, w, h):
    i = np.frombuffer(buf, np.uint8)
    px = np.stack([i, i, i, i], -1).reshape(-1, 32, 4)
    return _untile(px, w, h, 8, 4)


def decode_ia4(buf, w, h):
    b = np.frombuffer(buf, np.uint8)
    a, i = _expand(b >> 4, 4), _expand(b & 15, 4)
    px = np.stack([i, i, i, a], -1).reshape(-1, 32, 4)
    return _untile(px, w, h, 8, 4)


def decode_ia8(buf, w, h):
    b = np.frombuffer(buf, np.uint8).reshape(-1, 2)
    a, i = b[:, 0], b[:, 1]
    px = np.stack([i, i, i, a], -1).reshape(-1, 16, 4)
    return _untile(px, w, h, 4, 4)


def decode_cmpr(buf, w, h):
    blk = np.frombuffer(buf, np.uint8).reshape(-1, 8)  # DXT1 blocks, BE
    c0 = blk[:, 0].astype(np.uint32) << 8 | blk[:, 1]
    c1 = blk[:, 2].astype(np.uint32) << 8 | blk[:, 3]
    p0 = _rgb565(c0).astype(np.int32)
    p1 = _rgb565(c1).astype(np.int32)
    four = (c0 > c1)[:, None]
    p2 = np.where(four, (2 * p0 + p1) // 3, (p0 + p1) // 2)
    p3 = np.where(four, (p0 + 2 * p1) // 3, 0)
    alpha = np.full((len(blk), 4), 255, np.int32)
    alpha[:, 3] = np.where(four[:, 0], 255, 0)
    pal = np.stack([p0, p1, p2, p3], 1)                    # (n, 4, 3)
    pal = np.concatenate([pal, alpha[:, :, None]], -1)     # (n, 4, 4)
    bits = blk[:, 4:8]                                     # 4 rows x 1 byte
    idx = np.stack([bits >> 6, bits >> 4 & 3, bits >> 2 & 3, bits & 3], -1)
    idx = idx.reshape(-1, 16).astype(np.intp)              # (n, 16)
    px = np.take_along_axis(pal, idx[:, :, None], 1).astype(np.uint8)  # (n,16,4)
    # 8x8 tile = 2x2 sub-blocks of 4x4
    tw, th = (w + 7) // 8, (h + 7) // 8
    img = px.reshape(th, tw, 2, 2, 4, 4, 4)       # ty tx sy sx py px c
    img = img.transpose(0, 2, 4, 1, 3, 5, 6)      # ty sy py tx sx px c
    return img.reshape(th * 8, tw * 8, 4)


DECODERS = {
    "I4": decode_i4, "I8": decode_i8, "IA4": decode_ia4, "IA8": decode_ia8,
    "RGB565": decode_rgb565, "RGB5A3": decode_rgb5a3, "RGBA8": decode_rgba8,
    "CMPR": decode_cmpr,
}


def decode(fmt, buf, w, h):
    return DECODERS[fmt](buf, w, h)[:h, :w]


# ---------------------------------------------------------------------------
# Container parsing (lazy: only the chunk table and texture headers are read
# while scanning; pixel data is read from disk when a texture is decoded)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Compressed containers (e.g. zonefile_compressed.data): every section is
# zlib-compressed separately as [u32 compressed size][u32 size][zlib data],
# each block padded to 0x800; the uncompressed chunk table follows the last
# block. Decompressing the blocks, padding each to 0x800 and appending the
# table gives back the ordinary .data file byte for byte.
# ---------------------------------------------------------------------------
import io
import zlib

_DECOMP_CACHE = {}
_DECOMP_CACHE_MAX = 3


def _looks_compressed(head, n):
    if len(head) < 10:
        return False
    csz, usz = struct.unpack(">II", head[:8])
    return (8 < csz <= n - 8 and usz >= 1 and head[8] == 0x78
            and (head[8] * 256 + head[9]) % 31 == 0)


def decompress_container(data):
    """Rebuild the plain .data bytes from a compressed container."""
    p, parts = 0, []
    while p + 8 <= len(data):
        csz, usz = struct.unpack(">II", data[p:p + 8])
        if csz == 0 or p + 8 + csz > len(data) or data[p + 8] != 0x78:
            break
        try:
            blk = zlib.decompress(data[p + 8:p + 8 + csz])
        except zlib.error:
            break
        if len(blk) != usz:
            raise ValueError(f"block at 0x{p:x}: size mismatch")
        parts.append(blk + b"\0" * ((-len(blk)) % SECTION_ALIGN))
        p = (p + 8 + csz + SECTION_ALIGN - 1) // SECTION_ALIGN * SECTION_ALIGN
    if not parts:
        raise ValueError("no compressed blocks")
    return b"".join(parts) + data[p:]


def is_compressed_file(path):
    try:
        with open(path, "rb") as f:
            return _looks_compressed(f.read(10), os.fstat(f.fileno()).st_size)
    except OSError:
        return False


def open_data(path):
    """Open a game file for reading; compressed containers are transparently
    decompressed (once, cached) and returned as an in-memory file."""
    f = open(path, "rb")
    try:
        st = os.fstat(f.fileno())
        head = f.read(10)
        if not _looks_compressed(head, st.st_size):
            f.seek(0)
            return f
        key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
        data = _DECOMP_CACHE.get(key)
        if data is None:
            f.seek(0)
            try:
                data = decompress_container(f.read())
            except ValueError:
                f.seek(0)
                return f             # not really compressed: read it as is
            if len(_DECOMP_CACHE) >= _DECOMP_CACHE_MAX:
                _DECOMP_CACHE.pop(next(iter(_DECOMP_CACHE)))
            _DECOMP_CACHE[key] = data
        f.close()
        return io.BytesIO(data)
    except Exception:
        f.close()
        raise


def data_size(f):
    """Size of an open (possibly in-memory) game file."""
    pos = f.tell()
    f.seek(0, 2)
    n = f.tell()
    f.seek(pos)
    return n


class Chunk:
    __slots__ = ("kind", "group", "sub", "size", "offset", "section", "abs")

    def __init__(self, raw):
        self.kind, _, self.group, self.sub, self.size, self.offset = \
            struct.unpack(">BBBBII", raw)
        self.section = (self.kind >> 4) & 7
        self.abs = None

    @property
    def is_meta(self):
        return bool(self.kind & 0x80)


def _read_at(f, off, n):
    f.seek(off)
    return f.read(n)


def _parse_table(blob, n):
    """Parse 12-byte chunk records forward; stops at an all-zero record
    (padding). Returns (chunks, bytes_consumed, ok) - ok is False if a
    record is impossible (so `blob` is probably not a chunk table)."""
    out = []
    p = 0
    while p + 12 <= len(blob):
        rec = blob[p:p + 12]
        if rec == bytes(12):
            break
        size, off = struct.unpack(">II", rec[4:])
        # a zero-size record holds no data; its offset may point anywhere
        if size > n or (size and off > n):
            return out, p, False
        out.append(Chunk(rec))
        p += 12
    return out, p, True


class _TableProbe:
    """Answers "would _parse_table succeed from this offset, and how far
    would it get?" for many candidate offsets of one region, without
    building Chunk objects. Records are 12 bytes, so offsets fall into three
    phases; per phase we precompute, for every record, the index of the next
    all-zero record (end of table) and of the next impossible record."""

    def __init__(self, region, n):
        self.phases = {}
        self.region = region
        self.n = n

    def _phase(self, ph):
        if ph not in self.phases:
            buf = self.region[ph:]
            rows = len(buf) // 12
            a = np.frombuffer(buf, ">u4", rows * 3).reshape(rows, 3) if rows else np.zeros((0, 3), ">u4")
            zero = (a == 0).all(1)
            bad = ((a[:, 1] > self.n) | ((a[:, 2] > self.n) & (a[:, 1] != 0))) & ~zero
            idx = np.arange(rows)
            big = np.int64(rows)

            def next_true(mask):
                v = np.where(mask, idx, big)
                return np.minimum.accumulate(v[::-1])[::-1] if rows else v

            self.phases[ph] = (next_true(zero), next_true(bad), rows)
        return self.phases[ph]

    def test(self, off):
        """-> (record count, bytes used, ok) exactly as _parse_table would."""
        ph = off % 12
        nz, nb, rows = self._phase(ph)
        r0 = (off - ph) // 12
        if r0 >= rows:
            return 0, 0, True
        stop_zero, stop_bad = int(nz[r0]), int(nb[r0])
        if stop_bad < stop_zero:                 # impossible record first
            return stop_bad - r0, (stop_bad - r0) * 12, False
        return stop_zero - r0, (stop_zero - r0) * 12, True


def _layout(chunks, start):
    """Place the sections of one sub-file starting at `start`; fills in
    chunk.abs and returns the total size of the data sections."""
    extent = {}
    for c in chunks:
        # zero-size records hold no data and may point past the section's end
        if not c.is_meta and c.size:
            extent[c.section] = max(extent.get(c.section, 0), c.offset + c.size)
    base, pos = {}, 0
    for s in sorted(extent):
        base[s] = pos
        pos = (pos + extent[s] + SECTION_ALIGN - 1) // SECTION_ALIGN * SECTION_ALIGN
    for c in chunks:
        if not c.is_meta:
            c.abs = start + base.get(c.section, pos) + c.offset
    return pos


def _align_up(v, a=SECTION_ALIGN):
    return (v + a - 1) // a * a


def read_dict(path, n):
    """Read a .dict companion file: a list of sub-files inside the .data.
    Header: u32 hash, u16 version(0x0301), u16 0; then 0x4C-byte entries:
    +0x04 u32 offset of the sub-file in the .data, +0x0C u32 chunk-table
    size, +0x10/+0x18/+0x20/... (u32 size, u32 align) data section sizes,
    section 0 first (empty in most files, but not in e.g. unlockables).
    The chunk table follows the data sections. Returns [(table_pos, table_size)]."""
    try:
        x = open(path, "rb").read()
    except OSError:
        return None
    if len(x) < 8 + 0x4C or (len(x) - 8) % 0x4C not in (0, 4):
        return None
    out = []
    for r in range(8, len(x) - 0x4B, 0x4C):
        v = struct.unpack(">19I", x[r:r + 0x4C])
        off, tsize = v[1], v[3]
        data = sum(v[i] for i in range(4, 19, 2))
        tpos = off + data
        if tsize == 0 or tpos + tsize > n:
            return None
        out.append((tpos, tsize))
    return out


_SUBFILE_CACHE = {}
_SUBFILE_CACHE_MAX = 256


def _cache_key(path, n):
    try:
        st = os.stat(path)
        dp = os.path.splitext(path)[0] + ".dict"
        dm = os.stat(dp).st_mtime_ns if os.path.exists(dp) else 0
        return (os.path.abspath(path), n, st.st_mtime_ns, dm)
    except OSError:
        return None


def find_subfiles(f, n, path=None):
    """Cached wrapper: the texture scan and the model scan both need the
    layout of every file, so it is worked out once per file version."""
    key = _cache_key(path, n) if path else None
    if key is not None and key in _SUBFILE_CACHE:
        return _SUBFILE_CACHE[key]
    subs = _find_subfiles(f, n, path)
    if key is not None:
        if len(_SUBFILE_CACHE) >= _SUBFILE_CACHE_MAX:
            _SUBFILE_CACHE.pop(next(iter(_SUBFILE_CACHE)))
        _SUBFILE_CACHE[key] = subs
    return subs


def _find_subfiles(f, n, path=None):
    """A .data file is one or more sub-files laid end to end; each is
    [data sections, padded to 0x800][chunk table]. Returns a list of
    (start, chunks) in file order, or [] if no consistent layout is found."""
    # 1) companion .dict, if there is one
    if path:
        dp = os.path.splitext(path)[0] + ".dict"
        ents = read_dict(dp, n) if os.path.exists(dp) else None
        if ents:
            subs = []
            for tpos, tsize in ents:
                chunks, _, ok = _parse_table(_read_at(f, tpos, tsize), n)
                data = _layout(chunks, 0)
                start = tpos - data
                if not ok or not chunks or start < 0:
                    subs = None
                    break
                _layout(chunks, start)
                subs.append((start, chunks))
            if subs:
                return sorted(subs, key=lambda s: s[0])

    # 2) walk backwards from EOF, one sub-file at a time
    subs = []
    end = n
    while end > 0:
        # skip zero padding before `end`
        z = end
        while z > 0:
            blk = _read_at(f, max(0, z - 0x1000), min(0x1000, z))
            stripped = blk.rstrip(b"\0")
            if stripped:
                z = z - len(blk) + len(stripped)
                break
            z -= len(blk)
        if z == 0:
            break
        # The table starts on a 0x800 boundary. Records are 12 bytes, so a
        # parse started at the wrong boundary is misaligned and fails; the
        # earliest boundary that parses cleanly up to `z` is the table start.
        lo = max(0, (z - 1) // SECTION_ALIGN * SECTION_ALIGN - MAX_TABLE)
        region = _read_at(f, lo, end - lo)
        probe = _TableProbe(region, n)
        best = None
        s = (z - 1) // SECTION_ALIGN * SECTION_ALIGN
        while s >= lo:
            count, used, ok = probe.test(s - lo)
            if ok and count and z <= s + used <= end:
                best = s
            s -= SECTION_ALIGN
        if best is None:
            break
        tpos = best
        chunks, _used, _ok = _parse_table(region[tpos - lo:], n)
        data = _layout(chunks, 0)
        start = tpos - data
        if data == 0 or start < 0:
            break
        _layout(chunks, start)
        subs.append((start, chunks))
        end = start
    subs.reverse()
    return subs


class Texture:
    """One texture. `file_off` is the absolute file offset of mip 0;
    `avail` is how many bytes the texture may occupy (for format guessing)."""

    def __init__(self, hdr, pos, src=""):
        self.src = src
        self.pos = pos
        self.hash = struct.unpack(">I", hdr[0:4])[0]
        self.width, self.height = struct.unpack(">HH", hdr[0x0E:0x12])
        tag = hdr[0x14:0x17].rstrip(b"\0\t\x01\x02")
        # usually letters ("VP", "USH"); some formats store raw bytes -> show hex
        self.tag = tag.decode("ascii") if all(0x20 < c < 0x7F for c in tag) else tag.hex(" ")
        self.mips = max(hdr[0x18], 1)
        self.fmt_id, self.data_off = struct.unpack(">II", hdr[0x20:0x28])
        self.fmt = FORMAT_IDS.get(self.fmt_id)
        self.data_base = 0       # absolute offset of the data chunk
        self.data_len = 0        # size of the data chunk
        self.guessed = False

    @property
    def file_off(self):
        return self.data_base + self.data_off

    def name(self):
        return f"{self.hash:08x}"

    def mip_dims(self, m):
        return max(self.width >> m, 1), max(self.height >> m, 1)

    def byte_size(self):
        return image_size(self.fmt, self.width, self.height, self.mips) if self.fmt else 0

    def read_mip(self, m=0, f=None):
        """Decode mip level m -> HxWx4 uint8 RGBA numpy array."""
        if not self.fmt:
            raise ValueError(f"unknown pixel format id {self.fmt_id}")
        if not 0 <= m < self.mips:
            raise ValueError("bad mip level")
        off = self.file_off + sum(level_size(self.fmt, *self.mip_dims(i)) for i in range(m))
        w, h = self.mip_dims(m)
        sz = level_size(self.fmt, w, h)
        own = f is None
        if own:
            f = open_data(self.src)
        try:
            buf = _read_at(f, off, sz)
        finally:
            if own:
                f.close()
        if len(buf) < sz:
            raise ValueError("pixel data is truncated")
        return decode(self.fmt, buf, w, h)

    def image(self, m=0, f=None):
        return Image.fromarray(np.ascontiguousarray(self.read_mip(m, f)), "RGBA")


def is_header(d, p):
    return d[p + 8:p + 12] == b"WII\0" and d[p + 4:p + 8] == b"\0\0\0\1"


def _parse_headers(blob, base, src):
    out = []
    for p in range(0, len(blob) - HEADER_SIZE + 1, HEADER_SIZE):
        if is_header(blob, p):
            out.append(Texture(blob[p:p + HEADER_SIZE], base + p, src))
    return out


def _finish_group(texs, data_base, data_len, warn):
    for t in texs:
        t.data_base, t.data_len = data_base, data_len
    order = sorted(texs, key=lambda t: t.data_off)
    for i, t in enumerate(order):
        if t.fmt:
            continue
        end = order[i + 1].data_off if i + 1 < len(order) else data_len
        avail = end - t.data_off
        for f in BPP_GUESS.values():
            if image_size(f, t.width, t.height, t.mips) == avail:
                t.fmt, t.guessed = f, True
                break
        warn(f"texture {t.name()}: unknown format id {t.fmt_id} "
             f"(tag {t.tag}); guessed {t.fmt or 'nothing'}")


def _scan_whole_file(f, n, src, warn):
    """Fallback without a usable chunk table: find runs of texture headers
    and assume the pixel data follows the run (0x20-aligned)."""
    d = _read_at(f, 0, n)
    out, p = [], 0
    while True:
        p = d.find(b"WII\0", p)
        if p < 0:
            return out
        start = p - 8
        if start < 0 or not is_header(d, start):
            p += 4
            continue
        q = start
        while q + HEADER_SIZE <= n and is_header(d, q):
            q += HEADER_SIZE
        texs = _parse_headers(d[start:q], start, src)
        data_start = (q + 0x1F) & ~0x1F
        _finish_group(texs, data_start, n - data_start, warn)
        out += texs
        p = q


def scan_file(path, force_scan=False, allow_full_scan=True, warn=None):
    """Return the list of Textures in a file ([] if it has none).
    allow_full_scan: if the chunk table can't be used, read the whole file and
    search for headers (slow for big non-texture files)."""
    warn = warn or (lambda m: print(f"  ! {m}", file=sys.stderr))
    with open_data(path) as f:
        n = data_size(f)
        if n < HEADER_SIZE:
            return []
        if not force_scan:
            subs = find_subfiles(f, n, path)
            if subs and subs[0][0] != 0:
                warn(f"only part of the file could be mapped (from {subs[0][0]:#x}); "
                     f"textures before that may be missing")
            if subs:
                out = []
                for _, chunks in subs:
                    pending = None
                    for c in chunks:
                        if c.is_meta or c.group != GROUP_TEXTURE:
                            continue
                        if c.sub == SUB_HEADERS:
                            pending = c
                        elif c.sub == SUB_DATA and pending is not None:
                            blob = _read_at(f, pending.abs, pending.size)
                            texs = _parse_headers(blob, pending.abs, path)
                            _finish_group(texs, c.abs, c.size, warn)
                            out += texs
                            pending = None
                return out
        if not allow_full_scan:
            return []
        return _scan_whole_file(f, n, path, warn)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def export_texture(t, dest_dir, base, mips=False, f=None):
    """Write t as PNG(s) into dest_dir; returns the list of paths written."""
    os.makedirs(dest_dir, exist_ok=True)
    paths = []
    for m in range(t.mips if mips else 1):
        p = os.path.join(dest_dir, base + (f"_mip{m}" if m else "") + ".png")
        t.image(m, f).save(p)
        paths.append(p)
    return paths


def unique_names(texs):
    """index_hash names, with a suffix for repeated hashes within one file."""
    seen, names = {}, []
    for i, t in enumerate(texs):
        k = seen.get(t.name(), 0)
        seen[t.name()] = k + 1
        names.append(f"{i:04d}_{t.name()}" + (f"_{k}" if k else ""))
    return names


def process(path, outdir, mips, list_only, force_scan):
    texs = scan_file(path, force_scan)
    print(f"{path}: {len(texs)} texture(s)")
    if not texs:
        return 0
    stem = os.path.splitext(os.path.basename(path))[0]
    dest = os.path.join(outdir, stem)
    written = 0
    with open_data(path) as f:
        for i, (t, base) in enumerate(zip(texs, unique_names(texs))):
            line = (f"  [{i:4d}] {t.name()}  {t.width:4d}x{t.height:<4d} "
                    f"{(t.fmt or '?'):6s} mips={t.mips} tag={t.tag} @{t.pos:#x}")
            if list_only:
                print(line)
                continue
            if not t.fmt:
                print(line + "  SKIPPED (unknown format)", file=sys.stderr)
                continue
            try:
                export_texture(t, dest, base, mips, f)
                written += 1
            except Exception as e:
                print(f"{line}  FAILED ({e})", file=sys.stderr)
    if not list_only:
        print(f"  -> {written} written to {dest}")
    return written


def main():
    ap = argparse.ArgumentParser(
        description="Extract textures from Transformers: Cybertron Adventures "
                    "(Wii) .data files to PNG.")
    ap.add_argument("inputs", nargs="+", help=".data files or folders")
    ap.add_argument("-o", "--out", default="textures_out", help="output folder")
    ap.add_argument("--mips", action="store_true", help="also save mip levels")
    ap.add_argument("--list", action="store_true", help="list only, write nothing")
    ap.add_argument("--scan", action="store_true",
                    help="ignore the chunk table and scan for texture headers")
    a = ap.parse_args()

    files = []
    for inp in a.inputs:
        if os.path.isdir(inp):
            for root, _, names in os.walk(inp):
                files += [os.path.join(root, n) for n in sorted(names)
                          if n.lower().endswith(".data")]
        else:
            files.append(inp)
    total = 0
    for f in files:
        try:
            total += process(f, a.out, a.mips, a.list, a.scan)
        except Exception as e:  # keep going on batch runs
            print(f"{f}: ERROR {e}", file=sys.stderr)
    print(f"done: {total} texture(s) from {len(files)} file(s)")


if __name__ == "__main__":
    main()
