"""
Wwise Vorbis -> standard Ogg Vorbis.

Python port of the conversion in ww2ogg by Adam Gashlin (hcs64),
https://github.com/hcs64/ww2ogg - see LICENSE_ww2ogg.txt. Wwise stores
Vorbis audio with stripped headers and codebooks replaced by IDs into a
shared codebook library (packed_codebooks.bin); this rebuilds a normal
Ogg Vorbis stream that any Vorbis decoder can play.

decode_wwise_vorbis(b) -> (int16 pcm (samples, channels), rate) needs the
"soundfile" package (pip install soundfile) for the final Vorbis decoding.
"""

import io
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
CODEBOOK_FILES = ("packed_codebooks.bin", "packed_codebooks_aoTuV_603.bin")


class VorbisError(Exception):
    pass


# ------------------------------------------------------------- bit I/O --
class BitReader:
    """LSB-first bit reader over bytes (Vorbis bit order)."""

    def __init__(self, data, pos=0):
        self.data = data
        self.byte = pos
        self.bit = 0
        self.total = 0

    def read(self, n):
        v = 0
        for i in range(n):
            if self.byte >= len(self.data):
                raise VorbisError("out of bits")
            if (self.data[self.byte] >> self.bit) & 1:
                v |= 1 << i
            self.bit += 1
            if self.bit == 8:
                self.bit = 0
                self.byte += 1
        self.total += n
        return v


_CRC = []
for _i in range(256):
    _r = _i << 24
    for _ in range(8):
        _r = ((_r << 1) ^ 0x04C11DB7) if _r & 0x80000000 else (_r << 1)
    _CRC.append(_r & 0xFFFFFFFF)


def ogg_crc(data):
    c = 0
    for b in data:
        c = ((c << 8) & 0xFFFFFFFF) ^ _CRC[((c >> 24) & 0xFF) ^ b]
    return c


class OggWriter:
    """Bit writer that packs packets into Ogg pages (one packet per page,
    as ww2ogg does)."""

    def __init__(self):
        self.out = io.BytesIO()
        self.acc = 0          # pending bits (LSB first)
        self.nacc = 0
        self.payload = bytearray()
        self.granule = 0
        self.seqno = 0
        self.first = True
        self.continued = False

    def write(self, v, n):
        self.acc |= (v & ((1 << n) - 1)) << self.nacc
        self.nacc += n
        while self.nacc >= 8:
            self.payload.append(self.acc & 0xFF)
            self.acc >>= 8
            self.nacc -= 8

    def write_bytes(self, b):
        if self.nacc == 0:
            self.payload += b
        else:
            v = int.from_bytes(b, "little")
            self.write(v, len(b) * 8)

    def flush_bits(self):
        if self.nacc:
            self.payload.append(self.acc & 0xFF)
            self.acc = 0
            self.nacc = 0

    def flush_page(self, next_continued=False, last=False):
        self.flush_bits()
        p = bytes(self.payload)
        if not p:
            return
        if len(p) > 255 * 255:
            raise VorbisError("packet too large for one Ogg page")
        segs = len(p) // 255 + 1
        if segs == 256:
            segs = 255
        lace = bytearray()
        left = len(p)
        for _ in range(segs):
            lace.append(255 if left >= 255 else left)
            left -= min(left, 255)
        g = self.granule
        hdr = bytearray(b"OggS\0")
        hdr.append((1 if self.continued else 0) | (2 if self.first else 0) | (4 if last else 0))
        hdr += struct.pack("<IIIII", g & 0xFFFFFFFF, 0xFFFFFFFF if g == 0xFFFFFFFF else 0,
                           1, self.seqno, 0)
        hdr.append(segs)
        page = hdr + lace + p
        struct.pack_into("<I", page, 22, ogg_crc(page))
        self.out.write(page)
        self.seqno += 1
        self.first = False
        self.continued = next_continued
        self.payload = bytearray()

    def vorbis_header(self, t):
        self.write(t, 8)
        for c in b"vorbis":
            self.write(c, 8)


def ilog(v):
    r = 0
    while v:
        r += 1
        v >>= 1
    return r


def quantvals(entries, dims):
    bits = ilog(entries)
    vals = entries >> ((bits - 1) * (dims - 1) // dims)
    while True:
        acc = acc1 = 1
        for _ in range(dims):
            acc *= vals
            acc1 *= vals + 1
        if acc <= entries < acc1:
            return vals
        vals += -1 if acc > entries else 1


# ----------------------------------------------------------- codebooks --
class CodebookLibrary:
    def __init__(self, path):
        d = open(path, "rb").read()
        off_off = struct.unpack("<I", d[-4:])[0]
        n = (len(d) - off_off) // 4
        self.data = d[:off_off]
        self.offsets = list(struct.unpack(f"<{n}I", d[off_off:off_off + 4 * n]))
        self.name = os.path.basename(path)

    def get(self, i):
        if i < 0 or i >= len(self.offsets) - 1:
            return None
        return self.data[self.offsets[i]:self.offsets[i + 1]]

    def rebuild_id(self, i, w):
        cb = self.get(i)
        if cb is None:
            raise VorbisError(f"invalid codebook id {i}")
        r = BitReader(cb)
        rebuild_codebook(r, w)
        # exactly all bytes must be used (an extra 0 byte when bits end on a boundary)
        if r.total // 8 + 1 != len(cb):
            raise VorbisError("codebook size mismatch")


def rebuild_codebook(r, w):
    """Stripped Wwise codebook -> full Vorbis codebook."""
    dims = r.read(4)
    entries = r.read(14)
    w.write(0x564342, 24)
    w.write(dims, 16)
    w.write(entries, 24)
    ordered = r.read(1)
    w.write(ordered, 1)
    if ordered:
        w.write(r.read(5), 5)
        cur = 0
        while cur < entries:
            nb = ilog(entries - cur)
            num = r.read(nb)
            w.write(num, nb)
            cur += num
        if cur > entries:
            raise VorbisError("codebook entry count out of range")
    else:
        cll = r.read(3)
        sparse = r.read(1)
        if cll == 0 or cll > 5:
            raise VorbisError("nonsense codeword length")
        w.write(sparse, 1)
        for _ in range(entries):
            present = 1
            if sparse:
                present = r.read(1)
                w.write(present, 1)
            if present:
                w.write(r.read(cll), 5)
    lookup = r.read(1)
    w.write(lookup, 4)
    if lookup == 1:
        w.write(r.read(32), 32)
        w.write(r.read(32), 32)
        vl = r.read(4)
        w.write(vl, 4)
        w.write(r.read(1), 1)
        for _ in range(quantvals(entries, dims)):
            w.write(r.read(vl + 1), vl + 1)
    elif lookup != 0:
        raise VorbisError("invalid lookup type")


def copy_codebook(r, w):
    """Full (inline) Vorbis codebook -> copied unchanged."""
    ident, dims, entries = r.read(24), r.read(16), r.read(24)
    if ident != 0x564342:
        raise VorbisError("invalid codebook identifier")
    w.write(ident, 24)
    w.write(dims, 16)
    w.write(entries, 24)
    ordered = r.read(1)
    w.write(ordered, 1)
    if ordered:
        w.write(r.read(5), 5)
        cur = 0
        while cur < entries:
            nb = ilog(entries - cur)
            num = r.read(nb)
            w.write(num, nb)
            cur += num
    else:
        sparse = r.read(1)
        w.write(sparse, 1)
        for _ in range(entries):
            present = 1
            if sparse:
                present = r.read(1)
                w.write(present, 1)
            if present:
                w.write(r.read(5), 5)
    lookup = r.read(4)
    w.write(lookup, 4)
    if lookup == 1:
        w.write(r.read(32), 32)
        w.write(r.read(32), 32)
        vl = r.read(4)
        w.write(vl, 4)
        w.write(r.read(1), 1)
        for _ in range(quantvals(entries, dims)):
            w.write(r.read(vl + 1), vl + 1)
    elif lookup != 0:
        raise VorbisError("invalid lookup type")


# ------------------------------------------------------------ the file --
class WwiseVorbis:
    def __init__(self, b):
        self.b = b
        if b[:4] == b"RIFX":
            self.e = ">"
        elif b[:4] == b"RIFF":
            self.e = "<"
        else:
            raise VorbisError("missing RIFF")
        if b[8:12] != b"WAVE":
            raise VorbisError("missing WAVE")
        e = self.e
        riff_size = struct.unpack(e + "I", b[4:8])[0] + 8
        ch = {}
        p = 12
        while p + 8 <= min(riff_size, len(b)):
            cid = b[p:p + 4]
            size = struct.unpack(e + "I", b[p + 4:p + 8])[0]
            ch[cid] = (p + 8, size)
            p += 8 + size
        if b"fmt " not in ch or b"data" not in ch:
            raise VorbisError("expected fmt, data chunks")
        fo, fs = ch[b"fmt "]
        self.data_off, self.data_size = ch[b"data"]
        if b"vorb" in ch:
            vo, vs = ch[b"vorb"]
        elif fs == 0x42:
            vo, vs = fo + 0x18, -1
        else:
            raise VorbisError("expected vorb chunk")
        codec, self.channels, self.rate, self.avg_bps = struct.unpack(e + "HHII", b[fo:fo + 12])
        if codec != 0xFFFF:
            raise VorbisError("not Wwise Vorbis")
        self.loop = None
        if b"smpl" in ch:
            so = ch[b"smpl"][0]
            if struct.unpack(e + "I", b[so + 0x1C:so + 0x20])[0] == 1:
                self.loop = struct.unpack(e + "II", b[so + 0x2C:so + 0x34])
        if vs not in (-1, 0x28, 0x2A, 0x2C, 0x32, 0x34):
            raise VorbisError(f"unsupported vorb size {vs:#x}")
        rd = lambda o: struct.unpack(e + "I", b[vo + o:vo + o + 4])[0]  # noqa: E731
        self.sample_count = rd(0)
        self.no_granule = self.mod_packets = False
        self.triad = self.old_headers = False
        if vs in (-1, 0x2A):
            self.no_granule = True
            if rd(4) not in (0x4A, 0x4B, 0x69, 0x70):
                self.mod_packets = True
            o = 0x10
        else:
            o = 0x18
        self.setup_off, self.first_audio_off = rd(o), rd(o + 4)
        if vs in (0x28, 0x2C):
            self.triad = self.old_headers = True
        else:
            uo = 0x24 if vs in (-1, 0x2A) else 0x2C
            self.uid = rd(uo)
            self.bs0, self.bs1 = b[vo + uo + 4], b[vo + uo + 5]

    def _packet(self, off):
        """-> (payload offset, size, granule, next offset)"""
        e = self.e
        b = self.b
        if self.old_headers:
            size, gran = struct.unpack(e + "II", b[off:off + 8])
            return off + 8, size, gran, off + 8 + size
        if self.no_granule:
            size = struct.unpack(e + "H", b[off:off + 2])[0]
            return off + 2, size, 0, off + 2 + size
        size, gran = struct.unpack(e + "HI", b[off:off + 6])
        return off + 6, size, gran, off + 6 + size

    def to_ogg(self, codebooks):
        w = OggWriter()
        mode_blockflag, mode_bits = None, 0
        if self.triad:
            self._header_triad(w)
        else:
            mode_blockflag, mode_bits = self._header(w, codebooks)
        end = self.data_off + self.data_size
        off = self.data_off + self.first_audio_off
        prev_blockflag = False
        b = self.b
        hsize = 8 if self.old_headers else (2 if self.no_granule else 6)
        while off < end:
            if off + hsize > end:
                raise VorbisError("page header truncated")
            po, size, gran, nxt = self._packet(off)
            if po + size > len(b):
                raise VorbisError("file truncated")
            w.granule = 1 if gran == 0xFFFFFFFF else gran
            pkt = b[po:po + size]
            if self.mod_packets:
                if mode_blockflag is None:
                    raise VorbisError("no mode info for modified packets")
                w.write(0, 1)                       # packet type: audio
                r = BitReader(pkt)
                mode = r.read(mode_bits)
                w.write(mode, mode_bits)
                rem = r.read(8 - mode_bits)
                if mode_blockflag[mode]:
                    nb = False
                    if nxt + hsize <= end:
                        npo, nsize, _, _ = self._packet(nxt)
                        if nsize > 0:
                            nb = mode_blockflag[BitReader(b, npo).read(mode_bits)]
                    w.write(int(prev_blockflag), 1)
                    w.write(int(nb), 1)
                prev_blockflag = mode_blockflag[mode]
                w.write(rem, 8 - mode_bits)
                w.write_bytes(pkt[1:])
            else:
                w.write_bytes(pkt)
            off = nxt
            w.flush_page(False, off == end)
        if off > end:
            raise VorbisError("page truncated")
        return w.out.getvalue()

    def _ident_comment(self, w):
        w.vorbis_header(1)
        w.write(0, 32)
        w.write(self.channels, 8)
        w.write(self.rate, 32)
        w.write(0, 32)
        w.write((self.avg_bps * 8) & 0xFFFFFFFF, 32)
        w.write(0, 32)
        w.write(self.bs0, 4)
        w.write(self.bs1, 4)
        w.write(1, 1)
        w.flush_page()
        w.vorbis_header(3)
        vendor = b"converted from Audiokinetic Wwise by ww2ogg (Python port)"
        w.write(len(vendor), 32)
        for c in vendor:
            w.write(c, 8)
        w.write(0, 32)
        w.write(1, 1)
        w.flush_page()

    def _header(self, w, cbl):
        self._ident_comment(w)
        po, size, gran, nxt = self._packet(self.data_off + self.setup_off)
        if gran != 0:
            raise VorbisError("setup packet granule != 0")
        r = BitReader(self.b[po:po + size])
        w.vorbis_header(5)
        n = r.read(8)
        w.write(n, 8)
        cb_count = n + 1
        # "full setup": complete Vorbis codebooks inline (start with "BCV")
        full = BitReader(self.b[po:po + size], 1).read(24) == 0x564342
        if full:
            for _ in range(cb_count):
                copy_codebook(r, w)
        else:
            if cbl is None:
                raise VorbisError("needs the external codebook library")
            for _ in range(cb_count):
                cbl.rebuild_id(r.read(10), w)
        w.write(0, 6)                 # time domain transforms (placeholder)
        w.write(0, 16)
        if full:
            # the rest of the setup is already in standard form
            while r.total < size * 8:
                w.write(r.read(1), 1)
            w.flush_page()
            if nxt != self.data_off + self.first_audio_off:
                raise VorbisError("first audio packet doesn't follow setup packet")
            return self._mode_flags_unknown()
        # floors
        n = r.read(6)
        w.write(n, 6)
        floor_count = n + 1
        for _ in range(floor_count):
            w.write(1, 16)            # floor type 1
            parts = r.read(5)
            w.write(parts, 5)
            pclass = []
            for _ in range(parts):
                c = r.read(4)
                w.write(c, 4)
                pclass.append(c)
            dims = []
            for _ in range(max(pclass, default=-1) + 1):
                d = r.read(3)
                w.write(d, 3)
                dims.append(d + 1)
                sub = r.read(2)
                w.write(sub, 2)
                if sub:
                    mb = r.read(8)
                    w.write(mb, 8)
                    if mb >= cb_count:
                        raise VorbisError("invalid floor1 masterbook")
                for _ in range(1 << sub):
                    sb = r.read(8)
                    w.write(sb, 8)
                    if sb - 1 >= cb_count:
                        raise VorbisError("invalid floor1 subclass book")
            w.write(r.read(2), 2)     # multiplier
            rb = r.read(4)
            w.write(rb, 4)
            for c in pclass:
                for _ in range(dims[c]):
                    w.write(r.read(rb), rb)
        # residues
        n = r.read(6)
        w.write(n, 6)
        residue_count = n + 1
        for _ in range(residue_count):
            rt = r.read(2)
            w.write(rt, 16)
            if rt > 2:
                raise VorbisError("invalid residue type")
            w.write(r.read(24), 24)
            w.write(r.read(24), 24)
            w.write(r.read(24), 24)
            ncls = r.read(6)
            w.write(ncls, 6)
            book = r.read(8)
            w.write(book, 8)
            if book >= cb_count:
                raise VorbisError("invalid residue classbook")
            cascade = []
            for _ in range(ncls + 1):
                lo = r.read(3)
                w.write(lo, 3)
                flag = r.read(1)
                w.write(flag, 1)
                hi = 0
                if flag:
                    hi = r.read(5)
                    w.write(hi, 5)
                cascade.append(hi * 8 + lo)
            for c in cascade:
                for k in range(8):
                    if c & (1 << k):
                        rbk = r.read(8)
                        w.write(rbk, 8)
                        if rbk >= cb_count:
                            raise VorbisError("invalid residue book")
        # mappings
        n = r.read(6)
        w.write(n, 6)
        mapping_count = n + 1
        for _ in range(mapping_count):
            w.write(0, 16)
            sf = r.read(1)
            w.write(sf, 1)
            submaps = 1
            if sf:
                s = r.read(4)
                w.write(s, 4)
                submaps = s + 1
            sp = r.read(1)
            w.write(sp, 1)
            if sp:
                steps = r.read(8)
                w.write(steps, 8)
                nb = ilog(self.channels - 1)
                for _ in range(steps + 1):
                    mag, ang = r.read(nb), r.read(nb)
                    w.write(mag, nb)
                    w.write(ang, nb)
                    if mag == ang or mag >= self.channels or ang >= self.channels:
                        raise VorbisError("invalid coupling")
            res = r.read(2)
            w.write(res, 2)
            if res:
                raise VorbisError("mapping reserved field nonzero")
            if submaps > 1:
                for _ in range(self.channels):
                    mux = r.read(4)
                    w.write(mux, 4)
                    if mux >= submaps:
                        raise VorbisError("mapping_mux >= submaps")
            for _ in range(submaps):
                w.write(r.read(8), 8)
                fl = r.read(8)
                w.write(fl, 8)
                if fl >= floor_count:
                    raise VorbisError("invalid floor mapping")
                rs = r.read(8)
                w.write(rs, 8)
                if rs >= residue_count:
                    raise VorbisError("invalid residue mapping")
        # modes
        n = r.read(6)
        w.write(n, 6)
        mode_count = n + 1
        blockflags = []
        mode_bits = ilog(mode_count - 1)
        for _ in range(mode_count):
            bf = r.read(1)
            w.write(bf, 1)
            blockflags.append(bool(bf))
            w.write(0, 16)
            w.write(0, 16)
            mp = r.read(8)
            w.write(mp, 8)
            if mp >= mapping_count:
                raise VorbisError("invalid mode mapping")
        w.write(1, 1)
        w.flush_page()
        if (r.total + 7) // 8 != size:
            raise VorbisError("did not read exactly the setup packet")
        if nxt != self.data_off + self.first_audio_off:
            raise VorbisError("first audio packet doesn't follow setup packet")
        return blockflags, mode_bits

    def _mode_flags_unknown(self):
        if self.mod_packets:
            raise VorbisError("modified packets with a full setup header are not supported")
        return None, 0

    def _header_triad(self, w):
        off = self.data_off + self.setup_off
        for t in (1, 3):
            po, size, gran, nxt = self._packet(off)
            pkt = self.b[po:po + size]
            if gran != 0 or not pkt or pkt[0] != t:
                raise VorbisError("bad header packet")
            w.write_bytes(pkt)
            w.flush_page()
            off = nxt
        po, size, gran, nxt = self._packet(off)
        r = BitReader(self.b[po:po + size])
        if r.read(8) != 5:
            raise VorbisError("wrong type for setup packet")
        w.write(5, 8)
        for _ in range(6):
            w.write(r.read(8), 8)
        n = r.read(8)
        w.write(n, 8)
        for _ in range(n + 1):
            copy_codebook(r, w)
        while r.total < size * 8:
            w.write(r.read(1), 1)
        w.flush_page()


_LIBS = {}


def _library(name):
    if name not in _LIBS:
        _LIBS[name] = CodebookLibrary(os.path.join(HERE, name))
    return _LIBS[name]


def to_ogg(b):
    """Wwise Vorbis file bytes -> (Ogg Vorbis bytes, WwiseVorbis info).
    Tries each codebook library; the setup header only parses exactly with
    the one it was made with."""
    wv = WwiseVorbis(b)
    if wv.triad:
        return wv.to_ogg(None), wv
    last = None
    try:
        return wv.to_ogg(None), wv          # full setup: no library needed
    except VorbisError as ex:
        if "external codebook" not in str(ex):
            raise
    for name in CODEBOOK_FILES:
        if not os.path.exists(os.path.join(HERE, name)):
            continue
        try:
            return wv.to_ogg(_library(name)), wv
        except VorbisError as ex:
            last = ex
    raise VorbisError(f"could not rebuild the Vorbis setup ({last})" if last
                      else "packed_codebooks.bin is missing (keep it next to the scripts)")


def decode_wwise_vorbis(b):
    """-> (int16 numpy array (samples, channels), sample rate, info)"""
    ogg, wv = to_ogg(b)
    try:
        import soundfile as sf
    except ImportError:
        raise VorbisError("decoding Vorbis needs the soundfile package: pip install soundfile")
    # decode as float and clip: libsndfile's own int16 conversion wraps
    # around on samples that overshoot full scale (audible clicks)
    import numpy as np
    fl, rate = sf.read(io.BytesIO(ogg), dtype="float32", always_2d=True)
    pcm = np.clip(np.round(fl * 32768.0), -32768, 32767).astype(np.int16)
    if wv.sample_count and len(pcm) > wv.sample_count:
        pcm = pcm[:wv.sample_count]
    return pcm, rate, wv
