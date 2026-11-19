#!/usr/bin/env python3
"""
Transformers: Cybertron Adventures (Wii) - audio converter

The game's .wav files are not normal WAVs: they are big-endian "RIFX" files
holding Nintendo DSP ADPCM audio (format tag 0xFFF0, 4 bits per sample) with
the decoder coefficients in a custom "WiiH" chunk. Media players and FFmpeg
cannot play them. This tool converts them to standard 16-bit PCM WAV files,
keeping the loop points (smpl chunk).

Usage:
    python cybertron_audio.py <files or folders> [-o output_folder]
    python cybertron_audio.py            (asks for a folder)

Folders are searched recursively for .wav files. Output keeps the folder
structure and goes to "converted_audio" next to this script unless -o is
given. Files that are already normal WAVs are skipped.
"""

import argparse
import os
import struct
import sys

import numpy as np

DSP_TAG = 0xFFF0
WIIH_SIZE = 0x2E      # per channel: 16 s16 coefs, gain, ps, yn1, yn2, loop ps, loop yn1, loop yn2


class AudioError(Exception):
    pass


def _chunks(b, endian):
    p = 12
    while p + 8 <= len(b):
        cid = b[p:p + 4]
        size = struct.unpack(endian + "I", b[p + 4:p + 8])[0]
        yield cid, p + 8, min(size, len(b) - p - 8)
        p += 8 + size + (size & 1)


def _smpl_loops(b, ch, e):
    loops = []
    if b"smpl" in ch:
        so, ss = ch[b"smpl"]
        if ss >= 36:
            n = struct.unpack(e + "I", b[so + 28:so + 32])[0]
            for i in range(min(n, (ss - 36) // 24)):
                loops.append(struct.unpack(e + "6I", b[so + 36 + 24 * i: so + 60 + 24 * i]))
    return loops


def parse(b, allow_standard=False):
    """-> dict(codec, channels, rate, samples, coefs[ch], hist[ch], data, loops).
    codec "dsp": the game's Wii DSP ADPCM. codec "pcm": plain 16-bit PCM
    (big-endian RIFX, or a standard RIFF if allow_standard)."""
    if b[:4] == b"RIFF" and b[8:12] == b"WAVE":
        if not allow_standard:
            raise AudioError("already a standard WAV file")
        e = "<"
    elif b[:4] == b"RIFX" and b[8:12] == b"WAVE":
        e = ">"
    else:
        raise AudioError("not a RIFX/WAVE file")
    ch = {}
    for cid, off, size in _chunks(b, e):
        ch.setdefault(cid, (off, size))
    if b"fmt " not in ch or b"data" not in ch:
        raise AudioError("missing fmt or data chunk")
    fo, fs = ch[b"fmt "]
    tag, nch, rate, field, align, bits = struct.unpack(e + "HHIIHH", b[fo:fo + 16])
    if tag in (1, 0xFFFE) and bits == 16 and 1 <= nch <= 8:
        do, ds = ch[b"data"]
        return dict(codec="pcm", endian=e, channels=nch, rate=rate,
                    samples=ds // (2 * nch), data=b[do:do + ds], loops=_smpl_loops(b, ch, e))
    if tag == 0xFFFF:
        raise AudioError("Wwise Vorbis audio - not supported")
    if tag != DSP_TAG or bits != 4:
        raise AudioError(f"unsupported audio format 0x{tag:04X}, {bits} bits")
    if nch < 1 or nch > 8:
        raise AudioError(f"bad channel count {nch}")
    if b"WiiH" not in ch:
        raise AudioError("missing WiiH (DSP coefficient) chunk")
    wo, ws = ch[b"WiiH"]
    if ws < WIIH_SIZE * nch:
        raise AudioError(f"WiiH chunk too small for {nch} channel(s)")
    coefs, hist = [], []
    for c in range(nch):
        v = struct.unpack(">16h7h", b[wo + c * WIIH_SIZE: wo + (c + 1) * WIIH_SIZE])
        coefs.append(v[:16])
        hist.append((v[18], v[19]))        # yn1, yn2
    do, ds = ch[b"data"]
    data = b[do:do + ds]
    # the fmt "bytes per second" field holds the sample count (per channel)
    frames_per_ch = ds // (8 * nch)
    samples = field if 0 < field <= frames_per_ch * 14 else frames_per_ch * 14
    loops = _smpl_loops(b, ch, ">")
    return dict(codec="dsp", channels=nch, rate=rate, samples=samples, coefs=coefs, hist=hist,
                data=data, align=align, loops=loops)


def decode_dsp(frames, coefs, nsamples, yn1=0, yn2=0):
    """Nintendo DSP ADPCM: 8-byte frames, 1 header byte (predictor index in
    the high nibble, scale exponent in the low one) + 14 4-bit samples."""
    nframes = len(frames) // 8
    a = np.frombuffer(frames[:nframes * 8], np.uint8).reshape(nframes, 8)
    head = a[:, 0]
    nib = np.empty((nframes, 14), np.int32)
    nib[:, 0::2] = a[:, 1:] >> 4
    nib[:, 1::2] = a[:, 1:] & 15
    nib = np.where(nib >= 8, nib - 16, nib)
    scaled = (nib << (head & 15)[:, None].astype(np.int32)) << 11     # sample * scale, in 1/2048
    pred = np.minimum(head >> 4, 7)
    c1 = np.array(coefs[0::2], np.int64)[pred].tolist()
    c2 = np.array(coefs[1::2], np.int64)[pred].tolist()
    sc = scaled.tolist()
    out = [0] * (nframes * 14)
    k = 0
    for fi in range(nframes):
        a1, a2, row = c1[fi], c2[fi], sc[fi]
        for s in row:
            v = (s + a1 * yn1 + a2 * yn2 + 1024) >> 11
            v = -32768 if v < -32768 else (32767 if v > 32767 else v)
            out[k] = v
            k += 1
            yn2, yn1 = yn1, v
    return np.array(out[:nsamples], np.int16)


def decode(info, in_bank=False):
    """PCM (samples, channels). Multi-channel layout depends on where the
    sound is stored: streamed (loose) files interleave the channels every
    8-byte frame, sounds inside a soundbank store each channel as one block.
    (Both verified on the game's stereo sounds.)"""
    nch = info["channels"]
    data = info["data"]
    if info["codec"] == "pcm":
        n = len(data) // (2 * nch) * 2 * nch
        return np.frombuffer(data[:n], info["endian"] + "i2").astype(np.int16).reshape(-1, nch)
    if nch == 1:
        chans = [data]
    elif in_bank:
        per = len(data) // nch // 8 * 8
        chans = [data[c * per:(c + 1) * per] for c in range(nch)]
    else:
        nfr = len(data) // (8 * nch)
        a = np.frombuffer(data[:nfr * 8 * nch], np.uint8).reshape(nfr, nch, 8)
        chans = [a[:, c].tobytes() for c in range(nch)]
    pcm = [decode_dsp(chans[c], info["coefs"][c], info["samples"], *info["hist"][c]) for c in range(nch)]
    n = min(len(p) for p in pcm)
    return np.stack([p[:n] for p in pcm], axis=1)       # (samples, channels)


def write_wav(path, pcm, rate, loops=()):
    """Standard little-endian 16-bit PCM WAV; loop points go in a smpl chunk."""
    nch = pcm.shape[1]
    body = pcm.astype("<i2").tobytes()
    chunks = b"fmt " + struct.pack("<IHHIIHH", 16, 1, nch, rate, rate * nch * 2, nch * 2, 16)
    chunks += b"data" + struct.pack("<I", len(body)) + body
    if loops:
        smpl = struct.pack("<9I", 0, 0, int(1e9 / rate), 60, 0, 0, 0, len(loops), 0)
        for lp in loops:
            smpl += struct.pack("<6I", *lp)
        chunks += b"smpl" + struct.pack("<I", len(smpl)) + smpl
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)


def is_wwise_vorbis(b):
    if len(b) < 22 or b[:4] not in (b"RIFX", b"RIFF") or b[8:16] != b"WAVEfmt ":
        return False
    return struct.unpack((">" if b[:4] == b"RIFX" else "<") + "H", b[20:22])[0] == 0xFFFF


def load_bytes(b, in_bank=False):
    """Any supported sound (game DSP .wav, Wwise Vorbis, RIFX PCM, standard
    WAV) -> (pcm int16 array (samples, channels), info). in_bank: the bytes
    came from inside a .bnk soundbank (changes the DSP stereo layout)."""
    if is_wwise_vorbis(b):
        import cybertron_wwvorbis as wwv
        try:
            pcm, rate, wv = wwv.decode_wwise_vorbis(b)
        except wwv.VorbisError as ex:
            raise AudioError(f"Wwise Vorbis: {ex}")
        loops = []
        if wv.loop:
            loops = [(0, 0, wv.loop[0], wv.loop[1], 0, 0)]
        return pcm, dict(codec="vorbis", channels=pcm.shape[1], rate=rate,
                         samples=len(pcm), loops=loops)
    info = parse(b, allow_standard=True)
    return decode(info, in_bank), info


def duration_from_header(b):
    """Length in seconds from the first few hundred bytes of a sound, or None."""
    if len(b) < 12 or b[8:12] != b"WAVE" or b[:4] not in (b"RIFF", b"RIFX"):
        return None
    e = ">" if b[:4] == b"RIFX" else "<"
    fmt = data = None
    p = 12
    while p + 8 <= len(b):
        cid, size = b[p:p + 4], struct.unpack(e + "I", b[p + 4:p + 8])[0]
        if cid == b"fmt " and p + 24 <= len(b):
            fmt = struct.unpack(e + "HHIIHH", b[p + 8:p + 24])
        elif cid == b"data":
            data = size
            break
        p += 8 + size + (size & 1)
    if not fmt or not fmt[2]:
        return None
    tag, nch, rate, field, align, bits = fmt
    if tag == DSP_TAG:
        return field / rate
    if tag == 0xFFFF:                       # Wwise Vorbis: sample count in "vorb"
        p = 12
        while p + 12 <= len(b):
            cid, size = b[p:p + 4], struct.unpack(e + "I", b[p + 4:p + 8])[0]
            if cid == b"vorb":
                return struct.unpack(e + "I", b[p + 8:p + 12])[0] / rate
            if cid == b"data":
                break
            p += 8 + size + (size & 1)
        return None
    if data is not None and bits and nch:
        return data / (nch * bits / 8) / rate
    return None


def convert(src, dst):
    with open(src, "rb") as f:
        b = f.read()
    if b[:4] == b"RIFF" and not is_wwise_vorbis(b):
        raise AudioError("already a standard WAV file")
    pcm, info = load_bytes(b)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    write_wav(dst, pcm, info["rate"], info["loops"])
    return info, pcm


# --------------------------------------------------------------- Wwise ---
# The game's audio was built with Wwise. Loose sounds are named <id>.wav;
# art/WwiseAudio/SoundbanksInfo.xml maps every id to its original file name
# and lists which soundbank (.bnk) uses it. Short sounds are stored inside
# the .bnk files themselves (DIDX = index, DATA = the sound files).

def parse_bnk(b, expect_id=None):
    """Sounds stored in a Wwise soundbank -> {id: (offset, size)} (offsets
    into b). The byte order is detected from the BKHD chunk: its bank id
    must match the file name / XML id when that is known."""
    if b[:4] != b"BKHD":
        raise AudioError("not a Wwise soundbank")
    best = None
    for e in (">", "<"):
        size = struct.unpack(e + "I", b[4:8])[0]
        if 8 <= size + 8 <= len(b) and size < 0x1000:
            bank_id = struct.unpack(e + "I", b[12:16])[0] if size >= 8 else None
            score = (expect_id is not None and bank_id == expect_id) * 2 + 1
            if best is None or score > best[0]:
                best = (score, e)
    if best is None:
        raise AudioError("unreadable soundbank header")
    e = best[1]
    chunks = {}
    p = 0
    while p + 8 <= len(b):
        tag, size = b[p:p + 4], struct.unpack(e + "I", b[p + 4:p + 8])[0]
        if size > len(b) - p - 8:
            break
        chunks.setdefault(tag, (p + 8, size))
        p += 8 + size
    out = {}
    if b"DIDX" in chunks and b"DATA" in chunks:
        io, isz = chunks[b"DIDX"]
        do, dsz = chunks[b"DATA"]
        for k in range(isz // 12):
            sid, off, size = struct.unpack(e + "3I", b[io + 12 * k: io + 12 * k + 12])
            if off + size <= dsz:
                out[sid] = (do + off, size)
    return out


# speaker codes used in dialogue file names (DIA_<code>_...). Only codes that
# are clear from the names are spelled out; the rest stay as codes.
SPEAKERS = {
    "OPT": "Optimus Prime", "BUM": "Bumblebee", "IRO": "Ironhide", "SID": "Sideswipe",
    "RAT": "Ratchet", "AIR": "Air Raid", "JET": "Jetfire", "MEG": "Megatron",
    "STA": "Starscream", "SKY": "Skywarp", "THU": "Thundercracker", "SOU": "Soundwave",
    "BAR": "Barricade", "TRY": "Trypticon", "BUMBEE": "Bumblebee", "IRNHDE": "Ironhide",
}


def _speaker(code):
    return SPEAKERS.get(code.upper(), code.upper())


def categorise(name):
    """Original file name -> (category, subgroup, ...) for the browser tree."""
    base = os.path.splitext(name)[0]
    t = base.split("_")
    u = [x.upper() for x in t]
    p = u[0]
    sub = t[1] if len(t) > 1 else ""
    if p == "DIA" and len(t) > 2:
        return ("Dialogue", _speaker(t[1]), t[2].upper())
    if p == "DX" and len(t) > 1:
        return ("Dialogue", _speaker(t[1]), "DX")
    if p == "PLAY" and len(u) > 1 and u[1] == "DIA":
        return ("Dialogue", "Narrator", "Mission intros")
    if p == "NIS":
        return ("Cutscenes", sub.upper() or "Other")
    if p in ("MUZ", "MUS"):
        return ("Music", sub or "Other")
    if p == "AMB":
        return ("Ambience", sub.upper() or "Other")
    if p == "ENV":
        return ("Ambience", "Environment effects")
    if p == "CHAR":
        return ("Characters", _speaker(sub))
    if p == "ENE":
        s2 = {"DESTOYER": "Destroyer"}.get(sub.upper(), sub)
        return ("Enemies", s2 or "Other")
    if p == "GEN":
        return ("Weapons & general",)
    if p in ("PROP", "PROPS", "PRES", "ALLY"):
        return ("Props & scripted events",)
    if p == "FE":
        return ("Interface", "Front end")
    if p == "HUD":
        return ("Interface", "HUD")
    if p in ("PIP", "SQUELCH"):
        return ("Interface", "Radio")
    if p == "TEST":
        return ("Test tones",)
    return ("Other",)


class Sound:
    __slots__ = ("id", "name", "lang", "where", "banks", "file", "offset", "size",
                 "category", "duration", "standin")

    def __init__(self, sid, name, lang, where):
        self.id, self.name, self.lang, self.where = sid, name, lang, where
        self.banks = []           # soundbank names that use this sound
        self.file = None          # file holding it (.wav, or the .bnk)
        self.offset, self.size = 0, None
        self.duration = None
        self.standin = False      # localized line that uses the English recording
        self.category = categorise(name)

    def load(self):
        """-> (pcm, info)"""
        return load_bytes(self.read(), in_bank=self.where == "bank")

    def read(self):
        with open(self.file, "rb") as f:
            f.seek(self.offset)
            return f.read() if self.size is None else f.read(self.size)

    def read_header(self, n=512):
        with open(self.file, "rb") as f:
            f.seek(self.offset)
            return f.read(n if self.size is None else min(n, self.size))


def find_wwise_folder(folder):
    """The folder holding SoundbanksInfo.xml at or below `folder` (or above
    it, if the user picked a language subfolder)."""
    folder = os.path.abspath(folder)
    for cand in (folder, os.path.dirname(folder)):
        if os.path.isfile(os.path.join(cand, "SoundbanksInfo.xml")):
            return cand
    for root, dirs, files in os.walk(folder):
        if "SoundbanksInfo.xml" in files:
            return root
        dirs[:] = [d for d in dirs if not d.startswith(".")]
    return None


class WwiseIndex:
    """Every sound listed in SoundbanksInfo.xml, located on disk.
    sounds[lang] -> list of Sound for that language ("SFX" = not localized)."""

    def __init__(self, folder, warn=print):
        import xml.etree.ElementTree as ET
        root_dir = find_wwise_folder(folder)
        if not root_dir:
            raise AudioError("SoundbanksInfo.xml not found (it is in art/WwiseAudio)")
        self.dir = root_dir
        x = ET.parse(os.path.join(root_dir, "SoundbanksInfo.xml")).getroot()
        self.languages = []
        self.sounds = {}
        self.banks = {}            # (lang, bank name) -> [Sound]
        self.missing = []
        by_key = {}

        def lang_dir(lang):
            return root_dir if lang == "SFX" else os.path.join(root_dir, lang)

        def get(sid, name, lang, where):
            key = (lang, sid)
            snd = by_key.get(key)
            if snd is None:
                snd = by_key[key] = Sound(sid, name, lang, where)
                self.sounds.setdefault(lang, []).append(snd)
                if lang != "SFX" and lang not in self.languages:
                    self.languages.append(lang)
            return snd

        for f in x.find("StreamedFiles") or []:
            sid, lang = int(f.get("Id")), f.get("Language")
            snd = get(sid, f.findtext("ShortName") or f"{sid}.wav", lang, "stream")
            snd.standin = f.get("UsingReferenceLanguageAsStandIn") == "true"
            ext = os.path.splitext(f.findtext("Path") or "")[1].lower() or ".wav"
            for d in (lang_dir(lang), root_dir):
                for e in dict.fromkeys((ext, ".wav", ".ogg")):
                    path = os.path.join(d, f"{sid}{e}")
                    if os.path.isfile(path):
                        snd.file = path
                        break
                if snd.file:
                    break

        bank_cache = {}
        for sb in x.find("SoundBanks") or []:
            bname, blang = sb.findtext("ShortName"), sb.get("Language")
            bpath = os.path.join(root_dir, *(sb.findtext("Path") or "").split("\\"))
            members = self.banks.setdefault((blang, bname), [])
            for f in sb.find("ReferencedStreamedFiles") or []:
                snd = by_key.get((blang, int(f.get("Id")))) or by_key.get(("SFX", int(f.get("Id"))))
                if snd is not None:
                    snd.banks.append(bname)
                    members.append(snd)
            full = sb.find("IncludedFullFiles")
            if full is None or not len(full):
                continue
            if bpath not in bank_cache:
                try:
                    with open(bpath, "rb") as fh:
                        bank_cache[bpath] = parse_bnk(fh.read(), int(sb.get("Id")))
                except (OSError, AudioError) as ex:
                    warn(f"{os.path.basename(bpath)} ({bname}): {ex}")
                    bank_cache[bpath] = {}
            idx = bank_cache[bpath]
            for f in full:
                sid = int(f.get("Id"))
                # files carry their own language: sounds inside the
                # per-language mission banks are mostly not localized
                snd = get(sid, f.findtext("ShortName") or f"{sid}.wav", f.get("Language") or blang, "bank")
                snd.banks.append(bname)
                members.append(snd)
                if snd.file is None and sid in idx:
                    snd.file, (snd.offset, snd.size) = bpath, idx[sid]

        for lst in self.sounds.values():
            for snd in lst:
                if snd.file is None:
                    self.missing.append(snd)
        self.languages.sort(key=lambda l: (l != "English(US)", l))

    def for_language(self, lang):
        """Sounds heard when playing in `lang`: its own + the unlocalized ones."""
        return self.sounds.get("SFX", []) + self.sounds.get(lang, [])


def _gather(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in sorted(files):
                    if fn.lower().endswith((".wav", ".ogg")):
                        rel = os.path.relpath(os.path.join(root, fn), p)
                        yield os.path.join(root, fn), os.path.splitext(rel)[0] + ".wav"
        elif os.path.isfile(p):
            yield p, os.path.splitext(os.path.basename(p))[0] + ".wav"


def _pick_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog
        r = tk.Tk()
        r.withdraw()
        d = filedialog.askdirectory(title="Folder with the game's .wav files")
        r.destroy()
        return d or None
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert Cybertron Adventures .wav (RIFX DSP ADPCM) files to standard WAV.")
    ap.add_argument("inputs", nargs="*", help=".wav files or folders")
    ap.add_argument("-o", "--out", help="output folder (default: converted_audio next to this script)")
    a = ap.parse_args(argv)
    interactive = not a.inputs
    if interactive:
        d = _pick_folder()
        if not d:
            print("No folder given.")
            return 2
        a.inputs = [d]
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(here).lower() == "scripts":       # next to the launchers
        here = os.path.dirname(here)
    out = a.out or os.path.join(here, "converted_audio")
    ok = skipped = failed = 0
    for src, rel in _gather(a.inputs):
        dst = os.path.join(out, rel)
        try:
            info, pcm = convert(src, dst)
            ok += 1
            print(f"ok    {rel}  ({info['channels']} ch, {info['rate']} Hz, {len(pcm) / info['rate']:.1f} s)")
        except AudioError as e:
            if "already a standard" in str(e):
                skipped += 1
            else:
                failed += 1
                print(f"FAIL  {rel}: {e}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {rel}: {e!r}")
    print(f"\n{ok} converted, {skipped} already standard, {failed} failed -> {out}")
    if interactive and os.name == "nt":
        input("Press Enter to close...")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
