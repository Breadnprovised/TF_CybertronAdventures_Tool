#!/usr/bin/env python3
"""
Transformers: Cybertron Adventures (Wii) - audio browser

Lists every sound in the game, sorted into categories (dialogue by speaker,
music, ambience, characters, enemies, interface...) or by soundbank, using
the names in art/WwiseAudio/SoundbanksInfo.xml. Sounds can be previewed and
exported as standard 16-bit WAV files.

Usage:
    python cybertron_audio_gui.py ["path to the game folder"]

Keys: Space = play / stop, Enter = play, Ctrl+E = export selected,
      click the waveform to play from that point.
"""

import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageTk
except ImportError as ex:
    tk.Tk().withdraw()
    messagebox.showerror("Missing package", f"{ex}\n\nInstall with:\n  pip install numpy pillow")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cybertron_audio as ca  # noqa: E402

APP_TITLE = "Cybertron Adventures - Audio Browser"
WAVE_W, WAVE_H = 640, 170
CACHE_BYTES = 300 * 1024 * 1024


def fmt_time(t):
    if t is None:
        return ""
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:04.1f}"


# ------------------------------------------------------------- playback --
class Player:
    """Plays int16 PCM through the system: winsound on Windows, afplay on
    macOS, paplay/aplay/ffplay on Linux. Audio is handed over as a
    temporary WAV file."""

    def __init__(self):
        self.tmpdir = tempfile.mkdtemp(prefix="cybertron_audio_")
        self.flip = 0
        self.proc = None
        self.playing = False
        self.start_wall = 0.0
        self.start_pos = 0.0
        self.length = 0.0
        self.loop = False
        self.pcm = None
        self.rate = 32000
        self.backend = self._find_backend()

    @staticmethod
    def _find_backend():
        if os.name == "nt":
            try:
                import winsound  # noqa: F401
                return "winsound"
            except ImportError:
                return None
        import shutil
        cands = ["afplay"] if sys.platform == "darwin" else ["paplay", "aplay", "ffplay"]
        for c in cands:
            if shutil.which(c):
                return c
        return None

    def _write(self, pcm):
        self.flip ^= 1
        path = os.path.join(self.tmpdir, f"preview{self.flip}.wav")
        ca.write_wav(path, pcm, self.rate)
        return path

    def play(self, pcm, rate, pos=0.0, loop=False):
        """Play from `pos` seconds."""
        self.stop()
        if self.backend is None:
            raise RuntimeError("No audio player found on this system")
        self.pcm, self.rate, self.loop = pcm, rate, loop
        self.length = len(pcm) / rate
        pos = max(0.0, min(pos, self.length))
        start = int(pos * rate)
        path = self._write(pcm[start:] if start else pcm)
        whole_loop = loop and start == 0
        if self.backend == "winsound":
            import winsound
            flags = winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT
            if whole_loop:
                flags |= winsound.SND_LOOP
            winsound.PlaySound(path, flags)
        else:
            cmd = {"afplay": ["afplay", path], "paplay": ["paplay", path],
                   "aplay": ["aplay", "-q", path],
                   "ffplay": ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]}[self.backend]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.whole_loop = whole_loop
        self.playing = True
        self.start_wall = time.perf_counter()
        self.start_pos = pos

    def stop(self):
        if self.backend == "winsound":
            try:
                import winsound
                winsound.PlaySound(None, 0)
            except Exception:
                pass
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None
        self.playing = False

    def position(self):
        """Current position in seconds, or None when stopped. Also restarts
        looping sounds that were started part-way through."""
        if not self.playing:
            return None
        el = time.perf_counter() - self.start_wall
        pos = self.start_pos + el
        if pos >= self.length:
            if self.loop:
                if self.backend == "winsound" and self.whole_loop:
                    return (pos % self.length) if self.length else 0.0
                self.play(self.pcm, self.rate, 0.0, True)     # loop from the start
                return 0.0
            self.playing = False
            self.proc = None
            return None
        return pos

    def close(self):
        self.stop()
        try:
            for f in os.listdir(self.tmpdir):
                os.remove(os.path.join(self.tmpdir, f))
            os.rmdir(self.tmpdir)
        except OSError:
            pass


# ------------------------------------------------------------------ app --
class App(tk.Tk):
    def __init__(self, start_folder=None):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1400x800")
        self.minsize(900, 500)
        self.index = None
        self.leaf = {}               # tree iid -> Sound
        self.cache = {}              # (file, offset) -> (pcm, info)
        self.cache_order = []
        self.current = None          # Sound shown on the right
        self.current_pcm = None
        self.wave_img = None
        self.q = queue.Queue()
        self.busy = False
        self.stop_flag = threading.Event()
        self.player = Player()
        self.dur_job = None
        self._build_ui()
        self.after(50, self._poll)
        self.after(40, self._tick)
        self.protocol("WM_DELETE_WINDOW", self._close)
        if start_folder:
            self.after(100, lambda: self.open_folder(start_folder))
        else:
            self.status.config(text="Open the game folder (or its art/WwiseAudio folder) to begin.")

    # ---------------------------------------------------------- layout --
    def _build_ui(self):
        menu = tk.Menu(self)
        fm = tk.Menu(menu, tearoff=0)
        fm.add_command(label="Open game folder...", command=self.ask_folder)
        fm.add_separator()
        fm.add_command(label="Exit", command=self._close)
        menu.add_cascade(label="File", menu=fm)
        self.config(menu=menu)

        top = ttk.Frame(self, padding=(6, 6, 6, 2))
        top.pack(fill="x")
        ttk.Button(top, text="Open game folder...", command=self.ask_folder).pack(side="left")
        ttk.Label(top, text="  Language:").pack(side="left")
        self.lang_var = tk.StringVar()
        self.lang_cb = ttk.Combobox(top, textvariable=self.lang_var, state="readonly", width=15)
        self.lang_cb.pack(side="left")
        self.lang_cb.bind("<<ComboboxSelected>>", lambda e: self.rebuild())
        ttk.Label(top, text="  Sort by:").pack(side="left")
        self.group_var = tk.StringVar(value="Category")
        gcb = ttk.Combobox(top, textvariable=self.group_var, state="readonly", width=11,
                           values=["Category", "Soundbank"])
        gcb.pack(side="left")
        gcb.bind("<<ComboboxSelected>>", lambda e: self.rebuild())
        ttk.Label(top, text="  Search:").pack(side="left")
        self.search_var = tk.StringVar()
        se = ttk.Entry(top, textvariable=self.search_var, width=28)
        se.pack(side="left")
        self._search_job = None
        self.search_var.trace_add("write", lambda *a: self._search_later())
        ttk.Button(top, text="Export all listed...", command=lambda: self.export(all_listed=True)).pack(side="right")
        ttk.Button(top, text="Export selected...", command=self.export).pack(side="right", padx=4)
        self.path_lbl = ttk.Label(self, text="", foreground="#666", padding=(8, 0))
        self.path_lbl.pack(fill="x")

        pan = ttk.PanedWindow(self, orient="horizontal")
        pan.pack(fill="both", expand=True, padx=6, pady=4)
        left = ttk.Frame(pan)
        cols = [("dur", "Length", 60, "e"), ("fmt", "Format", 135, "w"),
                ("src", "Stored in", 170, "w"), ("id", "ID", 90, "e")]
        self.tree = ttk.Treeview(left, columns=[c[0] for c in cols], selectmode="extended")
        self.tree.heading("#0", text="Sound")
        self.tree.column("#0", width=300, stretch=True)
        for cid, head, w, anc in cols:
            self.tree.heading(cid, text=head)
            self.tree.column(cid, width=w, anchor=anc, stretch=False)
        ys = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=ys.set)
        self.tree.pack(side="left", fill="both", expand=True)
        ys.pack(side="right", fill="y")
        self.tree.tag_configure("bad", foreground="#a33")
        self.tree.tag_configure("group", foreground="#224")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", lambda e: self.play())
        self.tree.bind("<Return>", lambda e: self.play())
        pan.add(left, weight=3)

        right = ttk.Frame(pan, padding=(8, 0, 0, 0))
        self.title_lbl = ttk.Label(right, text="", font=("TkDefaultFont", 12, "bold"), wraplength=WAVE_W)
        self.title_lbl.pack(anchor="w")
        self.info_lbl = ttk.Label(right, text="", justify="left", wraplength=WAVE_W)
        self.info_lbl.pack(anchor="w", pady=(2, 8))
        self.canvas = tk.Canvas(right, width=WAVE_W, height=WAVE_H, bg="#1e2228",
                                highlightthickness=0, cursor="hand2")
        self.canvas.pack(anchor="w")
        self.canvas.bind("<Button-1>", self._on_wave_click)
        ctl = ttk.Frame(right)
        ctl.pack(anchor="w", fill="x", pady=6)
        self.play_btn = ttk.Button(ctl, text="Play  (Space)", command=self.toggle_play, width=14)
        self.play_btn.pack(side="left")
        self.loop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctl, text="Loop", variable=self.loop_var).pack(side="left", padx=10)
        ttk.Button(ctl, text="Export this sound...", command=self.export_current).pack(side="left", padx=4)
        self.time_lbl = ttk.Label(ctl, text="", font=("TkFixedFont", 10))
        self.time_lbl.pack(side="right")
        ttk.Label(right, text="Click the waveform to play from that point.",
                  foreground="#777").pack(anchor="w")
        pan.add(right, weight=2)
        self.pan = pan
        self.after(200, lambda: pan.sashpos(0, max(600, self.winfo_width() - WAVE_W - 40)))

        sb = ttk.Frame(self, padding=(6, 2))
        sb.pack(fill="x", side="bottom")
        self.status = ttk.Label(sb, text="")
        self.status.pack(side="left", fill="x", expand=True)
        self.prog = ttk.Progressbar(sb, length=220, mode="determinate")
        self.prog.pack(side="right")

        self.bind_all("<space>", self._space)
        self.bind_all("<Control-e>", lambda e: self.export())
        self._draw_empty_wave()

    # ----------------------------------------------------------- folder --
    def ask_folder(self):
        d = filedialog.askdirectory(title="Game folder (or art/WwiseAudio)")
        if d:
            self.open_folder(d)

    def open_folder(self, folder):
        if self.busy:
            messagebox.showinfo(APP_TITLE, "Please wait for the current job to finish.")
            return
        self.player.stop()
        self.status.config(text="Reading SoundbanksInfo.xml and soundbanks...")
        self.prog.configure(mode="indeterminate")
        self.prog.start(12)
        self.busy = True

        def work():
            warns = []
            try:
                ix = ca.WwiseIndex(folder, warn=warns.append)
                self.q.put(("index", ix, warns))
            except Exception as ex:
                self.q.put(("index_err", ex))
        threading.Thread(target=work, daemon=True).start()

    def _index_ready(self, ix, warns):
        self.busy = False
        self.prog.stop()
        self.prog.configure(mode="determinate", value=0)
        self.index = ix
        self.cache.clear()
        self.cache_order.clear()
        self.path_lbl.config(text=ix.dir)
        langs = ix.languages or ["SFX"]
        self.lang_cb.configure(values=langs)
        if self.lang_var.get() not in langs:
            self.lang_var.set(langs[0])
        for w in warns:
            print("warning:", w, file=sys.stderr)
        self.rebuild()
        self._start_durations()

    # ------------------------------------------------------------- tree --
    def _search_later(self):
        if self._search_job:
            self.after_cancel(self._search_job)
        self._search_job = self.after(250, self.rebuild)

    def _visible_sounds(self):
        if not self.index:
            return []
        terms = self.search_var.get().lower().split()
        out = []
        for s in self.index.for_language(self.lang_var.get()):
            if terms:
                hay = f"{s.name} {' '.join(s.category)} {' '.join(s.banks)} {s.id}".lower()
                if not all(t in hay for t in terms):
                    continue
            out.append(s)
        return out

    def rebuild(self):
        self._search_job = None
        sel_sound = self.current
        self.tree.delete(*self.tree.get_children())
        self.leaf.clear()
        sounds = self._visible_sounds()
        by_bank = self.group_var.get() == "Soundbank"
        groups = {}
        for s in sounds:
            paths = [(b,) for b in dict.fromkeys(s.banks)] if by_bank else [s.category]
            if by_bank and not paths:
                paths = [("(no soundbank)",)]
            for p in paths:
                groups.setdefault(p, []).append(s)
        searching = bool(self.search_var.get().strip())
        nodes = {}

        def node(path):
            if path in nodes:
                return nodes[path]
            parent = node(path[:-1]) if len(path) > 1 else ""
            iid = "g\x1f" + "\x1f".join(path)
            self.tree.insert(parent, "end", iid=iid, text=path[-1], tags=("group",),
                             open=searching and len(sounds) < 400)
            nodes[path] = iid
            return iid

        order = {"Dialogue": 0, "Music": 1, "Ambience": 2, "Cutscenes": 3, "Characters": 4,
                 "Enemies": 5, "Weapons & general": 6, "Props & scripted events": 7,
                 "Interface": 8, "Test tones": 9, "Other": 10}
        count = 0
        select_iid = None
        for path in sorted(groups, key=lambda p: (order.get(p[0], 50), [x.lower() for x in p])):
            g = node(path)
            for s in sorted(groups[path], key=lambda s: s.name.lower()):
                iid = f"s\x1f{s.lang}\x1f{s.id}\x1f{g}"
                tags = ("bad",) if s.file is None else ()
                self.tree.insert(g, "end", iid=iid, text=os.path.splitext(s.name)[0],
                                 values=self._row_values(s), tags=tags)
                self.leaf[iid] = s
                count += 1
                if s is sel_sound and select_iid is None:
                    select_iid = iid
        # group sizes
        for path, iid in nodes.items():
            n = self._count_leaves(iid)
            self.tree.item(iid, text=f"{path[-1]}  ({n})")
        if select_iid:
            self.tree.see(select_iid)
            self.tree.selection_set(select_iid)
        lang = self.lang_var.get()
        self.status.config(text=f"{count:,} sounds listed ({lang} + non-localized)"
                           + (f" - {len(self.index.missing):,} listed in the XML but not found"
                              if self.index and self.index.missing else ""))

    def _count_leaves(self, iid):
        n = 0
        for c in self.tree.get_children(iid):
            n += 1 if c in self.leaf else self._count_leaves(c)
        return n

    def _row_values(self, s):
        if s.file is None:
            where = "-"
        elif s.where == "stream":
            where = os.path.basename(s.file)
        else:
            where = f"{os.path.basename(s.file)} ({s.banks[0] if s.banks else '?'})"
        return (fmt_time(s.duration), _fmt_of(s), where, s.id)

    # -------------------------------------------------------- durations --
    def _start_durations(self):
        """Read the header of every sound (fast) to fill in the lengths."""
        ix = self.index
        todo = [s for lst in ix.sounds.values() for s in lst if s.file and s.duration is None]

        def work():
            for i, s in enumerate(todo):
                if self.index is not ix:
                    return
                try:
                    h = s.read_header()
                    s.duration = ca.duration_from_header(h)
                    _FMT[id(s)] = _describe_header(h)
                except OSError:
                    pass
                if i % 200 == 199:
                    self.q.put(("durations", ix))
            self.q.put(("durations", ix))
        threading.Thread(target=work, daemon=True).start()

    def _refresh_rows(self):
        for iid, s in self.leaf.items():
            self.tree.item(iid, values=self._row_values(s))

    # ------------------------------------------------------- selection --
    def _selected_sound(self):
        for iid in self.tree.selection():
            if iid in self.leaf:
                return self.leaf[iid]
        return None

    def _on_select(self, e=None):
        s = self._selected_sound()
        if s is None or s is self.current:
            return
        self.player.stop()
        self.current = s
        self.current_pcm = None
        self.title_lbl.config(text=os.path.splitext(s.name)[0])
        cat = " / ".join(s.category)
        banks = ", ".join(dict.fromkeys(s.banks)) or "-"
        where = (f"{s.file}" if s.where == "stream" else
                 f"inside {os.path.basename(s.file)} (soundbank)") if s.file else "not found"
        extra = "\n(uses the English recording)" if s.standin else ""
        self.info_lbl.config(text=f"Category: {cat}\nSoundbanks: {banks}\nID {s.id}   "
                                  f"{'language ' + s.lang if s.lang != 'SFX' else 'not localized'}"
                                  f"\nFile: {where}{extra}")
        self._draw_empty_wave("Loading..." if s.file else "File not found")
        self.time_lbl.config(text="")
        if s.file:
            self._load(s, autoplay=False)

    def _load(self, s, autoplay):
        key = (s.file, s.offset)
        if key in self.cache:
            self._loaded(s, *self.cache[key], autoplay=autoplay)
            return

        def work():
            try:
                pcm, info = s.load()
                self.q.put(("loaded", s, pcm, info, autoplay))
            except Exception as ex:
                self.q.put(("load_err", s, ex))
        threading.Thread(target=work, daemon=True).start()

    def _loaded(self, s, pcm, info, autoplay=False):
        key = (s.file, s.offset)
        if key not in self.cache:
            self.cache[key] = (pcm, info)
            self.cache_order.append(key)
            while sum(self.cache[k][0].nbytes for k in self.cache_order) > CACHE_BYTES and len(self.cache_order) > 1:
                self.cache.pop(self.cache_order.pop(0), None)
        if s is not self.current:
            return
        self.current_pcm = (pcm, info)
        s.duration = len(pcm) / info["rate"]
        self._draw_wave(pcm)
        self.time_lbl.config(text=f"0:00.0 / {fmt_time(s.duration)}")
        ch = "mono" if info["channels"] == 1 else f"{info['channels']} channels"
        loop = ""
        self.info_lbl.config(text=self.info_lbl.cget("text").split("\nAudio:")[0] +
                             f"\nAudio: {info['rate']:,} Hz {ch}, {fmt_time(s.duration)}{loop}")
        if autoplay:
            self.play()

    # ---------------------------------------------------------- waveform --
    def _draw_empty_wave(self, text=""):
        self.canvas.delete("all")
        if text:
            self.canvas.create_text(WAVE_W // 2, WAVE_H // 2, text=text, fill="#9aa")

    def _draw_wave(self, pcm):
        mono = pcm.astype(np.float32)
        mono = mono[:, 0] if mono.shape[1] == 1 else mono.mean(axis=1)
        n = len(mono)
        img = Image.new("RGB", (WAVE_W, WAVE_H), (30, 34, 40))
        d = ImageDraw.Draw(img)
        mid = WAVE_H // 2
        d.line([(0, mid), (WAVE_W, mid)], fill=(60, 66, 76))
        if n:
            edges = np.linspace(0, n, WAVE_W + 1).astype(int)
            for x in range(WAVE_W):
                seg = mono[edges[x]:max(edges[x + 1], edges[x] + 1)]
                lo, hi = float(seg.min()), float(seg.max())
                d.line([(x, mid - hi / 32768 * (mid - 4)), (x, mid - lo / 32768 * (mid - 4))],
                       fill=(110, 170, 235))
        self.wave_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.wave_img, anchor="nw")
        self.playhead = self.canvas.create_line(0, 0, 0, WAVE_H, fill="#ffcc44", width=2, state="hidden")

    def _on_wave_click(self, e):
        if not self.current_pcm:
            return
        pcm, info = self.current_pcm
        dur = len(pcm) / info["rate"]
        self.play(pos=max(0.0, min(1.0, e.x / WAVE_W)) * dur)

    # ---------------------------------------------------------- playback --
    def _space(self, e):
        if isinstance(e.widget, (tk.Entry, ttk.Entry)):
            return
        self.toggle_play()
        return "break"

    def toggle_play(self):
        if self.player.playing:
            self.player.stop()
        else:
            self.play()

    def play(self, pos=0.0):
        s = self._selected_sound() or self.current
        if s is None:
            return
        if s is not self.current:
            self._on_select()
        if not s.file:
            return
        if not self.current_pcm:
            self._load(s, autoplay=True)
            return
        pcm, info = self.current_pcm
        try:
            self.player.play(pcm, info["rate"], pos, self.loop_var.get())
        except Exception as ex:
            messagebox.showerror(APP_TITLE, f"Could not play the sound:\n{ex}")

    def _tick(self):
        pos = self.player.position()
        self.play_btn.config(text="Stop  (Space)" if pos is not None else "Play  (Space)")
        if self.current_pcm and getattr(self, "playhead", None):
            pcm, info = self.current_pcm
            dur = len(pcm) / info["rate"]
            if pos is None:
                self.canvas.itemconfigure(self.playhead, state="hidden")
            else:
                x = pos / dur * WAVE_W if dur else 0
                self.canvas.coords(self.playhead, x, 0, x, WAVE_H)
                self.canvas.itemconfigure(self.playhead, state="normal")
                self.time_lbl.config(text=f"{fmt_time(pos)} / {fmt_time(dur)}")
        self.after(40, self._tick)

    # ------------------------------------------------------------ export --
    def _sounds_under(self, iids):
        out, seen, used = [], set(), set()

        def walk(iid):
            if iid in self.leaf:
                s = self.leaf[iid]
                key = (s.lang, s.id)
                if key not in seen:
                    seen.add(key)
                    rel = self._export_path(iid, s)
                    if rel.lower() in used:            # different sounds with the same name
                        rel = rel[:-4] + f"_{s.id}.wav"
                    used.add(rel.lower())
                    out.append((s, rel))
            else:
                for c in self.tree.get_children(iid):
                    walk(c)
        for i in iids:
            walk(i)
        return out

    def _export_path(self, iid, s):
        parts = []
        p = self.tree.parent(iid)
        while p:
            parts.insert(0, p.split("\x1f")[-1])
            p = self.tree.parent(p)
        safe = [_safe(x) for x in parts]
        if s.lang != "SFX":
            safe.insert(0, _safe(s.lang))
        return os.path.join(*safe, _safe(os.path.splitext(s.name)[0]) + ".wav")

    def export_current(self):
        if self.current is None:
            return
        s = self.current
        path = filedialog.asksaveasfilename(defaultextension=".wav", initialfile=_safe(os.path.splitext(s.name)[0]) + ".wav",
                                            filetypes=[("WAV audio", "*.wav")])
        if path:
            self._run_export([(s, os.path.basename(path))], os.path.dirname(path))

    def export(self, all_listed=False):
        if self.busy or not self.index:
            return
        iids = self.tree.get_children("") if all_listed else self.tree.selection()
        items = self._sounds_under(iids)
        if not items:
            messagebox.showinfo(APP_TITLE, "Select sounds or folders in the list first." if not all_listed
                                else "Nothing is listed.")
            return
        d = filedialog.askdirectory(title=f"Export {len(items):,} sound(s) to...")
        if d:
            self._run_export(items, d)

    def _run_export(self, items, out_dir):
        self.busy = True
        self.stop_flag.clear()
        self.prog.configure(maximum=len(items), value=0)

        def work():
            fails, done = [], 0
            for i, (s, rel) in enumerate(items):
                if self.stop_flag.is_set():
                    break
                dst = os.path.join(out_dir, rel)
                try:
                    if not s.file:
                        raise ca.AudioError("file not found")
                    pcm, info = s.load()
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    ca.write_wav(dst, pcm, info["rate"], info.get("loops") or ())
                    done += 1
                except Exception as ex:
                    fails.append(f"{s.name}: {ex}")
                if i % 10 == 0:
                    self.q.put(("export_progress", i + 1, len(items)))
            self.q.put(("export_done", done, fails, out_dir))
        threading.Thread(target=work, daemon=True).start()

    # -------------------------------------------------------------- queue --
    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "index":
                    self._index_ready(msg[1], msg[2])
                elif kind == "index_err":
                    self.busy = False
                    self.prog.stop()
                    self.prog.configure(mode="determinate", value=0)
                    self.status.config(text="")
                    messagebox.showerror(APP_TITLE, f"Could not read the game's audio:\n{msg[1]}")
                elif kind == "durations":
                    if msg[1] is self.index:
                        self._refresh_rows()
                elif kind == "loaded":
                    self._loaded(msg[1], msg[2], msg[3], msg[4])
                elif kind == "load_err":
                    if msg[1] is self.current:
                        self._draw_empty_wave(f"Cannot play: {msg[2]}")
                elif kind == "export_progress":
                    self.prog.configure(value=msg[1])
                    self.status.config(text=f"Exporting... {msg[1]:,} / {msg[2]:,}")
                elif kind == "export_done":
                    self.busy = False
                    done, fails, out_dir = msg[1], msg[2], msg[3]
                    self.prog.configure(value=0)
                    self.status.config(text=f"Exported {done:,} sound(s) to {out_dir}"
                                       + (f" - {len(fails)} failed" if fails else ""))
                    if fails:
                        messagebox.showwarning(APP_TITLE, f"{len(fails)} sound(s) could not be exported:\n\n"
                                               + "\n".join(fails[:15]) + ("\n..." if len(fails) > 15 else ""))
        except queue.Empty:
            pass
        self.after(50, self._poll)

    def _close(self):
        self.stop_flag.set()
        self.player.close()
        self.destroy()


_FMT = {}


def _describe_header(h):
    import struct
    vorbis = ca.is_wwise_vorbis(h)
    try:
        if vorbis:
            e = ">" if h[:4] == b"RIFX" else "<"
            nch, rate = struct.unpack(e + "HI", h[22:28])
        else:
            info = ca.parse(h + b"\0" * 64, allow_standard=True)
            nch, rate = info["channels"], info["rate"]
    except Exception:
        return "unknown"
    ch = "mono" if nch == 1 else ("stereo" if nch == 2 else f"{nch} ch")
    khz = rate // 1000 if rate % 1000 == 0 else round(rate / 1000, 1)
    return f"{khz} kHz {ch}" + (" Vorbis" if vorbis else "")


def _fmt_of(s):
    if s.file is None:
        return "missing"
    return _FMT.get(id(s), "")


def _safe(name):
    bad = '<>:"/\\|?*'
    return "".join("_" if c in bad or ord(c) < 32 else c for c in name).strip(" .") or "_"


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    App(folder).mainloop()


if __name__ == "__main__":
    main()
