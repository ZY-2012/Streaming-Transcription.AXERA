#!/usr/bin/env python3
"""Live diarization+ASR demo viewer (diart-style), stdlib only.

Runs audio-pipeline in file-realtime mode (simulated microphone), tails the
pipeline's events.jsonl and pushes it to the browser over SSE.  The page plays
the same WAV in sync with the pipeline's stream start, so the gap between the
"now" cursor and the labels is the real end-to-end latency.

NOTE: no authentication -- anyone who can reach the port can start/stop runs.
Intended for a trusted lab LAN only.
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(REPO, "config/meeting-transcribe.yaml")
# Run workspace: intermediates (segments, events, lock, config) land here.
WORK = os.path.join(REPO, "runs")
EVENTS = os.path.join(WORK, "events.jsonl")
LOG = os.path.join(WORK, "run.log")
INPUTS = {
    "60": os.path.join(REPO, "samples/2P_meeting_60s.wav"),
    # optional longer meeting; drop any 16k mono WAV here to enable "full"
    "full": os.path.join(REPO, "samples/2P_meeting_full.wav"),
}
# Browser-uploaded custom audio lands here (16 kHz mono WAV, transcoded in
# the browser so the board only ever receives clean PCM WAV).
UPLOAD_DIR = os.path.join(REPO, "uploads")
UPLOAD_LIMIT = 300 * 1024 * 1024
REF_RTTM = os.path.join(REPO, "samples/reference.rttm")
STEPS = ["10", "2", "1"]
MAXSEGS = ["10", "5"]
PSPLITS = ["0", "4", "3"]
PREVIEWS = ["0", "3", "2", "1"]
TRAILS = ["1", "0.5"]
TICKS = ["0", "0.8", "0.5", "1.2"]

state = {"proc": None, "input": None, "launched": 0.0}
lock = threading.Lock()


def inputs():
    """Built-in samples plus every uploaded WAV ("up:<filename>" keys)."""
    d = dict(INPUTS)
    try:
        for name in sorted(os.listdir(UPLOAD_DIR)):
            if name.lower().endswith(".wav"):
                d["up:" + name] = os.path.join(UPLOAD_DIR, name)
    except OSError:
        pass
    return d


def make_config(wav, step, maxseg, psplit, preview, trail, tick):
    text = open(TEMPLATE, encoding="utf-8").read()
    # The template stores all run products under the relative "output/" tree;
    # redirect every run into the dedicated WORK directory.
    text = text.replace("output/", WORK + "/")
    text = re.sub(r"input_wav: .*", "input_wav: %s" % wav, text)
    text = re.sub(r"session_step_sec: .*", "session_step_sec: %s" % step, text)
    text = re.sub(r"max_segment_seconds: .*", "max_segment_seconds: %s" % maxseg, text)
    text = re.sub(r"pause_split_seconds: .*", "pause_split_seconds: %s" % psplit, text)
    text = re.sub(r"preview_seconds: .*", "preview_seconds: %s" % preview, text)
    text = re.sub(r"trailing_silence_seconds: .*", "trailing_silence_seconds: %s" % trail, text)
    # Streaming slicer: fixed-cadence text (stext/sfinal events).  When on,
    # force the old pause-gated preview path off (duplicate fast text).
    text = re.sub(r"stream_tick_seconds: .*", "stream_tick_seconds: %s" % tick, text)
    if tick != "0":
        text = re.sub(r"preview_seconds: .*", "preview_seconds: 0", text)
    path = os.path.join(WORK, "config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def env():
    e = dict(os.environ)
    libs = ["build", "driver/apm/lib", "driver/vad/lib",
            "driver/asr/sensevoice/lib"]
    e["LD_LIBRARY_PATH"] = ":".join([os.path.join(REPO, p) for p in libs] +
                                    ["/soc/lib", "/opt/lib"])
    e["AX_SENSEVOICE_SHARE_DIR"] = os.path.join(REPO, "driver/asr/sensevoice")
    e["OMP_NUM_THREADS"] = "1"
    return e


def running():
    p = state["proc"]
    return p is not None and p.poll() is None


def events_ended():
    """True once the pipeline has emitted its 'end' event (writing to EVENTS)."""
    try:
        with open(EVENTS, "rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(max(0, n - 8192))
            data = f.read()
            return b'"ev":"end"' in data or b'"ev": "end"' in data
    except OSError:
        return False


def _stop_locked():
    p = state["proc"]
    if p is not None and p.poll() is None:
        os.killpg(p.pid, signal.SIGINT)
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=5)
    return True


def _watchdog(proc):
    """SIGINT the pipeline ~10s after its 'end' event so the board frees the NPU
    promptly instead of lingering in diar session wrap-up (~3 min)."""
    ended_at = None
    while proc.poll() is None:
        time.sleep(1)
        if ended_at is None and events_ended():
            ended_at = time.time()
        if ended_at is not None and time.time() - ended_at > 10:
            with lock:
                if state["proc"] is proc:  # do not touch a newer run
                    _stop_locked()
            return


def start(key, step, maxseg, psplit, preview, trail, tick):
    with lock:
        if running():
            # Clicking 开始 is a clear intent to replace the current run:
            # stop it (a few seconds of final-window flush) and start fresh.
            _stop_locked()
        wav = inputs().get(key)
        if not wav or not os.path.isfile(wav):
            return False, "missing input %s" % wav
        if step not in STEPS or maxseg not in MAXSEGS or psplit not in PSPLITS or preview not in PREVIEWS or trail not in TRAILS:
            return False, "bad step/maxseg/psplit/preview/trail"
        if tick not in TICKS:
            return False, "bad tick"
        # Rotate the previous workspace out of the request path: renaming is a
        # single fast op even on NFS, the actual deletion happens in background
        # (rm -rf of ~1k files over NFS would stall /start for seconds).
        if os.path.isdir(WORK):
            old = "%s.old.%d" % (WORK, int(time.time() * 1000))
            try:
                os.rename(WORK, old)
                threading.Thread(target=shutil.rmtree, args=(old, True),
                                 daemon=True).start()
            except OSError:
                subprocess.run(["rm", "-rf", WORK])
        os.makedirs(os.path.join(WORK, "mp3"), exist_ok=True)
        cfg = make_config(wav, step, maxseg, psplit, preview, trail, tick)
        log = open(LOG, "w")
        state["log"] = log
        state["proc"] = subprocess.Popen(
            [os.path.join(REPO, "build/audio-pipeline"), "--config", cfg],
            cwd=REPO, env=env(), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        threading.Thread(target=_watchdog, args=(state["proc"],),
                         daemon=True).start()
        state["input"] = key
        state["launched"] = time.time()
        return True, "started"


def stop():
    with lock:
        _stop_locked()
        return True, "stopped"


def load_ref(limit):
    out = []
    if not os.path.isfile(REF_RTTM):
        return out
    for line in open(REF_RTTM, encoding="utf-8"):
        f = line.split()
        if len(f) >= 8 and f[0] == "SPEAKER":
            b, d = float(f[3]), float(f[4])
            if b < limit:
                out.append([b, b + d, f[7]])
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            data = open(os.path.join(HERE, "index.html"), "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif u.path == "/audio":
            self.serve_file(inputs().get(q.get("input", ["60"])[0]), "audio/wav")
        elif u.path == "/inputs":
            items = []
            for key, path in inputs().items():
                label = os.path.basename(path)
                if label.lower().endswith(".wav"):
                    label = label[:-4]
                items.append({"key": key, "label": label})
            self.send_json(items)
        elif u.path == "/ref":
            key = q.get("input", ["60"])[0]
            if key.startswith("up:"):
                self.send_json([])  # no reference labels for custom audio
            else:
                self.send_json(load_ref(float(q.get("limit", ["1e9"])[0])))
        elif u.path == "/status":
            self.send_json({"running": running(), "input": state["input"]})
        elif u.path == "/events":
            self.stream_events()
        else:
            self.send_error(404)

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/start":
            ok, msg = start(q.get("input", ["60"])[0], q.get("step", ["1"])[0],
                            q.get("maxseg", ["5"])[0], q.get("psplit", ["4"])[0],
                            q.get("preview", ["2"])[0], q.get("trail", ["0.5"])[0],
                            q.get("tick", ["0"])[0])
            self.send_json({"ok": ok, "msg": msg}, 200 if ok else 409)
        elif u.path == "/stop":
            ok, msg = stop()
            self.send_json({"ok": ok, "msg": msg})
        elif u.path == "/upload":
            self.handle_upload(q)
        else:
            self.send_error(404)

    def handle_upload(self, q):
        name = (q.get("name", ["audio.wav"])[0]
                .replace("\\", "/").split("/")[-1])
        name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .") or "audio.wav"
        if not name.lower().endswith(".wav"):
            name += ".wav"
        if len(name) > 100:
            name = name[-100:]
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = 0
        if size <= 44 or size > UPLOAD_LIMIT:
            self.send_json({"ok": False, "msg": "文件大小超出限制（1MB–300MB）"}, 400)
            self.rfile.read(min(size, 1 << 20))
            return
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        tmp = os.path.join(UPLOAD_DIR, ".%d.tmp" % time.time())
        try:
            with open(tmp, "wb") as f:
                left = size
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            head = open(tmp, "rb").read(12)
            if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                raise ValueError("not a WAV file")
            final = os.path.join(UPLOAD_DIR, name)
            os.replace(tmp, final)
        except Exception as e:  # noqa: BLE001 - report any failure to the page
            try:
                os.unlink(tmp)
            except OSError:
                pass
            self.send_json({"ok": False, "msg": "上传失败：%s" % e}, 400)
            return
        label = name[:-4] if name.lower().endswith(".wav") else name
        self.send_json({"ok": True, "key": "up:" + name, "label": label})

    def serve_file(self, path, ctype):
        if not path or not os.path.isfile(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        begin, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
        if m:
            if m.group(1):
                begin = int(m.group(1))
            if m.group(2):
                end = min(int(m.group(2)), size - 1)
            self.send_response(206)
            self.send_header("Content-Range", "bytes %d-%d/%d" % (begin, end, size))
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - begin + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(begin)
            left = end - begin + 1
            try:
                while left > 0:
                    chunk = f.read(min(65536, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def stream_events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        launched = state["launched"]
        # Tail EVENTS through one stable fd: even if a later /start rotates the
        # workspace (rename + background rmtree), we keep reading this inode.
        # "a+b": creates the file if the pipeline has not flushed its first
        # line yet; seek(0) replays the whole run for late-joining viewers.
        evf = open(EVENTS, "a+b")
        evf.seek(0)
        buf = b""
        try:
            while True:
                if state["launched"] != launched:
                    return  # new run: client reconnects
                chunk = evf.read(65536)
                if chunk:
                    parts = (buf + chunk).split(b"\n")
                    buf = parts.pop()  # keep the trailing partial line
                    for line in parts:
                        if line:
                            self.wfile.write(b"data: " + line + b"\n\n")
                status = {"ev": "status", "running": running(),
                          "since": round(time.time() - launched, 1) if launched else 0}
                self.wfile.write(("data: %s\n\n" % json.dumps(status)).encode())
                self.wfile.flush()
                time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    print("live demo on http://0.0.0.0:%d  (no auth, LAN only)" % port, flush=True)
    try:
        srv.serve_forever()
    finally:
        stop()


if __name__ == "__main__":
    main()
