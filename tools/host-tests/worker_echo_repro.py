#!/usr/bin/env python3
# Drive stream-asr-worker over its protocol: feed 12 s of audio, flush, then
# a 0.34 s fragment, flush - reproduces the "tiny epoch echoes the previous
# epoch's text" bug from the 2P run (epoch [73.12,73.456] -> '也会提高一学').
import os
import struct
import subprocess
import sys
import time

MODEL = "models-for-asr-chain/asr"
pcm = open("/tmp/ep12.wav", "rb").read()[44:]  # 16k mono s16

r3, w3 = os.pipe()
# The worker writes responses to fd 3: make fd 3 the WRITE end in the child.
pread = os.dup(r3)          # parent's private read handle
os.dup2(w3, 3)              # fd 3 := write end
p = subprocess.Popen(["./build/stream-asr-worker", "--model-dir", MODEL],
                     stdin=subprocess.PIPE, pass_fds=(3,))
os.close(w3)
r3 = pread


def frame(data: bytes):
    p.stdin.write(struct.pack("<I", len(data)) + data)
    p.stdin.flush()


def read_resp(timeout=15.0):
    """Read one framed response from fd 3: 1B type + 4B len + payload."""
    import fcntl
    fl = fcntl.fcntl(r3, fcntl.F_GETFL)
    fcntl.fcntl(r3, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    buf = b""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            chunk = os.read(r3, 65536)
            if not chunk:
                time.sleep(0.02)
                continue
            buf += chunk
            while len(buf) >= 5:
                typ = buf[0:1]
                (ln,) = struct.unpack("<I", buf[1:5])
                if len(buf) < 5 + ln:
                    break
                payload = buf[5:5 + ln]
                buf = buf[5 + ln:]
                yield typ, payload
        except BlockingIOError:
            time.sleep(0.02)


print("== ready:", next(read_resp(10.0))[1].decode())

# session 1: 12 s in 100 ms frames
step = 3200  # bytes per 100 ms (160 samples * 2)
for off in range(0, len(pcm), step):
    frame(pcm[off:off + step])
    time.sleep(0.1)  # real-time-ish
frame(b"")  # flush

fin = []
for typ, payload in read_resp(30.0):
    if typ == b"T":
        flags, cov = struct.unpack("<II", payload[:8])
        fin.append((flags, cov, payload[8:].decode("utf-8", "replace")))
        print("T flags=%d cov=%.2fs %r" % (flags, cov / 16000.0, payload[8:].decode("utf-8", "replace")[:40]))
        if flags & 2:
            break

# session 2: 0.34 s fragment + flush
tiny = pcm[100 * 16000:100 * 16000 + 10880]  # 0.34 s of audio from 100 s in
for off in range(0, len(tiny), 3200):
    frame(tiny[off:off + 3200])
    time.sleep(0.1)
frame(b"")  # flush

for typ, payload in read_resp(30.0):
    if typ == b"T":
        flags, cov = struct.unpack("<II", payload[:8])
        print("T flags=%d cov=%.2fs %r" % (flags, cov / 16000.0, payload[8:].decode("utf-8", "replace")[:60]))
        if flags & 2:
            break

p.stdin.close()
p.wait(timeout=5)
