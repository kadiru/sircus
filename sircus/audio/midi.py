"""Optional local CoreMIDI output. Initialization is operator-side, never control-side."""
import json
import os
import selectors
import subprocess


class MidiSink:
    def __init__(self, helper):
        self.process = subprocess.Popen([str(helper)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                if not selector.select(3):
                    raise RuntimeError("CoreMIDI helper did not become ready in 3 seconds")
                # Helper emits one short line atomically, before reading stdin.
                reply = os.read(self.process.stdout.fileno(), 1024)
                if b'"ready":true' not in reply:
                    raise RuntimeError("CoreMIDI unavailable; no audio output")
            os.set_blocking(self.process.stdin.fileno(), False)
        except Exception:
            self.close()
            raise

    def send(self, payload):
        if self.process.poll() is not None:
            raise RuntimeError("CoreMIDI helper exited")
        raw = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
        if os.write(self.process.stdin.fileno(), raw) != len(raw):
            raise RuntimeError("short MIDI pipe write")

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()  # EOF makes helper panic before disposal.
        try:
            self.process.wait(timeout=.3)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=.3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=.3)
        if self.process.stdout:
            self.process.stdout.close()
