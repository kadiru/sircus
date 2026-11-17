"""Fabricated post-audit report/JPEG fixtures and mocked codecs only.

These tests never invoke ffmpeg/ffprobe, load a policy or start a simulator.
Fake JPEG/video bodies are deliberately not claims of native or codec validity.
"""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import location_comparison_media as media
from location_comparison_publication import SCHEMA, digest
from location_comparison_server import published_file


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True).encode()
    if path.name.endswith(".gz"):
        with gzip.open(path, "wb") as stream:
            stream.write(raw)
    else:
        path.write_bytes(raw)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(count):
    return dict(requests=count, captures=1, clean=1, rejected=1, misses=count-2)


class EncoderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.native = self.root / "native"; self.native.mkdir()
        self.destination = self.root / "publication"
        self.report_path = self.root / "input-report.json"
        self.audit_sha = "a" * 64
        self.report = dict(schema=SCHEMA, status="completed_audited", synthetic=False, model_promoted=False,
            audit=dict(status="passed", complete=True, synthetic=False, audit_sha256=self.audit_sha,
                session_id="1"*32, manifest_fingerprint="b"*64, comparison_fingerprint="c"*64,
                native_exit_fingerprint="d"*64, audit_exit_fingerprint="e"*64, trace_sha256="f"*64),
            train={m: metrics(8960) for m in media.MODELS[1:]},
            final_tests={m: {"test_broad": metrics(70), "test_local": metrics(25)} for m in media.MODELS},
            costs=dict(scope="Invented source scope; must remain unchanged"), videos=[])
        frame_index = 0
        for clip, (model, phase, before, count) in enumerate(media.SCHEDULE):
            frames = []
            for _ in range(2):
                path = self.native / f"native/frames/{frame_index:05d}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"\xff\xd8SYNTHETIC-" + str(frame_index).encode() + b"\xff\xd9")
                frames.append(dict(frame=frame_index, jpeg_sha256=media._hash(path), resolution=[4, 2]))
                frame_index += 1
            relative = f"comparison-native/trial-{clip:05d}.json.gz"
            packet_sha = write(self.native / relative, dict(schema="sircus-location-comparison-native-trial-v1",
                synthetic=False, filming=True, comparison_fingerprint="c"*64, frames=frames))
            label = f"Invented {model} {phase} clip={clip}; #ordered"
            self.report["videos"].append(dict(id=f"batch-{clip:05d}", model=model, phase=phase,
                train_requests_before=before, active_slots=count, parked_slots=35-count, label=label,
                packet_file=relative, packet_sha256=packet_sha, frame_sequence_sha256=digest(frames), frames=2, fps=30,
                chapters=[dict(start_s=0, label=label, phase=phase, train_requests_before=before)],
                media=None, media_status="verified_frames_not_encoded"))
        self.pin = write(self.report_path, self.report)
        self.calls, self.probe_mutation, self.encode_mutation, self.decode_mutation = [], None, None, None
        self.encode_failure = False

    def fake_process(self, command, **kwargs):
        self.calls.append((command, kwargs))
        self.assertIs(kwargs["shell"], False)
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(kwargs["env"]["NVIDIA_VISIBLE_DEVICES"], "void")
        self.assertGreater(kwargs["timeout"], 0)
        tool = Path(command[0]).name
        if command[1:] == ["-version"]:
            return SimpleNamespace(stdout=(tool+" version 4.4.2-MOCK-ONLY\n").encode(), stderr=b"")
        if tool == "ffprobe":
            movie = Path(command[-1]); clip = int(movie.stem.split("-")[1])
            video = self.report["videos"][clip]
            probe = dict(streams=[dict(codec_type="video", codec_name="h264", pix_fmt="yuv420p",
                avg_frame_rate="30/1", nb_read_frames="2", width=4, height=2, duration=str(2/30)),
                dict(codec_type="data", codec_name="bin_data", codec_tag_string="text")],
                format=dict(duration=str(2/30)),
                chapters=[dict(start_time="0", end_time=str(2/30), tags=dict(title=video["label"]))])
            if self.probe_mutation:
                self.probe_mutation(probe)
            return SimpleNamespace(stdout=json.dumps(probe).encode(), stderr=b"")
        if command[-1] == "-":
            self.assertEqual(command[-3:], ["-f", "null", "-"])
            if self.decode_mutation:
                self.decode_mutation(Path(command[command.index("-i")+1]))
            return SimpleNamespace(stdout=b"", stderr=b"")
        self.assertEqual(tool, "ffmpeg")
        self.assertIn("-n", command); self.assertIn("libx264", command)
        self.assertEqual(command[command.index("-hwaccel")+1], "none")
        self.assertEqual(command[command.index("-threads")+1], "2")
        self.assertEqual(command[command.index("-frames:v")+1], "2")
        self.assertEqual(command[command.index("-vsync")+1], "0")
        self.assertNotIn("-fps_mode", command)
        snapshot = Path(command[command.index("-i")+1])
        movie = Path(command[-1]); clip = int(movie.stem.split("-")[1])
        inputs = [Path(command[i+1]) for i, argument in enumerate(command) if argument == "-i"]
        metadata = inputs[1].read_text()
        self.assertIn("TIMEBASE=1/30\nSTART=0\nEND=2\n", metadata)
        title = next(line[len("title="):] for line in metadata.splitlines() if line.startswith("title="))
        self.assertEqual(re.sub(r"\\(.)", r"\1", title), self.report["videos"][clip]["label"])
        expected = b"".join((self.native / f"native/frames/{index:05d}.jpg").read_bytes()
                            for index in (clip*2, clip*2+1))
        self.assertEqual(snapshot.read_bytes(), expected)
        self.assertFalse(movie.exists())
        movie.write_bytes(b"MOCK-MP4-" + movie.stem.encode())
        if self.encode_failure:
            raise subprocess.CalledProcessError(1, command, stderr=b"invented codec failure")
        if self.encode_mutation:
            self.encode_mutation(snapshot, movie)
        return SimpleNamespace(stdout=b"", stderr=b"")

    def encode(self, **kwargs):
        args = dict(expected_report_sha256=self.pin, expected_audit_sha256=self.audit_sha, execute_approved=True)
        args.update(kwargs)
        with mock.patch.object(media, "_tool", side_effect=lambda value: "/mock/"+Path(value).name), \
             mock.patch.object(media.subprocess, "run", side_effect=self.fake_process):
            return media.encode_publication(self.report_path, self.native, self.destination, **args)

    def save_report(self):
        self.pin = write(self.report_path, self.report)

    def test_exact_order_cpu_encoding_verified_attachment_and_unchanged_sources(self):
        before = {str(p.relative_to(self.native)): media._hash(p) for p in self.native.rglob("*") if p.is_file()}
        original = copy.deepcopy(self.report)
        result = self.encode(ffmpeg="/usr/bin/ffmpeg", ffprobe="/usr/bin/ffprobe")
        self.assertEqual(len(self.calls), 32)  # versions + encode/probe/decode for each of ten clips
        self.assertEqual(result["train"], original["train"])
        self.assertEqual(result["final_tests"], original["final_tests"])
        self.assertEqual(result["costs"], original["costs"])
        self.assertEqual(media._hash(self.report_path), self.pin)
        self.assertEqual(before, {str(p.relative_to(self.native)): media._hash(p) for p in self.native.rglob("*") if p.is_file()})
        self.assertEqual(list((self.destination/"staging").iterdir()), [])
        self.assertFalse((self.destination/"encoding-failure.json").exists())
        self.assertEqual(json.loads((self.destination/"report.json").read_text()), result)
        self.assertEqual(sum(v["phase"] == "train" for v in result["videos"]), 8)
        self.assertTrue(all(v["model"] == "parent" for v in result["videos"] if v["phase"] != "train"))
        for index, video in enumerate(result["videos"]):
            output = video["media"]
            self.assertEqual(video["chapters"], original["videos"][index]["chapters"])
            self.assertTrue(output["verified"])
            self.assertEqual(output["duration_s"], 2/30)
            self.assertIsNotNone(published_file(self.destination, output["url"]))
            receipt = json.loads((self.destination/output["provenance_file"]).read_text())
            self.assertEqual(receipt["frame_sequence_sha256"], video["frame_sequence_sha256"])
            self.assertEqual([f["frame"] for f in receipt["source_frames"]], [index*2,index*2+1])
            self.assertFalse(receipt["encoding"]["gpu"])
            self.assertFalse(receipt["encoding"]["overlays"])
            self.assertTrue(receipt["verification"]["full_decode_passed"])
            self.assertEqual(receipt["verification"]["chapter_data_streams"], 1)
            self.assertNotIn(str(self.root), json.dumps(receipt))

    def test_pending_synthetic_and_wrong_audit_reports_fail_before_any_process(self):
        for change in (dict(status="awaiting_audit"), dict(synthetic=True), dict(model_promoted=True)):
            with self.subTest(change=change):
                old = copy.deepcopy(self.report); self.report.update(change); self.save_report()
                with self.assertRaises(ValueError): self.encode()
                self.assertEqual(self.calls, []); self.assertFalse(self.destination.exists())
                self.report = old; self.save_report()
        with self.assertRaisesRegex(ValueError, "audit pin"):
            self.encode(expected_audit_sha256="0"*64)
        self.assertEqual(self.calls, [])

    def test_report_pin_and_paid_denominators_are_required(self):
        with self.assertRaisesRegex(ValueError, "Changed pinned"):
            self.encode(expected_report_sha256="0"*64)
        self.report["final_tests"]["broader"]["test_broad"]["requests"] = 69
        self.save_report()
        with self.assertRaisesRegex(ValueError, "request totals"):
            self.encode()
        self.assertEqual(self.calls, [])

    def test_all_ten_schedule_identity_and_unattached_media_are_required(self):
        original = copy.deepcopy(self.report)
        for mutation in (lambda r:r["videos"].pop(),
                         lambda r:r["videos"][-1].update(model="broader"),
                         lambda r:r["videos"][0].update(media={"url":"/media/already.mp4"}),
                         lambda r:r["videos"][0].update(frames=0)):
            self.report = copy.deepcopy(original); mutation(self.report); self.save_report()
            with self.assertRaises(ValueError): self.encode()
            self.assertEqual(self.calls, [])

    def test_changed_packet_or_jpeg_fails_before_encoder(self):
        packet = self.native/self.report["videos"][0]["packet_file"]
        raw = packet.read_bytes(); packet.write_bytes(raw+b"changed")
        with self.assertRaisesRegex(ValueError, "pinned media input"): self.encode()
        packet.write_bytes(raw)
        frame = self.native/"native/frames/00000.jpg"
        frame.write_bytes(b"\xff\xd8CHANGED\xff\xd9")
        with self.assertRaisesRegex(ValueError, "JPEG bytes"): self.encode()
        self.assertEqual(self.calls, [])

    def test_resealed_packet_cannot_reorder_frames_or_change_dimensions(self):
        path = self.native/self.report["videos"][0]["packet_file"]
        packet = json.loads(gzip.decompress(path.read_bytes()))
        packet["frames"].reverse()
        self.report["videos"][0]["packet_sha256"] = write(path,packet)
        self.report["videos"][0]["frame_sequence_sha256"] = digest(packet["frames"])
        self.save_report()
        with self.assertRaisesRegex(ValueError, "frame order"):
            self.encode()
        self.assertEqual(self.calls, [])

    def test_invalid_jpeg_framing_fails_even_with_matching_new_pins(self):
        frame = self.native/"native/frames/00000.jpg"; frame.write_bytes(b"not a jpeg")
        path = self.native/self.report["videos"][0]["packet_file"]
        packet = json.loads(gzip.decompress(path.read_bytes()))
        packet["frames"][0]["jpeg_sha256"] = media._hash(frame)
        self.report["videos"][0]["packet_sha256"] = write(path,packet)
        self.report["videos"][0]["frame_sequence_sha256"] = digest(packet["frames"])
        self.save_report()
        with self.assertRaisesRegex(ValueError, "JPEG framing"):
            self.encode()
        self.assertEqual(self.calls, [])

    def test_new_export_only_and_no_second_attachment(self):
        self.encode()
        report_sha = media._hash(self.destination/"report.json")
        calls = len(self.calls)
        with self.assertRaisesRegex(ValueError, "New export"):
            self.encode()
        self.assertEqual(len(self.calls), calls)
        self.assertEqual(media._hash(self.destination/"report.json"), report_sha)
        self.report_path = self.destination/"report.json"; self.pin = report_sha
        self.destination = self.root/"second"
        with self.assertRaisesRegex(ValueError, "already attached"):
            self.encode()
        self.assertFalse(self.destination.exists())

    def test_escaped_linked_and_source_nested_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "New export"):
            media.encode_publication(self.report_path,self.native,self.native/"publication",
                expected_report_sha256=self.pin,expected_audit_sha256=self.audit_sha,execute_approved=True)
        self.report["videos"][0]["packet_file"] = "../escape.json.gz"; self.save_report()
        with self.assertRaisesRegex(ValueError, "frame metadata"):
            self.encode()
        frame = self.native/"native/frames/00000.jpg"
        other = self.root/"original.jpg"; frame.rename(other); frame.symlink_to(other)
        with self.assertRaisesRegex(ValueError, "Linked"):
            media._file(self.native,"native/frames/00000.jpg")
        self.assertEqual(self.calls, [])

    def test_process_failure_keeps_partial_export_and_does_not_attach_report(self):
        self.encode_failure = True
        with self.assertRaises(subprocess.CalledProcessError): self.encode()
        self.assertFalse((self.destination/"report.json").exists())
        self.assertTrue((self.destination/"staging/batch-00000.mjpeg").exists())
        self.assertTrue((self.destination/"media/batch-00000.mp4").exists())
        failure = json.loads((self.destination/"encoding-failure.json").read_text())
        self.assertEqual(failure["status"], "failed_export_preserved_no_resume")
        self.assertFalse(failure["report_attached"])

    def test_probe_wrong_count_duration_size_audio_or_chapter_blocks_attachment(self):
        mutations = [lambda p:p["streams"][0].update(nb_read_frames="1"),
            lambda p:p["format"].update(duration="9"), lambda p:p["streams"][0].update(width=6),
            lambda p:p["streams"].append(dict(codec_type="audio")),
            lambda p:p["chapters"][0]["tags"].update(title="wrong cohort")]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                self.destination = self.root/f"bad-probe-{index}"
                self.probe_mutation = mutation
                with self.assertRaises(ValueError): self.encode()
                self.assertFalse((self.destination/"report.json").exists())

    def test_source_changed_during_encoding_or_output_changed_during_decode_fails(self):
        frame = self.native/"native/frames/00000.jpg"; original = frame.read_bytes()
        self.encode_mutation = lambda snapshot,movie:frame.write_bytes(b"changed mid-encode")
        with self.assertRaisesRegex(ValueError, "JPEG bytes"): self.encode()
        self.assertFalse((self.destination/"report.json").exists())
        frame.write_bytes(original); self.encode_mutation = None
        self.destination = self.root/"decode-change"
        self.decode_mutation = lambda movie:movie.write_bytes(movie.read_bytes()+b"changed")
        with self.assertRaisesRegex(ValueError, "during verification"): self.encode()
        self.assertFalse((self.destination/"report.json").exists())

    def test_opt_in_and_installed_binaries_are_required_without_installation(self):
        with self.assertRaisesRegex(ValueError, "opt-in"):
            self.encode(execute_approved=False)
        with mock.patch.object(media.shutil,"which",return_value=None), mock.patch.object(media.subprocess,"run") as run:
            with self.assertRaisesRegex(ValueError,"not found"):
                media.encode_publication(self.report_path,self.native,self.destination,
                    expected_report_sha256=self.pin,expected_audit_sha256=self.audit_sha,execute_approved=True)
            run.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_no_extra_chapter_track_is_also_valid_but_unknown_data_is_not(self):
        self.probe_mutation = lambda p:p["streams"].pop()
        result = self.encode()
        receipt = json.loads((self.destination/result["videos"][0]["media"]["provenance_file"]).read_text())
        self.assertEqual(receipt["verification"]["chapter_data_streams"], 0)
        self.destination = self.root/"unknown-data-track"
        self.probe_mutation = lambda p:p["streams"][1].update(codec_tag_string="other")
        with self.assertRaisesRegex(ValueError, "video/audio"):
            self.encode()
        self.assertFalse((self.destination/"report.json").exists())

    def test_late_change_to_earlier_source_blocks_final_report(self):
        frame = self.native/"native/frames/00000.jpg"
        def late_change(snapshot, movie):
            if movie.stem == "batch-00009":
                frame.write_bytes(b"changed after earlier clip verification")
        self.encode_mutation = late_change
        with self.assertRaisesRegex(ValueError, "JPEG bytes"):
            self.encode()
        self.assertEqual(len(list((self.destination/"provenance").glob("*.json"))), 10)
        self.assertFalse((self.destination/"report.json").exists())

    def test_timeout_is_bounded_and_keeps_failure_receipt(self):
        def timeout(snapshot, movie):
            raise subprocess.TimeoutExpired("mock ffmpeg", 1)
        self.encode_mutation = timeout
        with self.assertRaises(subprocess.TimeoutExpired):
            self.encode(timeout_s=1)
        failure = json.loads((self.destination/"encoding-failure.json").read_text())
        self.assertEqual(failure["error_type"], "TimeoutExpired")
        self.assertFalse((self.destination/"report.json").exists())


if __name__ == "__main__":
    unittest.main()
