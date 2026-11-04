"""Presentation edits preserve native evidence, failed trials and camera identity."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ISAAC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAAC))
import presentation_export as exporter
from board_route import BoardRoute
from entry_quality import EntryQuality
from rl_contract import ShotScore


class PresentationExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.camera = json.loads((ISAAC.parents[1]/"baselines/isaac-expanded-farm-20261102/selected-camera.json").read_text())

    def chapters(self, count=30, frames=76):
        return [dict(cycle=i, source_start_frame=i*200,
                     source_end_frame_inclusive=i*200+frames-1, source_frames=frames,
                     board_rebound=i != 0, route_success=i % 3 == 0,
                     terminal_reason="endpoint_rejected" if i == 0 else "floor_contact")
                for i in range(count)]

    def test_native_json_stage_preserves_exact_frozen_renderer_contract(self):
        source = json.loads((ISAAC.parents[1]/"baselines/isaac-expanded-farm-20261102/result.json").read_text())["stage"]
        expected = exporter.verify_stage(source, copy.deepcopy(source))
        self.assertEqual(expected, source)
        self.assertEqual(type(expected["floor_linear_rgb"]), list)

    def test_actual_or_original_stage_changes_are_rejected_without_tolerance(self):
        source = json.loads((ISAAC.parents[1]/"baselines/isaac-expanded-farm-20261102/result.json").read_text())["stage"]
        for mutate in (
            lambda s: s["floor_linear_rgb"].__setitem__(0, 0.2200001),
            lambda s: s["table_linear_rgb"].__setitem__(1, 0.4400001),
            lambda s: s["lighting"]["key"].__setitem__("intensity", 321),
            lambda s: s.__setitem__("unexpected_renderer_setting", True),
        ):
            changed = copy.deepcopy(source)
            mutate(changed)
            for actual,frozen in ((changed,source),(source,changed)):
                with self.assertRaisesRegex(ValueError,"Silver-gray lighting or floor changed"):
                    exporter.verify_stage(actual,frozen)

    def test_first_complete_prefix_retains_failures_and_ignores_capture_labels(self):
        chapters = self.chapters()
        selected = exporter.choose_prefix(chapters)
        self.assertEqual([c["cycle"] for c in selected], list(range(24)))
        self.assertEqual(selected[0]["terminal_reason"], "endpoint_rejected")
        changed = copy.deepcopy(chapters)
        for chapter in changed:
            chapter["route_success"] = not chapter["route_success"]
        self.assertEqual([c["cycle"] for c in exporter.choose_prefix(changed)], list(range(24)))

    def test_short_recordings_and_excessive_speed_are_rejected_without_padding(self):
        with self.assertRaisesRegex(ValueError, "Not enough"):
            exporter.choose_prefix(self.chapters(20))
        with self.assertRaisesRegex(ValueError, "excessive retiming"):
            exporter.choose_prefix(self.chapters(30, 201))
        no_rebounds = self.chapters()
        for chapter in no_rebounds:
            chapter["board_rebound"] = False
        with self.assertRaisesRegex(ValueError, "Not enough"):
            exporter.choose_prefix(no_rebounds)

    def test_frame_map_is_monotonic_unique_and_keeps_each_whole_chapter(self):
        selected = exporter.choose_prefix(self.chapters())
        mapping, chapters, speed = exporter.source_frame_map(selected)
        self.assertEqual(len(mapping), 1800)
        self.assertEqual(len({m["source_frame"] for m in mapping}), 1800)
        self.assertEqual(mapping[0]["source_frame"], 0)
        self.assertEqual(mapping[-1]["source_frame"], selected[-1]["source_end_frame_inclusive"])
        self.assertTrue(all(b["source_frame"] > a["source_frame"] for a,b in zip(mapping,mapping[1:])))
        self.assertEqual({m["cycle"] for m in mapping}, set(range(24)))
        self.assertAlmostEqual(speed, 1824/1800)
        self.assertEqual(chapters[0]["start_frame"], 0)
        self.assertEqual(sum(c["frames"] for c in chapters), 1800)
        self.assertTrue(all(b["start_frame"] == a["start_frame"]+a["frames"] for a,b in zip(chapters,chapters[1:])))

    def test_duplicate_raw_intervals_or_out_of_order_trials_are_rejected(self):
        chapters = self.chapters()
        chapters[1]["cycle"] = 4
        with self.assertRaisesRegex(ValueError, "chronological"):
            exporter.choose_prefix(chapters)
        chapters = self.chapters()
        chapters[1].update(source_start_frame=0, source_end_frame_inclusive=75)
        with self.assertRaisesRegex(ValueError, "looped"):
            exporter.source_frame_map(chapters)

    def frames(self):
        frames = []
        for i in range(39):
            frame = dict(frame=i, phase="move" if i < 36 else "release",
                         physics_step=-36+i if i < 36 else (i-36)*8,
                         eye_m=list(self.camera["eye_m"]), target_m=list(self.camera["target_m"]),
                         all_ball_world_m=np.zeros((35,3)).tolist(), focus_ball_world_m=[0,0,0])
            if i < 36:
                frame["focus_joints_rad"] = {name: [i/100]*6 for name in ("Arm10","Arm14")}
            frames.append(frame)
        return dict(frames=frames, physics_dt_s=1/240, release_stride=8, release_step=0)

    def test_fixed_camera_and_actual_driven_joint_states_are_required(self):
        clip = self.frames()
        _, _, motion = exporter.check_frames(clip, self.camera, 0)
        self.assertGreater(motion["Arm10"], 0.1)
        clip["frames"][38]["eye_m"][0] += 0.1
        with self.assertRaisesRegex(ValueError, "camera moved"):
            exporter.check_frames(clip, self.camera, 0)
        clip = self.frames()
        clip["frames"][0]["focus_joints_rad"]["Arm14"] = [0]*5
        with self.assertRaises(ValueError):
            exporter.check_frames(clip, self.camera, 0)

    def test_camera_frame_identity_and_native_cadence_cannot_be_changed(self):
        clip = self.frames()
        clip["frames"][37]["physics_step"] += 1
        with self.assertRaisesRegex(ValueError, "cadence"):
            exporter.check_frames(clip, self.camera, 0)
        clip = self.frames()
        clip["frames"][12]["focus_ball_world_m"] = [1,2,3]
        with self.assertRaisesRegex(ValueError, "identity changed"):
            exporter.check_frames(clip, self.camera, 0)

    def physics_fixture(self):
        """Synthetic two-step misses exercise the real scalar scoring verifier."""
        scene = json.loads((ISAAC/"scene.json").read_text())
        origins = np.zeros((35,3))
        origins[:,0] = np.arange(35)*3
        cells = [dict(id=f"test-{i}") for i in range(35)]
        positions = np.repeat(origins[None], 2, axis=0)
        positions[0,:,2] = 0.3
        positions[1,:,2] = -0.3
        data = dict(time_step=np.array([1,2]), position_m=positions,
                    velocity_m_s=np.zeros((2,35,3)), forces_N=np.zeros((2,35,6)),
                    joint_position_rad=np.zeros((2,35,2,6)),
                    joint_velocity_rad_s=np.zeros((2,35,2,6)),
                    board_position_m=np.zeros((2,35,3)),
                    board_quaternion_wxyz=np.tile([1,0,0,0], (2,35,1)))
        path = self.root/"physics-00.npz"
        np.savez_compressed(path, **data)
        telemetry = [dict(physics_step=i+1, ball_world_m=positions[i,1].tolist(),
                          ball_velocity_m_s=[0,0,0], forces_N=dict.fromkeys(exporter.FORCES,0.),
                          activity={name: [] for name in exporter.FORCES}) for i in range(2)]
        clip = dict(physics_file=path.name, physics_sha256=exporter.sha(path),
                    physics_cell_ids=[c["id"] for c in cells],
                    physics_columns=dict(cell_ids=[c["id"] for c in cells],
                        force_names=list(exporter.FORCES), arm_names=["Arm10","Arm14"],
                        coordinate_frame="world", position_units="metres"),
                    release_step=0, capture_end_step=2, focus_telemetry=telemetry,
                    frames=[dict(phase="release", physics_step=2, all_ball_world_m=positions[1].tolist(),
                        all_joints_rad={name: np.zeros((35,6)).tolist() for name in ("Arm10","Arm14")})])
        scorer = ShotScore(scene["bucket"],scene["ball"]["radius_m"])
        quality = EntryQuality(scene["bucket"],scene["ball"]["radius_m"])
        route = BoardRoute(1/240)
        for i in range(2):
            scorer.step(positions[i,0]-origins[0],0,1/240,0)
            quality.step(positions[i,0]-origins[0],0,scorer.entered)
            route.step(dict.fromkeys(exporter.FORCES,0.))
        scored = scorer.result()
        measured_quality = quality.result(scored["success"])
        row = dict(invalid=None, bucket_xy_m=scene["bucket"]["center_xy_m"],
                   settled_board_normal=[0,0,1],
                   settled=dict(position_error_mm=0, normal_error_deg=0, joint_speed_rad_s=0,
                       position_drift_mm=0, normal_drift_deg=0, dropper_position_error_mm=0),
                   **scored, **measured_quality, **route.result(scored["success"],measured_quality["quality_score"]))
        rows = [copy.deepcopy(row) for _ in range(35)]
        return data, clip, rows, scene, origins, cells

    def test_all35_native_records_are_independently_rescored(self):
        _, clip, rows, scene, origins, cells = self.physics_fixture()
        summary, audits = exporter.verify_physics(self.root,clip,rows,scene,origins,cells)
        self.assertEqual(summary["file"], "physics-00.npz")
        self.assertEqual(summary["sha256"], exporter.sha(self.root/summary["file"]))
        self.assertEqual(summary["rescored_worlds"],35)
        self.assertEqual(len(audits),35)
        self.assertTrue(all(a["scoring_stop_step"] == 2 for a in audits))
        self.assertTrue(all(not a["physical_retained_capture"] for a in audits))

    def test_native_packet_requires_unchanged_bytes_and_cell_order(self):
        _, clip, rows, scene, origins, cells = self.physics_fixture()
        with (self.root/clip["physics_file"]).open("ab") as handle:
            handle.write(b"altered")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            exporter.verify_physics(self.root,clip,rows,scene,origins,cells)
        _, clip, rows, scene, origins, cells = self.physics_fixture()
        clip["physics_cell_ids"][0] = "wrong-world"
        with self.assertRaisesRegex(ValueError, "ordering changed"):
            exporter.verify_physics(self.root,clip,rows,scene,origins,cells)

    def test_render_readback_background_contact_and_clock_changes_are_detected(self):
        for change, pattern in (
            (lambda c: c["frames"][0]["all_joints_rad"]["Arm14"][4].__setitem__(0,1), "articulation readback"),
            (lambda c: c["frames"][0]["all_ball_world_m"][4].__setitem__(0,1), "ball readback"),
            (lambda c: c["focus_telemetry"][0]["activity"]["board"].append(3), "Background activity"),
            (lambda c: c["focus_telemetry"][0].__setitem__("physics_step",2), "JSON observations"),
        ):
            with self.subTest(pattern=pattern):
                _, clip, rows, scene, origins, cells = self.physics_fixture()
                change(clip)
                with self.assertRaisesRegex(ValueError,pattern):
                    exporter.verify_physics(self.root,clip,rows,scene,origins,cells)

    def test_claimed_background_capture_cannot_override_measured_miss(self):
        _, clip, rows, scene, origins, cells = self.physics_fixture()
        rows[20]["success"] = True
        with self.assertRaisesRegex(ValueError,"scoring replay disagrees"):
            exporter.verify_physics(self.root,clip,rows,scene,origins,cells)

    def test_encoder_receives_only_unique_original_files_and_caption_is_explicit(self):
        frames = self.root/"frames"
        frames.mkdir()
        mapping, chapters, _ = exporter.source_frame_map(exporter.choose_prefix(self.chapters()))
        for item in mapping:
            (frames/f"{item['source_frame']:05d}.jpg").write_bytes(f"original-{item['source_frame']}".encode())
        summary = dict(frame_map=mapping,chapters=chapters,resolution=[1920,1080])
        def encode(folder,output,actual_chapters,resolution,caption=True):
            originals = [p.resolve() for p in sorted(folder.glob("*.jpg"))]
            self.assertEqual(len(originals),1800)
            self.assertEqual(len(set(originals)),1800)
            self.assertEqual(originals[0],(frames/"00000.jpg").resolve())
            self.assertFalse(caption)
            self.assertEqual(actual_chapters,chapters)
            return dict(frames=1800,duration_s=60)
        with patch.object(exporter,"encode_presentation_movie",side_effect=encode):
            exporter.render_movie(self.root,self.root/"film.mp4",summary,caption=False)
        self.assertEqual(mapping[0]["source_frame_sha256"],exporter.sha(frames/"00000.jpg"))

    def encoded_probe(self, chapters):
        return dict(streams=[dict(codec_type="video", nb_read_frames="1800", avg_frame_rate="30/1",
                    codec_name="h264", pix_fmt="yuv420p", width=1920, height=1080, duration="60.000000")],
                    format=dict(duration="60.000000"), chapters=[dict(
                        start_time=str(c["start_frame"]/30), end_time=str((c["start_frame"]+c["frames"])/30),
                        tags=dict(title=c["title"])) for c in chapters])

    def test_caption_encode_checks_chapters_frame_count_and_full_decode(self):
        _, chapters, _ = exporter.source_frame_map(exporter.choose_prefix(self.chapters()))
        movie = self.root/"output.mp4"
        calls, captions = [], []
        def command(args, **kwargs):
            calls.append(args)
            if "libx264" in args:
                movie.write_bytes(b"encoder-result-fixture")
                filters = args[args.index("-vf")+1]
                self.assertIn("fontsize=22",filters)
                self.assertIn("DejaVuSans.ttf",filters)
                for token in filters.split("textfile='")[1:]:
                    captions.append(Path(token.split("'")[0]).read_text())
        with patch.object(exporter,"caption_font",return_value=Path("/installed/DejaVuSans.ttf")), \
             patch.object(exporter.subprocess,"run",side_effect=command), \
             patch.object(exporter.subprocess,"check_output",return_value=json.dumps(self.encoded_probe(chapters)).encode()):
            result = exporter.encode_presentation_movie(self.root,movie,chapters,[1920,1080])
        self.assertEqual(captions,["SIRCUS · Native Isaac Sim","Saved configuration trials · action edit"])
        self.assertEqual(result["non_simulation_content"],["text_caption"])
        self.assertTrue(result["decode_verified"])
        self.assertEqual(len(calls),2)
        self.assertIn("-xerror",calls[1])
        self.assertEqual(result["sha256"],exporter.sha(movie))

    def test_encoder_rejects_missing_frames_reported_by_actual_decoder(self):
        _, chapters, _ = exporter.source_frame_map(exporter.choose_prefix(self.chapters()))
        probe = self.encoded_probe(chapters)
        probe["streams"][0]["nb_read_frames"] = "1799"
        with patch.object(exporter.subprocess,"run"), \
             patch.object(exporter.subprocess,"check_output",return_value=json.dumps(probe).encode()):
            with self.assertRaisesRegex(ValueError,"differs from the verified frame map"):
                exporter.encode_presentation_movie(self.root,self.root/"short.mp4",chapters,[1920,1080],caption=False)


if __name__ == "__main__":
    unittest.main()
