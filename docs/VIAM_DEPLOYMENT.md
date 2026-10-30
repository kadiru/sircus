# Viam deployment preparation

This package prepares the interface and commissioning boundary for the next physical session. **It cannot command hardware.** The default coordinator is blocked; `dry_run` is a deterministic contract rehearsal using an in-memory fake adapter. Its fixture coordinates are neither observed geometry nor trained predictions. No Viam machine configuration has been changed by this implementation.

## What exists

- `sircus/deployment/contracts.py`: typed poses, RGB-D observations with bracketing joint states, bucket estimates, verified transforms, board kinematic estimates, release plans and adapter/detector protocols.
- `controller.py`: operator state machine, fresh-scene checks, plan compatibility, board lock, recording/grasp/release interlocks, explicit ownership and cancellation.
- `adapters.py`: `UncommissionedAdapter` methods fail closed; `UnknownBucketDetector` returns unknown. `MockAdapter` and `MockBucketDetector` are synthetic fixtures, not a physical Viam adapter or numerical simulator.
- `viam_module.py`: runnable SDK module exposing a generic coordinator and vision stub. Construction performs no robot connection, resource lookup, image capture or actuation. Vision properties declare every unsupported capability false; detection calls raise `NotImplementedError` instead of fabricating detections.
- `viam-fragment.template.json`: local-module staging template, default blocked, without credentials or physical dependencies. It is not an installed module or a deployment.
- `commissioning.template.json`: unresolved measurements, resources and historical evidence. Editing flags does not enable hardware.

The existing physical dashboard/controller remains unchanged. Its adapter only owns dropper motion; this new two-arm execution contract does **not** silently extend those permissions. A reviewed physical adapter, test evidence and explicit deployment authorization are still required.

## Run without hardware or cloud

From the repository root, use the pinned environment already installed for Sircus, or create an isolated Python environment and install `sircus/deployment/requirements.txt` (SDK0.85.1):

```sh
python -m sircus.deployment                 # blocked capability/status JSON
python -m sircus.deployment --rehearse      # synthetic complete operator sequence
python -m unittest discover -s sircus/deployment/tests -v
```

The rehearsal has no credentials, sockets, physical calls, numerical bounce model or automatic capture outcome. Its final outcome is `unknown` even after a synthetic release. It does not affect the simulation service or original hardware dashboard.

For a separately authorized installation, place the checkout at `/opt/sircus`, install the pinned environment at `/opt/sircus/.venv`, and configure `run-module.sh` as the local module executable. It forwards Viam's socket arguments to `python -m sircus.deployment.viam_module`. `SIRCUS_MODULE_PYTHON` can override the interpreter path. **Do not attach the template to a running machine until the executable is installed and the blocked module is tested.** No existing service was restarted. A private, unattached cloud fragment was staged separately as documented below; it has zero deployments.

Models/resources:

| API | Model | Proposed resource |
| --- | --- | --- |
| `rdk:service:generic` | `sircus:deployment:operation-coordinator` | `sircus-operation-coordinator` |
| `rdk:service:vision` | `sircus:deployment:bucket-vision-stub` | `sircus-bucket-vision` |

`DoCommand({"action":"status"})` or `capabilities` returns `schema: sircus.deployment.v1`, `mode`, immutable `hardware_enabled:false`, `phase`, `busy`, `revision`, `available_actions`, `blockers`, `board`, `bucket`, `observation`, `plan`, `trial`, and provenance-bearing events. Existing Viam authentication applies when a module is installed; there is no unauthenticated HTTP listener. The vision stub returns `status:unknown`, `estimate:null`, `camera_read:false`.

## Operator workflow contract

1. `position_board` (dry-run only, world `xyz_mm` and Viam `orientation_vector_deg`) then `confirm_board`. A future adapter must obtain arm14 joints and compose verified base/FK/tool/suction/board transforms. Board vision is not required.
2. Operator moves the bucket; `confirm_bucket_placed` increments the scene revision.
3. `scan_bucket`: move arm10 to a separately validated scan pose, establish stationary readbacks, capture RGB-D with joint states bracketing exposure, obtain a camera-frame bucket estimate, then apply the matched calibrated camera-to-world transform.
4. `review_bucket` with `confirmed:true`: operator reviews the estimate. Plan must match the bucket observation, board joints and revision; collision, limits and native trajectory checks are distinct requirements.
5. `go_loading`: move to the validated loading pose, check settled, then open the empty dropper gripper. Operator inserts the ball and clears the workspace.
6. `confirm_loaded` with `confirmed:true`, then explicit `grip_drop`: recording ready → close gripper and confirm grasp → recheck fixed board → validated release motion → recheck board/stationarity/recording → capture measured release pose → open dropper gripper. The board does not move during the shot.
7. `stop` cancels pending work and stops only operation IDs owned by this coordinator on either arm. It never opens the gripper or releases suction. No vacuum actuation method is exposed. `reset` clears evidence and requires a fresh board/bucket sequence.

Unknown grasp, bucket estimate, scale, calibration, moving arms, changed board (>0.5°), stale scan (>120s in the rehearsal), incompatible plan, recording failure or unsettled release blocks progress. A configured tolerance is not a verified physical safety limit. Trial/events preserve UUID, revision, observation/plan IDs, timestamps, requested and measured release poses, evidence provenance and unknown outcome. Current mock records are in-memory; durable physical data integration is outstanding.

## Physical integration points to reuse

The future commissioned adapter should reuse `sircus/physical_trials/viam_adapter.py` connection scoping, `Motion.Move` planning, snapshot/camera methods, local recording, `capture-control` discovery and Sequence/data upload pathways. It must adapt those into the new typed contracts, add separately reviewed board-arm ownership and stop support, and retain existing credential redaction. Do not wrap legacy `stop()` and claim it stops arm14: it intentionally only stops arm10.

`capture_stationary.py`, `inspect_capture.py` and alignment/fiducial inspection utilities remain reusable for recording evidence. Raw depth remains counts until scale is verified. Parent-frame configuration and an attractive point-cloud overlay are not proof of extrinsic calibration. There is no commissioned physical bucket localization algorithm in this package.

Minimum data the UI/backend handoff must carry:

- Both arms' readback joints, timestamps, moving flags, resource names and evidence source.
- Board impact-plane pose in world, measured dimensions, mount/base calibration IDs and the joint snapshot used for FK.
- Camera serial/frame, intrinsics, aligned RGB/depth references and metric scale, exposure/readback timing limitations.
- Bucket top-opening center/normal, radius, uncertainty, observation ID and detector/version evidence; missing remains `null`.
- Verified camera-to-tool/world transform, known-reference calibration ID, matching observation and serial. The pure transform function rejects mismatched frames/serial/observation, reflections, nonrigid rotations and mixed synthetic/physical evidence.
- Validated scan/loading/release poses and limits, held-ball offset, collision geometry and native path validation evidence.
- Recording readiness and trial/event IDs. UI must gate with `available_actions`, show `hardware_enabled:false`, and never interpret a synthetic fixture as a physical pose.

## Measurements and commissioning still needed

- Camera is user-confirmed on arm10's tool bracket, unchanged. Reported D435 serial112322070710 must be pinned/verified at reconnect. Camera-to-tool extrinsics, world registration, exact depth scale, alignment and exposure-to-joint association require reference-based validation.
- Board measures200×300×5mm, landscape90°. Physical display proxy is parented to `vacuum-board` at Z122mm; the numerical simulator uses a provisional100mm mount. Neither establishes calibrated arm14 tool-to-impact-surface pose. Measure the actual translation/rotation and account for compliance.
- Arm base separation1337.31mm is manually supplied52.65in; full relative base rotation and translation still need verification.
- Bucket inner top/bottom diameters177/150mm and outer200/158mm. Reported175mm remains ambiguous overall height vs interior depth. Straight-sided taper and placement remain provisional. Preserve prior scene fingerprints/results.
- Historical screenshot TCP coordinates load[150,-600,200]mm and drop[600,-150,600]mm; orientation[-0.01,-0.01,-1,-85.26] is a Viam orientation vector plus angle, **not Euler angles**. These are evidence only, not validated world poses, limits, scan positions or safe trajectories.
- Audit whether `sfhack14` still references the same physical board-arm controller. Establish exactly one motion owner before future board commands.
- Verify stop/cancellation ownership, gripper held-object response, recording-before-release, capture-control and Sequences, and complete native paths with calibrated geometry under operator supervision.

Recommended next physical step: one bounded stationary RGB-D capture at a deliberately taught scan pose, with a known-size/reference target and both joint-state brackets. Verify metric depth and calibrated transforms before implementing world-frame bucket pose or any autonomous release.

## Verification and SDK references

Mock tests cover blocked defaults, injected-adapter rejection, unknown detector, calibration/serial/observation mismatch, stale scan, changed board, unvalidated plan, recording/grasp/settling failure, command conflicts, timeout, cancellation of owned board/dropper moves, no tool release on stop, UUID/provenance, and Viam SDK model construction/registration. The module was additionally served on an isolated loopback-only Viam server for native RPC validation, described below. It has not been installed on sfhack10 or any physical machine. These checks validate software contracts and local RPC integration only.

The implementation was checked against installed `viam-sdk==0.85.1` signatures and current official documentation: [generic service](https://docs.viam.com/reference/services/generic/), [logic modules](https://docs.viam.com/manage/software/control-logic/), [module registration](https://docs.viam.com/extend/modular-resources/), and [Viam module architecture](https://docs.viam.com/reference/glossary/).

## Native local RPC validation — October 30, 2026

The module also loaded successfully in viam-server1.14.3 on a dedicated loopback
port18768 with no cloud identity, remotes or hardware components. The coordinator
refused Grip & drop in blocked mode; vision reported unknown with detection
capabilities false and no camera read. A separate mock coordinator completed all
seven workflow RPC stages and retained outcome unknown. This validates module
registration/RPC and the synthetic state machine, not physical motion or vision.
The temporary native server was stopped after validation. Evidence is local under
`.runtime/commissioning-native/verification-repro.json` for the final reproducible CLI run (the earlier local probe record is `verification.json`).

Always set a dedicated `VIAM_HOME` when running this commissioning test. The first
startup used the default Viam home and Viam logged removal of the unused local
`sircus-two-arm` module-data directory. The project module stores its records under
its repository data directory, not `VIAM_MODULE_DATA`; no existing Viam process was
running at the check. Subsequent validation used an isolated home. Source files,
project trial evidence and cloud configuration were not modified by this test.


## Cloud staging — October 30, 2026

A private, **unattached** fragment was created, saved and reloaded in Kadir Firat's Org:

- Name: `sircus-deployment-stubs-20261030`
- [Open the staged fragment](https://app.viam.com/fragment/51f52996-b901-403c-83fa-275a10284122/configure/builder?org=c3773683-dc59-4095-b21b-3a5372bc5f6b)
- Builder verified two services, Private visibility and **zero deployments**.
- The Hackathons organization did not expose a fragment-creation control in the inspected session; staging therefore used the user-owned organization available in that work-account session. This is not proof of permission to install or deploy into Hackathons.

The fragment contains the blocked coordinator/vision-stub template. It is a saved configuration asset, not an installed executable, working detector or deployed controller. No physical machine configurations were changed. Actual installation, cross-organization access if needed, calibrated physical adapter implementation and commissioning remain separate tasks.

### Missing-value wire caveat

In the pinned SDK's native RPC verification, Python `None` values decoded as numeric `0` in the captured generic `DoCommand` response (including missing estimate/error/trial-ID fields). Consequently, consumers must use explicit status/capability fields, not interpret `estimate:0` as a pose. The vision stub now returns `estimate_available:false` alongside `status:unknown`, `camera_read:false` and `hardware_enabled:false`; its unavailable estimate must remain unusable regardless of null decoding. The explicit absence flag passed module unit tests and the final isolated native RPC verification; the temporary server was stopped afterward.


## Reproduce the isolated native RPC check

`python -m sircus.deployment.check_native` probes an **already running** test server. It never launches a server, reads credentials, resolves physical dependencies or accepts non-loopback targets. It checks an explicit local configuration, rejects components/remotes/cloud/auth, restricts service/resource inventory to the known stubs (plus optional built-in motion), checks the blocked coordinator and unknown vision, and optionally rehearses the separately named mock coordinator. The provided configuration must point to this checkout's module launcher. Configuration validation assumes the operator started the server with that file; it cannot attest an arbitrary existing process's startup configuration.

Use an unused loopback port and a fresh directory. In a terminal at the repository root with the pinned Python environment active, derive this local configuration from the committed fragment template:

```sh
export SIRCUS_MODULE_PYTHON="$(command -v python)"
python - <<'PYCONFIG'
import json, os
from pathlib import Path
root = Path.cwd().resolve()
run = root / '.runtime' / 'native-stub-check'
run.mkdir(parents=True, exist_ok=False)
(run / 'home').mkdir()
config = json.loads((root / 'sircus/deployment/viam-fragment.template.json').read_text())
config['modules'][0]['executable_path'] = str(root / 'sircus/deployment/run-module.sh')
config['modules'][0]['env'] = {'SIRCUS_MODULE_PYTHON': os.environ['SIRCUS_MODULE_PYTHON']}
config['services'].append({
    'name': 'sircus-coordinator-rehearsal', 'api': 'rdk:service:generic',
    'model': 'sircus:deployment:operation-coordinator',
    'attributes': {'mode': 'dry_run', 'hardware_enabled': False}})
config['network'] = {'bind_address': '127.0.0.1:18768'}
(run / 'viam.json').write_text(json.dumps(config, indent=2) + '\n')
print(run)
PYCONFIG
```

Set `SIRCUS_NATIVE_SERVER` to the already installed, separately verified `viam-server` binary (the observed test used version1.14.3). Start only this dedicated foreground server; always use the isolated Viam home:

```sh
SIRCUS_NATIVE_SERVER=/absolute/path/to/viam-server
VIAM_HOME="$PWD/.runtime/native-stub-check/home" "$SIRCUS_NATIVE_SERVER" \
  --config "$PWD/.runtime/native-stub-check/viam.json" \
  --log-file "$PWD/.runtime/native-stub-check/server.log" \
  --disable-mdns --ftdc=false
```

On macOS, the native `viam-server` installation also needs its matching architecture shared libraries available to the dynamic loader. Use the platform installation’s library setup and, where needed, its documented loader search-path configuration. Those host-specific library locations belong in local environment setup, not this repository or the cloud fragment. The final test used a working local native installation; the two flags above disabled mDNS and FTDC during that isolated run.

In a second terminal at the same repository root, using the same pinned Python environment:

```sh
python -m sircus.deployment.check_native \
  --config .runtime/native-stub-check/viam.json \
  --target 127.0.0.1:18768 --rehearsal \
  --output .runtime/native-stub-check/verification.json
```

Then stop only the foreground test server with Ctrl-C. Do not run an existing simulation/physical launcher, reuse the normal Viam home, or attach the cloud fragment. To test only blocked/vision resources, omit the rehearsal service from the generated configuration and omit `--rehearsal`.

The observed, credential-free summary is committed at `docs/verification/native-stub-rpc-20261030.json`, derived from the actual latest local verification record and carrying its SHA256. It reports all seven mock stages ending in `complete`, trial outcome `unknown`, `estimate_available:false` and no physical verification. The source probe `python -m sircus.deployment.check_native` subsequently ran successfully against the isolated localhost configuration and exited 0. This summary now derives from that actual reproducible CLI run at 2026-10-30T12:08:14.500258Z, with server version queried as `v1.14.3`. The temporary server was stopped afterward. Configuration/inventory guards are also covered by unit tests; no physical validation is implied.
