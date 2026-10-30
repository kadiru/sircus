"""Typed physical commissioning boundary; all positions are millimetres.

A verified flag is evidence metadata, not an authorization to execute hardware.
The shipped controller has no physical execution backend regardless of these flags.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from typing import Literal, Protocol

Vec3 = tuple[float, float, float]
Source = Literal["observed", "kinematic", "manual", "synthetic"]


def vector(value, size=3):
    if len(value) != size or any(isinstance(x, bool) or not math.isfinite(float(x)) for x in value):
        raise ValueError(f"Expected {size} finite numbers")
    return tuple(float(x) for x in value)


def unit(value):
    value = vector(value)
    norm = math.sqrt(sum(x*x for x in value))
    if norm < 1e-9:
        raise ValueError("Normal has zero length")
    return tuple(x/norm for x in value)


@dataclass(frozen=True)
class Evidence:
    source: Source
    reference: str
    verified: bool = False
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class Pose:
    frame: str
    xyz_mm: Vec3
    orientation_vector_deg: tuple[float, float, float, float]
    evidence: Evidence

    def __post_init__(self):
        vector(self.xyz_mm)
        vector(self.orientation_vector_deg, 4)
        if not self.frame:
            raise ValueError("Pose frame required")


@dataclass(frozen=True)
class RobotState:
    timestamp_monotonic: float
    dropper_joints_deg: tuple[float, ...]
    board_joints_deg: tuple[float, ...]
    dropper_moving: bool
    board_moving: bool
    evidence: Evidence

    def __post_init__(self):
        vector(self.dropper_joints_deg, 6)
        vector(self.board_joints_deg, 6)
        if not math.isfinite(self.timestamp_monotonic):
            raise ValueError("State timestamp must be finite")
        if type(self.dropper_moving) is not bool or type(self.board_moving) is not bool:
            raise ValueError("Moving flags must be known booleans")


@dataclass(frozen=True)
class RGBDObservation:
    observation_id: str
    camera: str
    frame: str
    timestamp_monotonic: float
    camera_serial: str
    rgb_ref: str
    depth_ref: str
    depth_unit_mm: float | None
    depth_scale_verified: bool
    alignment_verified: bool
    intrinsics_verified: bool
    state_before: RobotState
    state_after: RobotState
    evidence: Evidence
    synchronization_limits: tuple[str, ...]


@dataclass(frozen=True)
class BucketEstimate:
    observation_id: str
    frame: str
    center_mm: Vec3
    normal: Vec3
    inner_top_radius_mm: float
    uncertainty_mm: float
    evidence: Evidence

    def __post_init__(self):
        vector(self.center_mm)
        unit(self.normal)
        if not math.isfinite(self.uncertainty_mm) or self.uncertainty_mm < 0:
            raise ValueError("Uncertainty must be finite and nonnegative")
        if not math.isfinite(self.inner_top_radius_mm) or self.inner_top_radius_mm <= 0:
            raise ValueError("Opening radius must be positive")


@dataclass(frozen=True)
class VerifiedTransform:
    source_frame: str
    target_frame: str
    rotation: tuple[Vec3, Vec3, Vec3]
    translation_mm: Vec3
    observation_id: str
    camera_serial: str
    calibration_id: str
    evidence: Evidence

    def validate(self):
        if not self.evidence.verified or not self.calibration_id or not self.evidence.reference:
            raise ValueError("Camera-to-world transform lacks verified reference calibration")
        vector(self.translation_mm)
        if len(self.rotation) != 3:
            raise ValueError("Rotation must be 3x3")
        rows = [vector(x) for x in self.rotation]
        for i in range(3):
            for j in range(3):
                if abs(sum(a*b for a,b in zip(rows[i],rows[j])) - (i == j)) > 1e-6:
                    raise ValueError("Transform rotation is not orthonormal")
        a,b,c = rows
        determinant = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
        if abs(determinant-1) > 1e-6:
            raise ValueError("Transform must be a proper rotation")


def camera_to_world(estimate: BucketEstimate, observation: RGBDObservation,
                    transform: VerifiedTransform) -> BucketEstimate:
    """Only transform a matched, metric, aligned observation with reference evidence."""
    transform.validate()
    if not (estimate.evidence.verified and estimate.evidence.reference and observation.evidence.verified):
        raise ValueError("Detection or observation evidence unverified")
    if not (observation.depth_scale_verified and observation.alignment_verified and observation.intrinsics_verified):
        raise ValueError("RGB-D metric scale/alignment/intrinsics verification missing")
    if observation.depth_unit_mm is None or not math.isfinite(observation.depth_unit_mm) or observation.depth_unit_mm <= 0:
        raise ValueError("Metric depth scale missing")
    if (estimate.observation_id != observation.observation_id or transform.observation_id != observation.observation_id
            or estimate.frame != observation.frame or transform.source_frame != observation.frame
            or transform.camera_serial != observation.camera_serial or transform.target_frame != "world"):
        raise ValueError("Observation/frame/serial mismatch in transform chain")
    # Do not allow a synthetic fixture to be presented as an observed world pose.
    if len({estimate.evidence.source == "synthetic", observation.evidence.source == "synthetic",
            transform.evidence.source == "synthetic"}) != 1:
        raise ValueError("Synthetic and physical evidence cannot be mixed")
    center = tuple(sum(row[i]*estimate.center_mm[i] for i in range(3))+t
                   for row,t in zip(transform.rotation, transform.translation_mm))
    normal = unit(tuple(sum(row[i]*estimate.normal[i] for i in range(3)) for row in transform.rotation))
    return BucketEstimate(estimate.observation_id, "world", center, normal, estimate.inner_top_radius_mm,
                          estimate.uncertainty_mm, Evidence(estimate.evidence.source,
                          f"{estimate.evidence.reference}; calibration={transform.calibration_id}", True,
                          estimate.evidence.limitations + observation.synchronization_limits))


@dataclass(frozen=True)
class BoardEstimate:
    center_mm: Vec3
    normal: Vec3
    joints_deg: tuple[float, ...]
    frame: str
    mount_calibration_id: str
    evidence: Evidence


@dataclass(frozen=True)
class ReleasePlan:
    plan_id: str
    revision: int
    bucket_observation_id: str
    board_joints_deg: tuple[float, ...]
    release_pose: Pose
    collision_checked: bool
    limits_checked: bool
    trajectory_checked: bool
    evidence: Evidence


class DeploymentAdapter(Protocol):
    """Execution port. A future commissioned adapter must implement all methods.

    Reuse physical_trials.ViamAdapter capture/recording and Motion.Move plumbing;
    do not duplicate credentials, bypass planning, or use raw arm move RPCs.
    No such physical adapter is constructed or connected by this package.
    """
    scan_pose: Pose
    loading_pose: Pose
    async def snapshot(self) -> RobotState: ...
    async def board_from_kinematics(self, state: RobotState) -> BoardEstimate: ...
    async def move_to_pose(self, arm: str, pose: Pose, operation_id: str) -> None: ...
    async def stop_motion(self, arm: str, operation_id: str) -> None: ...
    async def capture_rgbd(self) -> RGBDObservation: ...
    async def camera_world_transform(self, observation: RGBDObservation) -> VerifiedTransform: ...
    async def plan_release(self, board: BoardEstimate, bucket: BucketEstimate, revision: int) -> ReleasePlan: ...
    async def start_recording(self, trial_id: str, metadata: dict) -> bool: ...
    async def finish_recording(self, trial_id: str) -> None: ...
    async def close_gripper(self) -> bool: ...
    async def open_gripper(self) -> None: ...
    async def at_pose_and_settled(self, pose: Pose) -> bool: ...
    async def measured_release_pose(self) -> Pose: ...
    async def recording_status(self) -> bool: ...


class BucketDetector(Protocol):
    async def detect(self, observation: RGBDObservation) -> BucketEstimate | None: ...


def serializable(value):
    return asdict(value) if hasattr(value, "__dataclass_fields__") else value
