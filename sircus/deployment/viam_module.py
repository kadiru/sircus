"""Viam SDK0.85.1 module: blocked coordinator + explicitly unavailable vision.

Registration and serving happen only under main(). Neither resource resolves or
commands physical dependencies, even when installed on a physical machine.
"""
from __future__ import annotations
import asyncio
from google.protobuf.json_format import MessageToDict
from viam.module.module import Module
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.services.vision import Vision
from viam.proto.service.vision import GetPropertiesResponse
from .controller import DeploymentController

FAMILY = ModelFamily("sircus", "deployment")
COORDINATOR_MODEL = Model(FAMILY, "operation-coordinator")
VISION_MODEL = Model(FAMILY, "bucket-vision-stub")


def attributes(config):
    return MessageToDict(config.attributes)


def validate_coordinator(config):
    attrs = attributes(config)
    if attrs.get("mode", "blocked") not in ("blocked", "dry_run") or attrs.get("hardware_enabled", False) is not False:
        raise ValueError("This module cannot enable hardware; only blocked/dry_run are valid")
    return [], []  # No automatic physical-resource resolution.


class OperationCoordinator(Generic):
    @classmethod
    def new(cls, config, dependencies):
        validate_coordinator(config)
        return cls(config.name, attributes(config).get("mode", "blocked"))

    def __init__(self, name, mode="blocked"):
        super().__init__(name)
        self.controller = DeploymentController(mode)

    async def do_command(self, command, *, timeout=None, **kwargs):
        return await self.controller.command(dict(command))

    async def close(self):
        await self.controller.close()


class BucketVisionStub(Vision):
    @classmethod
    def new(cls, config, dependencies):
        return cls(config.name)

    async def do_command(self, command, *, timeout=None, **kwargs):
        if command.get("action", "status") not in ("status", "capabilities"):
            raise ValueError("Bucket vision is uncommissioned; only status/capabilities supported")
        return {"schema": "sircus.bucket-perception.v1", "status": "unknown", "estimate_available": False, "estimate": None,
                "hardware_enabled": False, "camera": "cam", "camera_read": False,
                "reason": "No physical detector or verified camera-world calibration is installed"}

    async def get_properties(self, *, extra=None, timeout=None):
        return GetPropertiesResponse(classifications_supported=False, detections_supported=False,
                                     object_point_clouds_supported=False, detections_3d_supported=False,
                                     default_camera="cam")

    async def get_detections(self, image, *, extra=None, timeout=None):
        raise NotImplementedError("Bucket detector uncommissioned; estimate is unknown")
    async def get_detections_from_camera(self, camera_name, *, extra=None, timeout=None):
        raise NotImplementedError("Bucket detector uncommissioned; no camera RPC performed")
    async def get_detections_3d(self, camera_name, *, extra=None, timeout=None):
        raise NotImplementedError("Metric bucket detector and transform calibration missing")
    async def get_classifications(self, image, count, *, extra=None, timeout=None):
        raise NotImplementedError("Classifications unsupported")
    async def get_classifications_from_camera(self, camera_name, count, *, extra=None, timeout=None):
        raise NotImplementedError("Classifications unsupported")
    async def get_object_point_clouds(self, camera_name, *, extra=None, timeout=None):
        raise NotImplementedError("Bucket segmentation unsupported")
    async def capture_all_from_camera(self, camera_name, return_image=False, return_classifications=False,
                                      return_detections=False, return_object_point_clouds=False,
                                      return_detections_3d=False, *, extra=None, timeout=None):
        raise NotImplementedError("No physical capture/detector is commissioned")


def register_models():
    Registry.register_resource_creator(Generic.API, COORDINATOR_MODEL,
                                       ResourceCreatorRegistration(OperationCoordinator.new, validate_coordinator))
    Registry.register_resource_creator(Vision.API, VISION_MODEL, ResourceCreatorRegistration(BucketVisionStub.new))


async def main():
    register_models()
    module = Module.from_args()
    module.add_model_from_registry(Generic.API, COORDINATOR_MODEL)
    module.add_model_from_registry(Vision.API, VISION_MODEL)
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())
