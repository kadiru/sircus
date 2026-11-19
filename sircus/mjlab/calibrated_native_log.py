"""Dense, read-only native contact observations for the calibrated v2 study.

The solver's contact fields describe the integration interval's collision state.
Pre/post ball states and a CPU reconstruction check establish their timing; this
module never calls forward(), changes contact parameters or actuates anything.
"""

import warp as wp


VERSION = "sircus-native-substep-contact-log-v1"


@wp.kernel
def minimum_bucket_contacts(
    count: wp.array(dtype=wp.int32),
    worlds: wp.array(dtype=wp.int32),
    geom: wp.array(dtype=wp.vec2i),
    distance: wp.array(dtype=wp.float32),
    position: wp.array(dtype=wp.vec3),
    frame: wp.array(dtype=wp.mat33),
    tags: wp.array(dtype=wp.int32),
    ball_geom: int,
    tick: int,
    output_distance: wp.array3d(dtype=wp.float32),
    output_geom: wp.array4d(dtype=wp.int32),
    output_position: wp.array4d(dtype=wp.float32),
    output_normal: wp.array4d(dtype=wp.float32),
):
    world = wp.tid()
    for category in range(2):
        nearest = float(wp.inf)
        first = int(-1)
        second = int(-1)
        point = wp.vec3(0.0)
        normal = wp.vec3(0.0)
        for contact in range(count[0]):
            if worlds[contact] == world:
                a = geom[contact][0]
                b = geom[contact][1]
                other = int(-1)
                if a == ball_geom:
                    other = b
                elif b == ball_geom:
                    other = a
                if other >= 0:
                    if tags[other] == category + 1 and distance[contact] < nearest:
                        nearest = distance[contact]
                        first = a
                        second = b
                        point = position[contact]
                        normal = wp.vec3(frame[contact][0, 0], frame[contact][0, 1], frame[contact][0, 2])
        output_distance[tick, world, category] = nearest
        output_geom[tick, world, category, 0] = first
        output_geom[tick, world, category, 1] = second
        for axis in range(3):
            output_position[tick, world, category, axis] = point[axis]
            output_normal[tick, world, category, axis] = normal[axis]


class DenseNativeLog:
    """Record every integration state and each interval's observed contacts.

    Call on the simulator's Warp stream (also set as the active Torch stream).
    ``time_s`` is exact tick-derived simulation time. The separately stored
    ``native_time_s`` exposes the engine's accumulated floating-point clock.
    """

    def __init__(self, simulation, count, ticks, ball_q, ball_v, ball_geom, tags):
        import torch

        self.sim, self.count, self.ticks = simulation, count, ticks
        self.ball_q, self.ball_v, self.ball_geom = ball_q, ball_v, ball_geom
        self.dt = float(simulation.mj_model.opt.timestep)
        device = simulation.device if hasattr(simulation, "device") else "cuda:0"
        self.position = torch.empty((ticks + 1, count, 3), device=device)
        self.velocity = torch.empty_like(self.position)
        self.clock = torch.empty((ticks + 1, count), device=device)
        self.forces = torch.empty((ticks, count, 6), device=device)
        self.distance = torch.empty((ticks, count, 2), device=device)
        self.geom = torch.empty((ticks, count, 2, 2), device=device, dtype=torch.int32)
        self.point = torch.empty((ticks, count, 2, 3), device=device)
        self.normal = torch.empty_like(self.point)
        self.tags = wp.array(tags, dtype=wp.int32, device=simulation.wp_device)
        self.outputs = [
            wp.from_torch(value)
            for value in (self.distance, self.geom, self.point, self.normal)
        ]
        self._state(0)

    def _state(self, index):
        data = self.sim.data
        self.position[index].copy_(data.qpos[:self.count, self.ball_q:self.ball_q + 3])
        self.velocity[index].copy_(data.qvel[:self.count, self.ball_v:self.ball_v + 3])
        self.clock[index].copy_(data.time[:self.count])

    def step(self, index, force):
        self._state(index + 1)
        self.forces[index].copy_(force[:self.count])
        contact = self.sim.data.contact
        wp.launch(
            minimum_bucket_contacts,
            dim=self.count,
            inputs=[
                self.sim.data.nacon.wp_array,
                contact.worldid.wp_array,
                contact.geom.wp_array,
                contact.dist.wp_array,
                contact.pos.wp_array,
                contact.frame.wp_array,
                self.tags,
                self.ball_geom,
                index,
                *self.outputs,
            ],
            device=self.sim.wp_device,
        )

    def arrays(self, count=None):
        import numpy as np
        import torch

        torch.cuda.synchronize()
        count = self.count if count is None else count
        if not 0 < count <= self.count:
            raise ValueError("Invalid recorded world count")
        values = {
            "position_m": self.position[:, :count].cpu().numpy(),
            "velocity_m_s": self.velocity[:, :count].cpu().numpy(),
            "forces_N": self.forces[:, :count].cpu().numpy(),
            "native_time_s": self.clock[:, :count].cpu().numpy(),
            "contact_distance_m": self.distance[:, :count].cpu().numpy(),
            "contact_geom_ids": self.geom[:, :count].cpu().numpy(),
            "contact_position_m": self.point[:, :count].cpu().numpy(),
            "contact_normal": self.normal[:, :count].cpu().numpy(),
            "time_s": np.arange(self.ticks + 1, dtype=np.float64) * self.dt,
            "dt_s": np.asarray(self.dt, dtype=np.float64),
        }
        if not np.isfinite(values["native_time_s"]).all():
            raise RuntimeError("Non-finite native clock")
        if np.max(np.ptp(values["native_time_s"], axis=1)) > 1e-7:
            raise RuntimeError("World clocks disagree")
        return values
