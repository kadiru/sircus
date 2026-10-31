"""RGB-D geometry estimator. No simulator imports, scene file or truth access.

Inputs: registered RGB, axial metre depth, camera intrinsics and calibrated
camera-to-world transform only. Uses appearance/shape priors, never object poses.
"""

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

COLORS = {
    "table": (60, 210, 230),
    "board": (50, 240, 140),
    "bucket": (255, 190, 45),
    "ball": (255, 90, 190),
}


def cloud(depth, k, transform):
    v, u = np.indices(depth.shape)
    xyz = np.stack(
        [(u - k[0, 2]) * depth / k[0, 0], (v - k[1, 2]) * depth / k[1, 1], depth],
        axis=-1,
    )
    return xyz @ transform[:3, :3].T + transform[:3, 3]


def components(mask, minimum=30):
    lab, _n = ndimage.label(mask)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    for i in np.argsort(sizes)[::-1][:12]:
        if sizes[i] >= minimum:
            yield lab == i


def fit_plane(points, threshold=0.003):
    if len(points) < 30:
        raise ValueError("insufficient plane support")
    rng = np.random.default_rng(18)
    sample = points[rng.choice(len(points), min(4000, len(points)), replace=False)]
    best = None
    for _ in range(60):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal /= norm
        use = np.abs((sample - a) @ normal) < threshold
        if best is None or use.sum() > best.sum():
            best = use
    if best is None or best.sum() < 25:
        raise ValueError("no supported plane")
    center = sample[best].mean(0)
    _, _, axes = np.linalg.svd(sample[best] - center, full_matrices=False)
    normal = axes[-1]
    if normal[2] < 0:
        normal = -normal
    use = np.abs((points - center) @ normal) < threshold
    p = points[use]
    center = p.mean(0)
    _, _, axes = np.linalg.svd(p - center, full_matrices=False)
    normal = axes[-1]
    if normal[2] < 0:
        normal = -normal
    uv = (p - center) @ axes[:2].T
    low, high = np.percentile(uv, [0.5, 99.5], axis=0)
    center = center + ((low + high) / 2) @ axes[:2]
    residual = (p - center) @ normal
    return center, normal, high - low, use, float(np.sqrt(np.mean(residual**2)))


def fit_sphere(points):
    if len(points) < 20:
        raise ValueError("insufficient sphere depth support")
    origin = points.mean(0)
    q = points - origin
    singular = np.linalg.svd(q, compute_uv=False)
    if singular[-1] / max(singular[0], 1e-9) < 0.06:
        raise ValueError("insufficient observed sphere curvature")
    a = np.column_stack([2 * q, np.ones(len(q))])
    b = (q * q).sum(1)
    sol = np.linalg.lstsq(a, b, rcond=None)[0]
    radius = np.sqrt(max(1e-8, sol[3] + sol[:3] @ sol[:3]))
    center = origin + sol[:3]
    if not 0.015 < radius < 0.08:
        raise ValueError("sphere radius inconsistent")
    fit = least_squares(
        lambda x: np.linalg.norm(points - x[:3], axis=1) - x[3],
        np.r_[center, radius],
        bounds=(np.r_[center - 0.1, 0.015], np.r_[center + 0.1, 0.08]),
        loss="soft_l1",
        f_scale=0.002,
        max_nfev=80,
    )
    rms = float(np.sqrt(np.mean(fit.fun**2)))
    if rms > 0.004:
        raise ValueError("sphere fit residual too high")
    span = np.linalg.norm(np.ptp(points, axis=0))
    if span < fit.x[3] * 0.65:
        raise ValueError("visible cap too small to locate center")
    return fit.x[:3], float(fit.x[3]), rms


def fit_rim(points, table_normal):
    # Upright-container prior: the opening is the highest colored rim support.
    height = points @ table_normal
    top = np.percentile(height, 99)
    rim = points[height > top - 0.003]
    if len(rim) < 18:
        raise ValueError("too little visible rim")
    center, normal, _, _, _rms = fit_plane(rim, 0.002)
    if normal @ table_normal < 0.95:
        raise ValueError("rim not consistent with upright bucket")
    x = np.cross(normal, [0, 1, 0])
    x /= np.linalg.norm(x)
    y = np.cross(normal, x)
    uv = (rim - center) @ np.array([x, y]).T

    def err(a):
        return np.linalg.norm(uv - a[:2], axis=1) - a[2]

    solution = least_squares(
        err,
        [0, 0, 0.095],
        bounds=([-0.2, -0.2, 0.04], [0.2, 0.2, 0.2]),
        loss="soft_l1",
        f_scale=0.004,
    )
    cc = center + solution.x[0] * x + solution.x[1] * y
    angles = np.arctan2(uv[:, 1] - solution.x[1], uv[:, 0] - solution.x[0])
    coverage = np.count_nonzero(np.histogram(angles, bins=12, range=(-np.pi, np.pi))[0])
    radial_rms = float(np.sqrt(np.mean(solution.fun**2)))
    if coverage < 7 or radial_rms > 0.009:
        raise ValueError("opening rim is too incomplete or noncircular")
    return cc, normal, float(solution.x[2]), radial_rms, len(rim), coverage


def estimate(rgb, depth, k, transform, frame_id="unknown", timestamp=None):
    rgb = np.asarray(rgb)
    depth = np.asarray(depth, dtype=float)
    k = np.asarray(k, dtype=float)
    transform = np.asarray(transform, dtype=float)
    if rgb.shape != (*depth.shape, 3) or k.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("invalid registered RGB-D/calibration shapes")
    if (
        not np.isfinite(k).all()
        or min(k[0, 0], k[1, 1]) <= 0
        or not np.isfinite(transform).all()
    ):
        raise ValueError("invalid calibration")
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 5)
    xyz = cloud(np.where(valid, depth, np.nan), k, transform)
    objects = {
        n: {"status": "missing", "reason": "no supported RGB-D candidate"}
        for n in COLORS
    }
    masks = {}
    candidates = []
    # Local depth normals isolate horizontal support surfaces without a known Z.
    dx = xyz[:, 3:] - xyz[:, :-3]
    dy = xyz[3:] - xyz[:-3]
    norm = np.cross(dx[:-3], dy[:, :-3])
    lengths = np.linalg.norm(norm, axis=-1)
    horizontal = np.zeros(valid.shape, dtype=bool)
    horizontal[:-3, :-3] = (
        (np.abs(norm[:, :, 2]) > 0.985 * lengths)
        & (lengths > 1e-8)
        & (np.linalg.norm(dx[:-3], axis=-1) < 0.05)
        & (np.linalg.norm(dy[:, :-3], axis=-1) < 0.05)
    )
    horizontal &= valid
    for mask in components(horizontal, 500):
        try:
            center, normal, extent, use, rms = fit_plane(xyz[mask])
            if min(extent) < 0.45 or max(extent) > 3.5 or normal[2] < 0.98:
                continue
            candidates.append(
                (center[2], int(use.sum()), mask, center, normal, extent, rms)
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
    table_normal = np.array([0.0, 0.0, 1.0])
    table_center = None
    if candidates:
        _, _, mask, center, normal, extent, rms = max(candidates, key=lambda a: a[0])
        table_center, table_normal = center, normal
        objects["table"] = {
            "status": "estimated",
            "center_m": center.tolist(),
            "normal": normal.tolist(),
            "visible_extent_m": extent.tolist(),
            "plane_offset_m": float(-normal @ center),
            "support_pixels": int(mask.sum()),
            "fit_rms_m": rms,
            "quality": "good",
            "caveat": "highest broad horizontal support plane; center is visible patch, not full table center",
        }
        masks["table"] = mask
    colors = rgb.astype(float) / 255
    r, g, b = np.moveaxis(colors, -1, 0)
    appearance = {
        "ball": (r > g * 1.5) & (b > g * 1.35) & (r > b * 0.85) & (r > 0.15),
        "bucket": (r > b * 1.5) & (g > b * 1.3) & (g > r * 0.45) & (r > 0.2),
        "board": ((colors.max(-1) - colors.min(-1)) < 0.12 * colors.max(-1))
        & (colors.mean(-1) > 0.1)
        & (colors.mean(-1) < 0.75),
    }
    if table_center is not None:
        height = (xyz - table_center) @ table_normal
    else:
        height = np.full(depth.shape, np.nan)
    for name, color_mask in appearance.items():
        possible = []
        mask0 = color_mask & valid
        if name == "board":
            mask0 &= height > 0.08
        # Remove blended silhouettes/flying depth at object edges.
        interior = ndimage.binary_erosion(mask0, iterations=1)
        for mask in components(interior, 20 if name == "ball" else 60):
            points = xyz[mask]
            try:
                if name == "ball":
                    center, radius, rms = fit_sphere(points)
                    result = {
                        "status": "estimated",
                        "center_m": center.tolist(),
                        "radius_m": radius,
                        "fit_rms_m": rms,
                        "support_pixels": len(points),
                        "quality": "good" if len(points) > 60 else "limited",
                        "caveat": "pink color-assisted sphere fit; hidden balls are unknown",
                    }
                elif name == "bucket":
                    if table_center is None:
                        raise ValueError("table plane unavailable")
                    center, normal, radius, rms, count, coverage = fit_rim(
                        points, table_normal
                    )
                    result = {
                        "status": "estimated",
                        "center_m": center.tolist(),
                        "normal": normal.tolist(),
                        "rim_mid_radius_m": radius,
                        "fit_rms_m": rms,
                        "support_pixels": count,
                        "rim_sectors_observed": coverage,
                        "quality": "good" if coverage >= 10 else "limited",
                        "caveat": "yellow color-assisted rim fit; upright bucket prior; midpoint rim radius is not inner opening radius",
                    }
                else:
                    center, normal, extent, use, rms = fit_plane(points, 0.002)
                    if (
                        min(extent) < 0.13
                        or max(extent) > 0.36
                        or min(extent) > 0.25
                        or max(extent) < 0.22
                        or use.mean() < 0.85
                    ):
                        raise ValueError("plane does not fit the board size prior")
                    partial = (
                        abs(max(extent) - 0.3) > 0.03 or abs(min(extent) - 0.2) > 0.025
                    )
                    result = {
                        "status": "estimated",
                        "center_m": center.tolist(),
                        "normal": normal.tolist(),
                        "visible_extent_m": extent.tolist(),
                        "fit_rms_m": rms,
                        "support_pixels": int(use.sum()),
                        "quality": "limited" if partial else "good",
                        "caveat": "neutral-color planar rectangle; 200x300mm size prior; partial visibility biases center",
                    }
                possible.append((len(points), mask, result))
            except (ValueError, np.linalg.LinAlgError):
                continue
        if possible:
            _, mask, result = max(possible, key=lambda p: p[0])
            objects[name] = result
            masks[name] = mask
    return {
        "frame_id": str(frame_id),
        "timestamp_utc": timestamp,
        "objects": objects,
        "coordinate_frame": "world via supplied camera calibration",
        "method": "RGB-assisted depth geometry; known shape/appearance priors; not learned instance segmentation",
        "ground_truth_used": False,
        "quality_note": "quality labels are geometric checks, not calibrated probabilities",
        "capture_verdict": "unknown: temporal visual bounce/retention observer not implemented",
    }, masks
