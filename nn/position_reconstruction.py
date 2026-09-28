"""Tangent-guided position reconstruction (paper section 5, analytic).

Five-step pipeline over one frame, all vectorized:

  classification     hair pixel: Psi coverage > 0
                     valid pixel:  input coverage > 0 (a fragment was
                     rasterized -> its position is a real sample)
  depth inpainting   missing depth copied from the nearest pixel with
                     depth data (eq. 9, via the exact EDT)
  stage I            per-pixel world-space step l_p at depth D, measured
                     through the actual view-projection matrix
  stage II           screen-space curvature centers: Kasa circle fit over
                     the hair pixels of an 11x11 window, radius >= 1 px
  stage III          3D tangents projected to the image plane
  stage IV           backward repair (eqs. 11-12): for each invalid hair
                     pixel take the valid 3x3 neighbor with the closest
                     curvature center, then P*(p) = P(s*) + l_p * T*(p)
  stage V            forward voting (eqs. 13-15): each valid pixel votes
                     P(s) + l_s * T(s) into its 8 neighbors; accepted when
                     the screen tangent aligns with the vote direction
                     within 30 deg for BOTH tangent signs; the recipient
                     keeps the frontmost vote (by screen depth)
  composition        valid pixels keep their real sample; invalid pixels
                     take backward repair, else frontmost vote, else the
                     inpainted-depth ray fallback.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage

COS_THETA_MAX = float(np.cos(np.radians(30.0)))
CURVATURE_WINDOW = 11


def pixel_coordinates(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:height, 0:width]
    return (xs + 0.5).astype(np.float64), (ys + 0.5).astype(np.float64)


def project_to_pixels(points: np.ndarray, view: np.ndarray,
                      projection: np.ndarray, width: int, height: int) -> np.ndarray:
    """World points (..., 3) -> pixel coordinates (..., 2), y down."""
    clip = np.concatenate([points, np.ones(points.shape[:-1] + (1,))], axis=-1)
    clip = clip @ (projection @ view).T
    ndc = clip[..., :3] / clip[..., 3:4]
    return np.stack([(ndc[..., 0] + 1.0) * 0.5 * width,
                     (1.0 - ndc[..., 1]) * 0.5 * height], axis=-1)


def unproject_pixels(px: np.ndarray, py: np.ndarray, depth: np.ndarray,
                     view: np.ndarray, projection: np.ndarray,
                     width: int, height: int) -> np.ndarray:
    ndc_x = 2.0 * px / width - 1.0
    ndc_y = 1.0 - 2.0 * py / height
    clip = np.stack([ndc_x, ndc_y, depth, np.ones_like(depth)], axis=-1)
    world = clip @ np.linalg.inv(projection @ view).T
    return world[..., :3] / world[..., 3:4]


def pixel_world_step(depth: np.ndarray, view: np.ndarray,
                     projection: np.ndarray) -> np.ndarray:
    """Stage I: world-space length of one pixel step at each pixel depth."""
    height, width = depth.shape[:2]
    xs, ys = pixel_coordinates(height, width)
    center = unproject_pixels(xs, ys, depth, view, projection, width, height)
    right = unproject_pixels(xs + 1.0, ys, depth, view, projection, width, height)
    up = unproject_pixels(xs, ys - 1.0, depth, view, projection, width, height)
    step = np.sqrt(((right - center) ** 2).sum(-1) + ((up - center) ** 2).sum(-1))
    return step.astype(np.float32)


def screen_tangents(tangent: np.ndarray, position: np.ndarray, view: np.ndarray,
                    projection: np.ndarray, width: int, height: int) -> np.ndarray:
    """Stage III: project each pixel's 3D tangent into the image plane."""
    tip = position + tangent * 0.01
    base = project_to_pixels(position, view, projection, width, height)
    tip = project_to_pixels(tip, view, projection, width, height)
    screen = tip - base
    norm = np.linalg.norm(screen, axis=-1, keepdims=True)
    return (screen / np.maximum(norm, 1e-8)).astype(np.float32)


def inpaint_depth(coverage: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Eq. 9: missing hair depth from the nearest pixel with depth data."""
    has_depth = (coverage > 0.0) & (depth > 0.0) & (depth < 1.0)
    filled = depth.copy()
    if has_depth.any():
        _, indices = ndimage.distance_transform_edt(~has_depth, return_indices=True)
        filled = depth[tuple(indices)]
    return np.where(coverage > 0.0, filled, depth).astype(np.float32)


def curvature_centers(xs: np.ndarray, ys: np.ndarray, hair: np.ndarray,
                      window: int = CURVATURE_WINDOW) -> np.ndarray:
    """Stage II: per-pixel screen-space curvature centers from a Kasa
    circle fit over the hair pixels of an 11x11 window; radius >= 1 px."""
    height, width = hair.shape
    radius = window // 2

    xs_p = np.pad(xs, radius, mode="edge").astype(np.float64)
    ys_p = np.pad(ys, radius, mode="edge").astype(np.float64)
    hair_p = np.pad(hair.astype(np.float32), radius, mode="constant")

    xs_w = sliding_window_view(xs_p, (window, window))  # views, no copy
    ys_w = sliding_window_view(ys_p, (window, window))
    mask_w = sliding_window_view(hair_p, (window, window))

    # Kasa normal-equation accumulators, computed in row chunks (the full
    # window views would materialize ~900 MB each in float64).
    zeros = lambda: np.zeros((height, width), dtype=np.float64)
    count, sx, sy = zeros(), zeros(), zeros()
    sxx, syy, sxy = zeros(), zeros(), zeros()
    sz, sxz, syz = zeros(), zeros(), zeros()

    chunk = 64
    for y0 in range(0, height, chunk):
        y1 = min(y0 + chunk, height)
        m = mask_w[y0:y1]
        xw = xs_w[y0:y1].astype(np.float64)
        yw = ys_w[y0:y1].astype(np.float64)
        zw = xw ** 2 + yw ** 2

        def acc(target, field):
            target[y0:y1] += (field * m).sum(axis=(-2, -1))

        acc(count, np.ones_like(m))
        acc(sx, xw); acc(sy, yw)
        acc(sxx, xw ** 2); acc(syy, yw ** 2); acc(sxy, xw * yw)
        acc(sz, zw); acc(sxz, xw * zw); acc(syz, yw * zw)

    centers = np.stack([xs, ys], axis=-1).astype(np.float64)
    chunk = 96
    for y0 in range(0, height, chunk):
        y1 = min(y0 + chunk, height)
        a = np.stack([
            np.stack([sxx[y0:y1], sxy[y0:y1], sx[y0:y1]], axis=-1),
            np.stack([sxy[y0:y1], syy[y0:y1], sy[y0:y1]], axis=-1),
            np.stack([sx[y0:y1], sy[y0:y1], count[y0:y1]], axis=-1),
        ], axis=-2)
        rhs = np.stack([sxz[y0:y1], syz[y0:y1], sz[y0:y1]], axis=-1)

        # Ridge regularization keeps empty windows solvable; a circle fit
        # needs at least 3 points, anything else falls back to the pixel.
        eye = np.eye(3, dtype=np.float64)
        solution = np.linalg.solve(a + 1e-9 * eye, rhs)  # (h, W, 3)
        fittable = count[y0:y1] >= 3
        center_x = np.where(fittable, solution[..., 0] * 0.5, xs[y0:y1])
        center_y = np.where(fittable, solution[..., 1] * 0.5, ys[y0:y1])

        radius_px = np.hypot(center_x - xs[y0:y1], center_y - ys[y0:y1])
        scale = np.clip(radius_px, 1.0, None) / np.maximum(radius_px, 1e-9)
        center_x = xs[y0:y1] + (center_x - xs[y0:y1]) * scale
        center_y = ys[y0:y1] + (center_y - ys[y0:y1]) * scale

        finite = np.isfinite(center_x) & np.isfinite(center_y)
        centers[y0:y1, :, 0] = np.where(finite, center_x, xs[y0:y1])
        centers[y0:y1, :, 1] = np.where(finite, center_y, ys[y0:y1])
    return centers.astype(np.float32)


def view_space_z(position: np.ndarray, view: np.ndarray) -> np.ndarray:
    """View-space depth of world positions (..., 3)."""
    hom = np.concatenate([position, np.ones(position.shape[:-1] + (1,))], axis=-1)
    return (hom @ view.T)[..., 2]


def _shift(map2d: np.ndarray, dy: int, dx: int, fill) -> np.ndarray:
    padded = np.full((map2d.shape[0] + 2, map2d.shape[1] + 2), fill,
                     dtype=map2d.dtype)
    padded[1:-1, 1:-1] = map2d
    return padded[1 + dy:1 + dy + map2d.shape[0],
                  1 + dx:1 + dx + map2d.shape[1]]


def reconstruct_positions(coverage_pred: np.ndarray, tangent_pred: np.ndarray,
                          coverage_input: np.ndarray, position_input: np.ndarray,
                          depth_input: np.ndarray, view: np.ndarray,
                          projection: np.ndarray,
                          max_propagation_px: int = 4,
                          max_iterations: int = 8) -> dict:
    height, width = coverage_pred.shape[:2]
    xs, ys = pixel_coordinates(height, width)
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]

    hair = coverage_pred > 0.0
    valid = (coverage_input > 0.0) & hair

    # Reconstruction can only propagate a couple of pixels from a real
    # sample (3x3 repair + 1-px voting). Pixels deeper inside missed
    # regions have no geometric evidence and stay unfilled - the paper's
    # support-mask suppression handles them at shading time.
    if (coverage_input > 0.0).any():
        distance_to_valid = ndimage.distance_transform_edt(coverage_input <= 0.0)
    else:
        distance_to_valid = np.full((height, width), np.inf, dtype=np.float32)
    invalid = hair & ~valid & (distance_to_valid <= max_propagation_px)

    # Depth inpainting (eq. 9) and stage I.
    depth = inpaint_depth(coverage_input, depth_input)
    step = pixel_world_step(depth, view, projection)

    # Stage III: screen-space tangents at hair pixels. After depth
    # inpainting EVERY hair pixel has a depth, so the tangent projection is
    # anchored at the pixel's own (inpainted) position - anchoring invalid
    # pixels at the world origin would give garbage screen directions.
    position_anchor = unproject_pixels(*pixel_coordinates(width=width, height=height),
                                       depth=depth, view=view,
                                       projection=projection, width=width,
                                       height=height).astype(np.float32)
    position_anchor = np.where(valid[..., None], position_input,
                               position_anchor).astype(np.float32)
    tangent_hair = np.where(hair[..., None], tangent_pred, 0.0).astype(np.float32)
    t_screen = screen_tangents(tangent_hair, position_anchor, view, projection,
                               width, height)
    # Voting sources are valid pixels: they project from their real samples.
    position_valid = np.where(valid[..., None], position_input,
                              position_anchor).astype(np.float32)

    # Stage II: curvature centers.
    centers = curvature_centers(xs, ys, hair)

    # Depth-consistency guard: propagation across a depth discontinuity
    # (two strands crossing in screen space) walks into the wrong strand.
    target_vz = view_space_z(
        unproject_pixels(xs, ys, depth, view, projection, width, height), view)
    depth_tolerance = 8.0 * step  # adaptive: ~8 pixel steps in depth

    # Stage IV: backward repair (eqs. 11-12), applied iteratively so the
    # propagation walks along strands through gaps wider than one pixel:
    # each pass treats already-repaired pixels as sources for their 3x3
    # neighbors, exactly like the single-pass eq. 12 but chained.
    best_dist = np.full((height, width), np.inf, dtype=np.float64)
    has_position = valid.copy()
    current = position_input.copy()
    repaired = position_input.copy()
    has_repair = np.zeros((height, width), dtype=bool)
    repair = np.zeros((height, width, 3), dtype=np.float32)
    tangent_step = tangent_hair * step[..., None]

    for _ in range(max_iterations):
        newly = invalid & ~has_position
        if not newly.any():
            break
        took_any = False
        for dy, dx in offsets:
            neighbor_has = _shift(has_position.astype(np.float32), dy, dx, 0.0) > 0.5
            candidate = neighbor_has & newly
            if not candidate.any():
                continue
            neighbor_center = np.stack([_shift(centers[..., 0], dy, dx, np.nan),
                                        _shift(centers[..., 1], dy, dx, np.nan)], axis=-1)
            dist = ((neighbor_center - centers) ** 2).sum(-1)
            source_vz = view_space_z(
                np.stack([_shift(current[..., 0], dy, dx, 0.0),
                          _shift(current[..., 1], dy, dx, 0.0),
                          _shift(current[..., 2], dy, dx, 0.0)], axis=-1), view)
            depth_ok = np.abs(source_vz - target_vz) <= depth_tolerance
            take = candidate & np.isfinite(dist) & (dist < best_dist) & depth_ok
            if not take.any():
                continue
            for ch in range(3):
                source_p = _shift(current[..., ch], dy, dx, 0.0)
                repair[..., ch] = np.where(take, source_p, repair[..., ch])
            # Orient the tangent step TOWARD the target: the strand tangent
            # is unsigned (T = -T), so the sign of the TARGET's own screen
            # tangent against the source->target pixel direction decides
            # which way to walk (the target's depth was inpainted, so its
            # screen tangent is well-defined even without a real sample).
            toward_target = t_screen[..., 0] * (-dx) + t_screen[..., 1] * (-dy)
            step_sign = np.where(toward_target >= 0, 1.0, -1.0)
            oriented_step = tangent_step * step_sign[..., None]
            for ch in range(3):
                repair[..., ch] = np.where(take, repair[..., ch] + oriented_step[..., ch],
                                           repair[..., ch])
            best_dist = np.where(take, dist, best_dist)
            has_repair = has_repair | take
            has_position = has_position | take
            took_any = True
        if not took_any:
            break
        repaired = np.where(has_repair[..., None], repair,
                            position_input).astype(np.float32)
        current = repaired.copy()

    # Stage V: forward voting (eqs. 13-15). Each valid pixel votes along
    # its own tangent; votes are accepted for both tangent signs.
    vote_position = np.zeros((height, width, 3, 8), dtype=np.float32)
    all_vote_vz = np.full((height, width, 8), np.inf, dtype=np.float64)
    for slot, (dy, dx) in enumerate(offsets):
        source_valid = _shift(valid.astype(np.float32), dy, dx, 0.0) > 0.5
        source_pos = np.stack([_shift(position_valid[..., 0], dy, dx, 0.0),
                               _shift(position_valid[..., 1], dy, dx, 0.0),
                               _shift(position_valid[..., 2], dy, dx, 0.0)], axis=-1)
        source_step = _shift(step, dy, dx, 0.0)
        source_tan = np.stack([_shift(tangent_hair[..., 0], dy, dx, 0.0),
                               _shift(tangent_hair[..., 1], dy, dx, 0.0),
                               _shift(tangent_hair[..., 2], dy, dx, 0.0)], axis=-1)
        source_t_screen = np.stack([_shift(t_screen[..., 0], dy, dx, 0.0),
                                    _shift(t_screen[..., 1], dy, dx, 0.0)], axis=-1)

        # Orient the vote along the source->recipient direction.
        toward_recipient = np.stack([source_t_screen[..., 0] * (-dx),
                                     source_t_screen[..., 1] * (-dy)],
                                    axis=-1).sum(-1)
        vote_sign = np.where(toward_recipient >= 0, 1.0, -1.0)
        vote = source_pos + source_tan * (source_step * vote_sign)[..., None]
        vote_vz = view_space_z(vote.astype(np.float64), view)
        direction = np.stack([-np.full((height, width), float(dx)),
                              -np.full((height, width), float(dy))], axis=-1)
        alignment = (source_t_screen * direction).sum(-1)

        # Alignment gate for both tangent signs + the depth guard.
        accepted = (source_valid & hair & (np.abs(alignment) >= COS_THETA_MAX)
                    & (np.abs(vote_vz - target_vz) <= depth_tolerance))
        vote_vz_slot = np.where(accepted, vote_vz, np.inf)
        for ch in range(3):
            vote_position[..., ch, slot] = np.where(accepted, vote[..., ch], 0.0)
        all_vote_vz[..., slot] = vote_vz_slot

    has_vote = np.isfinite(all_vote_vz).any(axis=-1)
    best_slot = np.argmin(all_vote_vz, axis=-1)   # frontmost: min view z
    frontmost = np.take_along_axis(vote_position, best_slot[..., None, None], axis=-1)[..., 0]

    # Composition: valid keeps its real sample; invalid takes backward
    # repair, else the frontmost vote. Anything else stays unfilled (no
    # geometric evidence) - reported via 'unfilled'.
    final = repaired.copy()
    still_missing = invalid & ~has_repair
    final = np.where(still_missing[..., None] & has_vote[..., None],
                     frontmost, final)
    unfilled = hair & ~valid & ~has_repair & ~has_vote

    return {
        "position": final.astype(np.float32),
        "has_repair": has_repair,
        "has_vote": has_vote,
        "unfilled": unfilled,
    }
