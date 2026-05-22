"""
Generate 32x32 grayscale images of 3D shapes (sphere, cube, tetrahedron)
at random 3D rotations with directional lighting.

Output: sphere_mix.txt, cube_mix.txt, tetrahedron_mix.txt
Format: one image per line, 1024 comma-separated integers (0-255)
"""

import numpy as np
import argparse
import os
import sys

# ============================================================
# 3D GEOMETRY — vertices centred at origin
# ============================================================

CUBE_VERTICES = np.array([
    [-1, -1, -1],  # 0
    [-1, -1,  1],  # 1
    [-1,  1, -1],  # 2
    [-1,  1,  1],  # 3
    [ 1, -1, -1],  # 4
    [ 1, -1,  1],  # 5
    [ 1,  1, -1],  # 6
    [ 1,  1,  1],  # 7
], dtype=np.float64)

# Outward-facing faces (CCW from outside); normal points *out*.
CUBE_FACES = [
    [0, 1, 3, 2],  # left   (-x)
    [4, 6, 7, 5],  # right  (+x)
    [0, 4, 5, 1],  # bottom (-y)
    [2, 3, 7, 6],  # top    (+y)
    [0, 2, 6, 4],  # back   (-z)
    [1, 5, 7, 3],  # front  (+z)
]

# Regular tetrahedron
TET_VERTICES = np.array([
    [ 1,  1,  1],
    [ 1, -1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
], dtype=np.float64) * (1.0 / np.sqrt(3))  # normalise to unit-ish scale

TET_FACES = [
    [0, 2, 1],
    [0, 1, 3],
    [0, 3, 2],
    [1, 2, 3],
]

# ============================================================
# Rotation
# ============================================================

def random_rotation(rng: np.random.Generator):
    """Uniform random 3D rotation via orthonormalising a random 3x3 matrix."""
    m = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(m)
    d = np.sign(np.diag(r))
    q = q * d  # enforce det = +1
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


# ============================================================
# Scanline polygon fill (convex polygons)
# ============================================================

def fill_convex_polygon(canvas: np.ndarray, vertices: np.ndarray, value: float):
    """
    Fill a convex polygon into *canvas* (mutated in-place).
    vertices : (N, 2) — float image-space coordinates.
    value    : float intensity written into covered pixels.
    """
    H, W = canvas.shape
    n = len(vertices)
    if n < 3:
        return

    ys = vertices[:, 1]
    y_min = max(0, int(np.floor(ys.min())))
    y_max = min(H - 1, int(np.ceil(ys.max())))

    for y in range(y_min, y_max + 1):
        x_intersect = []
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n]

            if abs(y2 - y1) < 1e-10:
                continue  # horizontal edge — skip

            # Does the scanline cross this edge?
            if (y1 <= y < y2) or (y2 <= y < y1):
                t = (y - y1) / (y2 - y1)
                x = x1 + t * (x2 - x1)
                x_intersect.append(x)

        if len(x_intersect) < 2:
            continue

        x_intersect.sort()

        # fill spans between even-odd pairs
        for k in range(0, len(x_intersect) - 1, 2):
            x_start = int(np.clip(np.ceil(x_intersect[k]), 0, W - 1))
            x_end = int(np.clip(np.floor(x_intersect[k + 1]), 0, W - 1))
            if x_start <= x_end:
                canvas[y, x_start:x_end + 1] = value


# ============================================================
# Lighting
# ============================================================

def face_brightness(face_normal: np.ndarray, light_dir: np.ndarray,
                    ambient: float = 0.25) -> float:
    """Lambertian shading: ambient + diffuse."""
    n = face_normal / (np.linalg.norm(face_normal) + 1e-12)
    l = light_dir / (np.linalg.norm(light_dir) + 1e-12)
    diffuse = max(0.0, float(np.dot(n, l)))
    return ambient + (1.0 - ambient) * diffuse


# ============================================================
# Polyhedron renderer
# ============================================================

def render_polyhedron(vertices_3d: np.ndarray,
                      faces: list,
                      rotation: np.ndarray,
                      scale: float,
                      img_size: int,
                      light_dir: np.ndarray,
                      supersample: int = 3) -> np.ndarray:
    """
    Render a convex polyhedron.
    Returns uint8 array shape (img_size, img_size).
    """
    H = img_size * supersample
    canvas = np.zeros((H, H), dtype=np.float64)
    half = H / 2.0

    # rotate & project
    rotated = vertices_3d @ rotation.T                     # (V, 3)
    proj_2d = rotated[:, :2] * (scale * half) + half       # (V, 2)

    # Camera at +z looking toward -z  →  face is front-facing if rotated_normal.z > 0
    for face_ix in faces:
        v0, v1, v2 = rotated[face_ix[0]], rotated[face_ix[1]], rotated[face_ix[2]]
        normal = np.cross(v1 - v0, v2 - v0)

        if normal[2] <= 0:   # back-face — cull
            continue

        brightness = face_brightness(normal, light_dir)
        val = brightness * 255.0

        face_verts = proj_2d[face_ix]
        fill_convex_polygon(canvas, face_verts, val)

    # downsample
    img = canvas.reshape(img_size, supersample, img_size, supersample).mean(axis=(1, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


# ============================================================
# Sphere renderer
# ============================================================

def render_sphere(img_size: int, scale: float, light_dir: np.ndarray,
                  supersample: int = 3) -> np.ndarray:
    """
    Render a sphere (always a circle in orthographic projection)
    with Lambertian shading.
    """
    H = img_size * supersample
    half = H / 2.0
    radius = scale * half

    l = light_dir / np.linalg.norm(light_dir)
    ambient = 0.25

    # coordinate grids
    yv, xv = np.mgrid[0:H, 0:H]
    dx = xv + 0.5 - half
    dy = yv + 0.5 - half
    d2 = dx * dx + dy * dy

    mask = d2 <= radius * radius
    canvas = np.zeros((H, H), dtype=np.float64)

    # surface normals for the visible hemisphere
    dz = np.sqrt(np.maximum(0, radius * radius - d2))
    nx = dx / radius
    ny = dy / radius
    nz = dz / radius

    diffuse = np.maximum(0, nx * l[0] + ny * l[1] + nz * l[2])
    shade = ambient + (1.0 - ambient) * diffuse
    canvas[mask] = shade[mask] * 255.0

    # downsample
    img = canvas.reshape(img_size, supersample, img_size, supersample).mean(axis=(1, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


# ============================================================
# Dataset generation
# ============================================================

def apply_noise(img: np.ndarray, intensity: float, rng: np.random.Generator) -> np.ndarray:
    """Add uniform random noise scaled by intensity (0-1) to a uint8 image."""
    if intensity <= 0:
        return img
    noise = rng.uniform(-intensity * 255, intensity * 255, size=img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def apply_brightness_shift(img: np.ndarray, magnitude: float, rng: np.random.Generator) -> np.ndarray:
    """Apply a uniform grayscale offset to the whole image.
    magnitude: max absolute shift (0-255). A random offset in [-magnitude, magnitude]
               is chosen and added to every pixel, then clipped to [0,255].
    """
    if magnitude <= 0:
        return img
    shift = rng.uniform(-magnitude, magnitude)
    return np.clip(img.astype(np.float64) + shift, 0, 255).astype(np.uint8)


def generate_dataset(n_per_class: int = 1000,
                     img_size: int = 32,
                     output_dir: str = ".",
                     seed: int = 42,
                     supersample: int = 3,
                     noise_props: list = None,
                     shift_magnitude: float = 0.0,
                     shift_prob: float = 0.0):
    """Generate equal numbers of sphere, cube and tetrahedron images.

    noise_props: list of 11 floats (0.0-1.0) for noise levels 0%,10%,...,100%.
    shift_magnitude: max brightness offset (0-255). At 255 can fully invert.
    shift_prob: probability (0-1) that a given image gets shifted.
    """

    rng = np.random.default_rng(seed)

    if noise_props is None:
        noise_props = [1.0] + [0.0] * 10

    # normalise noise_props to sum to 1; compute exact counts per level
    total_p = sum(noise_props)
    if total_p <= 0:
        noise_props = [1.0] + [0.0] * 10
        total_p = 1.0
    norm_props = [p / total_p for p in noise_props]
    counts = [int(round(n_per_class * p)) for p in norm_props]
    # fix rounding so sum == n_per_class
    diff = n_per_class - sum(counts)
    for i in range(abs(diff)):
        idx = i % len(counts)
        counts[idx] += 1 if diff > 0 else -1

    configs = {
        "sphere":       {"type": "sphere",       "scale": 0.40},
        "cube":         {"type": "cube",          "scale": 0.38},
        "tetrahedron":  {"type": "tetrahedron",   "scale": 0.42},
    }

    for name, cfg in configs.items():
        filename = os.path.join(output_dir, f"{name}_mix.txt")
        lines = []

        for level in range(11):
            n = counts[level]
            if n <= 0:
                continue
            intensity = level / 10.0  # 0.0, 0.1, ..., 1.0
            for i in range(n):
                light = rng.normal(size=3)
                light = light / np.linalg.norm(light)

                if cfg["type"] == "sphere":
                    img = render_sphere(img_size, cfg["scale"], light, supersample)
                else:
                    R = random_rotation(rng)
                    verts = CUBE_VERTICES if cfg["type"] == "cube" else TET_VERTICES
                    faces = CUBE_FACES if cfg["type"] == "cube" else TET_FACES
                    img = render_polyhedron(verts, faces, R, cfg["scale"],
                                            img_size, light, supersample)

                img = apply_noise(img, intensity, rng)

                if shift_prob > 0 and rng.random() < shift_prob:
                    img = apply_brightness_shift(img, shift_magnitude, rng)

                line = ",".join(str(int(p)) for p in img.flatten())
                lines.append(line)

            if n > 0:
                print(f"  {name}: noise {int(intensity*100)}% -> {n} images")

        with open(filename, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

        print(f"Saved {n_per_class} images -> {filename}")

    print("Done.")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 32×32 grayscale images of 3D shapes at random rotations"
    )
    parser.add_argument("-n", "--n-per-class", type=int, default=1000,
                        help="Number of images per class (default: 1000)")
    parser.add_argument("-s", "--size", type=int, default=32,
                        help="Image size in pixels (default: 32)")
    parser.add_argument("-o", "--output-dir", type=str, default=".",
                        help="Output directory (default: .)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--supersample", type=int, default=3,
                        help="Supersampling factor for anti-aliasing (default: 3)")
    parser.add_argument("--noise-props", type=str, default=None,
                        help="Comma-separated 11 proportions for noise 0%%,10%%,...,100%% (e.g. 0.3,0.2,0.1,0.1,0.1,0.05,0.05,0.05,0.03,0.01,0.01)")
    parser.add_argument("--shift-magnitude", type=float, default=0.0,
                        help="Max brightness shift magnitude 0-255 (default: 0, off)")
    parser.add_argument("--shift-prob", type=float, default=0.0,
                        help="Probability of applying brightness shift 0-1 (default: 0)")
    args = parser.parse_args()

    noise_props = None
    if args.noise_props:
        noise_props = [float(x) for x in args.noise_props.split(",")]
        if len(noise_props) != 11:
            print(f"ERROR: --noise-props must have exactly 11 values, got {len(noise_props)}")
            sys.exit(1)

    generate_dataset(args.n_per_class, args.size, args.output_dir,
                     args.seed, args.supersample, noise_props,
                     args.shift_magnitude, args.shift_prob)
