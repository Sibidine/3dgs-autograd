"""
End-to-end 3D Gaussian Splatting trainer running entirely on the custom
Tensor autograd engine (tensor.py) + a hand-written Adam. L1 loss only.

Expects in the working directory:  points3D.bin, images.bin, cameras.bin,
and an images/ folder whose filenames match images.bin.

    python3 gs_train.py            # train on the real data
    python3 gs_train.py selftest   # sanity run on real geometry + synthetic target
                                   #   (no images/ folder needed)

This is a toy: the compositor is a Python loop over Gaussians, so keep the
Gaussian count and resolution small. It exists to prove the gradient flows
end to end on your own autograd, not to produce a finished reconstruction.
"""
import sys, struct, math
import numpy as np
from tensor import Tensor, Adam
from plyfile import PlyElement, PlyData
from PIL import Image

# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
N_GAUSSIANS = 1000       # subsample the COLMAP cloud to this many
ITERS       = 100
LONG_EDGE   = 96         # render long image edge (px); dominates speed + memory
                         #   (raise toward 128+ locally if you have the RAM; the whole
                         #    forward graph is retained until backward finishes)
SEED        = 42
C0 = 0.28209479177387814          # degree-0 SH normalization

# ======================================================================
# 1. COLMAP loaders  (verified earlier against 3dgs.py)
# ======================================================================
def _read(f, fmt):
    fmt = "<" + fmt
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

NUM_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}

def read_cameras(path):
    cams = {}
    with open(path, "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            cam_id, model_id, w, h = _read(f, "iiQQ")
            p = _read(f, "d" * NUM_PARAMS[model_id])
            if model_id == 1:  fx, fy, cx, cy = p
            else:              fx = fy = p[0]; cx, cy = p[1], p[2]
            cams[cam_id] = dict(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h)
    return cams

def read_images(path):
    imgs = []
    with open(path, "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            img_id, qw, qx, qy, qz, tx, ty, tz, cam_id = _read(f, "idddddddi")
            name = b""
            c = f.read(1)
            while c != b"\x00":
                name += c; c = f.read(1)
            n_pts = _read(f, "Q")[0]
            f.read(24 * n_pts)
            imgs.append(dict(qvec=np.array([qw, qx, qy, qz]),
                             tvec=np.array([tx, ty, tz]),
                             cam_id=cam_id, name=name.decode()))
    return imgs

def read_points3D(path):
    xyz, rgb = [], []
    with open(path, "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            pid, x, y, z, r, g, b, err = _read(f, "QdddBBBd")
            xyz.append((x, y, z)); rgb.append((r, g, b))
            f.read(8 * _read(f, "Q")[0])
    return np.array(xyz), np.array(rgb, dtype=np.float64) / 255.0

# ======================================================================
# 2. init helpers
# ======================================================================
def knn_mean_sq_dist(points, k=3):
    d2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    idx = np.argpartition(d2, k, axis=1)[:, :k]
    return np.take_along_axis(d2, idx, axis=1).mean(1)

def inverse_sigmoid(x):
    return math.log(x / (1 - x))

def quat_to_rotmat_np(q):                    # constant camera pose -> R
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),     1 - 2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),     2*(y*z+w*x),   1 - 2*(x*x+y*y)]])

class GaussianModel:
    def __init__(self, xyz, rgb):
        N = xyz.shape[0]
        dist2 = knn_mean_sq_dist(xyz, k=3)
        self.means      = Tensor(xyz)
        self.scales     = Tensor(np.repeat(np.log(np.sqrt(dist2))[:, None], 3, axis=1))
        q = np.zeros((N, 4)); q[:, 0] = 1.0
        self.quats      = Tensor(q)
        self.opacities  = Tensor(np.full(N, inverse_sigmoid(0.1)))
        self.colors     = Tensor(rgb.copy())      # direct RGB (view-independent; no SH)
    def parameters(self):
        return [self.means, self.scales, self.quats, self.opacities, self.colors]


# ======================================================================
# 4. verified pipeline stages (Tensor engine)
# ======================================================================
def col(t, j): return t[:, j]

def t_quat_to_R(q):
    n = ((q * q).sum(axis=1, keepdims=True)) ** 0.5
    q = q / n
    w, x, y, z = col(q, 0), col(q, 1), col(q, 2), col(q, 3)
    two = 2.0
    r = [1 - two*(y*y+z*z), two*(x*y-w*z),   two*(x*z+w*y),
         two*(x*y+w*z),     1 - two*(x*x+z*z), two*(y*z-w*x),
         two*(x*z-w*y),     two*(y*z+w*x),   1 - two*(x*x+y*y)]
    return Tensor.stack(r, axis=1).reshape(-1, 3, 3)

def t_build_covariance(scales, quats):
    R = t_quat_to_R(quats)
    s = scales.exp().reshape(-1, 1, 3)
    M = R * s
    return M @ M.swapaxes(-1, -2)

def t_project_to_pixels(mean_cam, fx, fy, cx, cy):
    x, y = col(mean_cam, 0), col(mean_cam, 1)
    z = col(mean_cam, 2).clamp(min=1e-6)
    return Tensor.stack([fx * (x / z) + cx, fy * (y / z) + cy], axis=1)

def t_project_covariance(mean_cam, Sigma3d, view_R, fx, fy):
    x, y = col(mean_cam, 0), col(mean_cam, 1)
    z = col(mean_cam, 2); zz = z * z
    Z = Tensor(np.zeros(mean_cam.data.shape[0]))
    row0 = Tensor.stack([fx / z, Z, -(fx * x) / zz], axis=1)
    row1 = Tensor.stack([Z, fy / z, -(fy * y) / zz], axis=1)
    J = Tensor.stack([row0, row1], axis=1)
    T = J @ Tensor(view_R)
    cov2d = T @ Sigma3d @ T.swapaxes(-1, -2)
    return cov2d + Tensor(np.array([[0.3, 0.0], [0.0, 0.3]]))

def t_cov2d_to_conic(cov2d):
    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    det = (a * c - b * b).clamp(min=1e-6)
    return Tensor.stack([c / det, (-b) / det, a / det], axis=1)

def t_alpha_field(mean2d, conic, opacity, px_x, px_y):
    dx = px_x - mean2d[0]; dy = px_y - mean2d[1]
    A, B, C = conic[0], conic[1], conic[2]
    power = Tensor(-0.5) * (A * dx**2 + C * dy**2) - B * dx * dy
    alpha = (opacity * power.exp()).clamp(max=0.99)
    mask = Tensor((power.data <= 0.0).astype(np.float64))
    return alpha * mask

def t_composite(means2d, conics, colors, opacities, px_x, px_y, H, W):
    image = Tensor(np.zeros((H, W, 3)))
    T = Tensor(np.ones((H, W)))
    for i in range(means2d.data.shape[0]):
        a = t_alpha_field(means2d[i], conics[i], opacities[i], px_x, px_y)
        weight = a * T
        image = image + weight.reshape(H, W, 1) * colors[i]
        T = T * (Tensor(1.0) - a)
    return image

# ======================================================================
# 5. camera preprocessing + render  (the previously-untested seam)
# ======================================================================
def build_view(im, cam, gt=None):
    R = quat_to_rotmat_np(im["qvec"])          # world -> camera (3x3)
    t = im["tvec"].astype(np.float64)
    scale = LONG_EDGE / max(cam["width"], cam["height"])
    Wpx, Hpx = round(cam["width"] * scale), round(cam["height"] * scale)
    return dict(R=R, t=t,
                fx=cam["fx"] * scale, fy=cam["fy"] * scale,
                cx=cam["cx"] * scale, cy=cam["cy"] * scale,
                H=Hpx, W=Wpx, gt=gt)

def _pixel_grid(H, W):
    gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return Tensor(gx.astype(np.float64)), Tensor(gy.astype(np.float64))

def render(model, view):
    R, t = view["R"], view["t"]
    fx, fy, cx, cy, H, W = view["fx"], view["fy"], view["cx"], view["cy"], view["H"], view["W"]

    means_cam = model.means @ Tensor(R.T) + Tensor(t)     # (N,3), differentiable in means
    vis = means_cam[:, 2].data > 0.2                      # near-plane cull (detached mask)

    # apply the SAME mask to every per-Gaussian tensor  <-- the culling-consistency fix
    means_cam = means_cam[vis]
    scales    = model.scales[vis]
    quats     = model.quats[vis]
    colors    = model.colors[vis]
    opac_raw  = model.opacities[vis]

    means2d = t_project_to_pixels(means_cam, fx, fy, cx, cy)
    Sigma3d = t_build_covariance(scales, quats)
    cov2d   = t_project_covariance(means_cam, Sigma3d, R, fx, fy)
    conics  = t_cov2d_to_conic(cov2d)
    opac    = opac_raw.sigmoid()
    colors  = colors.clamp(min=0.0)

    depth = means_cam[:, 2].data                          # detached; sort is non-differentiable
    order = np.argsort(depth)
    means2d, conics = means2d[order], conics[order]
    colors, opac    = colors[order], opac[order]

    px, py = _pixel_grid(H, W)
    return t_composite(means2d, conics, colors, opac, px, py, H, W)

# ======================================================================
# 6. training
# ======================================================================
def l1_loss(pred, gt):
    return abs(pred - gt).mean()

def subsample(xyz, rgb, n, seed=SEED):
    if n >= len(xyz): return xyz, rgb
    idx = np.random.default_rng(seed).choice(len(xyz), size=n, replace=False)
    return xyz[idx], rgb[idx]

def load_image(path, size):        # size = (W, H)

    img = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.float64) / 255.0



def save_ply(model, path):
    xyz     = model.means.data                        # .data replaces .detach().cpu().numpy()
    N       = len(xyz)
    normals = np.zeros_like(xyz)

    # our color is direct RGB; convert to the DC SH coefficient a standard 3DGS
    # viewer expects, so that its  C0 * f_dc + 0.5  reproduces your rgb
    f_dc    = (model.colors.data - 0.5) / C0          # (N,3)

    opac    = model.opacities.data.reshape(N, 1)      # raw logit  (viewer applies sigmoid)
    scale   = model.scales.data                       # raw log     (viewer applies exp)
    rot     = model.quats.data                        # raw quat    (viewer normalizes)

    fields  = ['x','y','z','nx','ny','nz']
    fields += [f'f_dc_{i}' for i in range(3)]          # degree 0: DC only, no f_rest
    fields += ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
    dtype   = [(f, 'f4') for f in fields]

    data = np.concatenate([xyz, normals, f_dc, opac, scale, rot], axis=1)
    elements = np.empty(N, dtype=dtype)
    elements[:] = list(map(tuple, data))
    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)


def train():
    xyz, rgb = read_points3D("points3D.bin")
    cams     = read_cameras("cameras.bin")
    images   = read_images("images.bin")
    xyz, rgb = subsample(xyz, rgb, N_GAUSSIANS)
    model    = GaussianModel(xyz, rgb)

    views = []
    for im in images:
        v = build_view(im, cams[im["cam_id"]])
        v["gt"] = Tensor(load_image(f"images/{im['name']}", (v["W"], v["H"])))
        views.append(v)

    opt = Adam(model.parameters(), lr=1e-2)
    rng = np.random.default_rng(SEED)
    for it in range(ITERS):
        view = views[rng.integers(len(views))]
        img  = render(model, view)
        loss = l1_loss(img, view["gt"])
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 5 == 0:
            print(f"iter {it:4d}   L1 {float(loss.data):.5f}")
    print("done")
    save_ply(model, "point_cloud.ply")

# ======================================================================
# 7. self-test: real geometry, synthetic target, no images/ folder needed
# ======================================================================
def selftest():
    xyz, rgb = read_points3D("points3D.bin")
    cams     = read_cameras("cameras.bin")
    images   = read_images("images.bin")
    xyz, rgb = subsample(xyz, rgb, 300)          # small for a quick check
    model    = GaussianModel(xyz, rgb)

    global LONG_EDGE
    LONG_EDGE = 48                               # tiny image for speed
    view = build_view(images[0], cams[images[0]["cam_id"]])

    img0 = render(model, view)
    print(f"render output: shape {img0.data.shape}  finite={np.isfinite(img0.data).all()}  "
          f"range [{img0.data.min():.3f}, {img0.data.max():.3f}]")

    target = Tensor(np.full(img0.data.shape, 0.5))   # gray target to overfit toward
    opt = Adam(model.parameters(), lr=2e-2)
    print("\noverfitting a gray target (loss must fall):")
    for it in range(12):
        img  = render(model, view)
        loss = l1_loss(img, target)
        opt.zero_grad(); loss.backward(); opt.step()
        gmax = max(np.abs(p.grad).max() for p in model.parameters())
        print(f"  iter {it:2d}   L1 {float(loss.data):.5f}   max|grad| {gmax:.2e}")
    



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        train()
