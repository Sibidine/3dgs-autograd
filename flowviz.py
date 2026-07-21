"""
Gradient-flow visualization for the Tensor autograd 3DGS trainer.

Two tools:
  1. render_one(model, view, i)  -> render ONE Gaussian through the whole
     pipeline, so its graph is isolated and every node's grad is unambiguous.
     draw_dot_grad(loss) then draws the operation graph annotated with |grad|
     (this is the "which operations does the gradient flow back through" graph).

  2. track_gaussians(...) -> record the gradient norm of chosen Gaussians every
     iteration over a full training run, and plot the curves.

Requires gs_train.py, tensor.py in the same folder. draw_dot_grad needs graphviz
(`pip install graphviz` + the system `dot`); the tracker plot needs matplotlib.
"""
import numpy as np
import train_gs as G
from tensor import Tensor


# ======================================================================
# 1. single-Gaussian render + operation-graph with gradients
# ======================================================================
def render_one(model, view, i):
    """Run Gaussian i alone through the full pipeline. Returns (image, stages)."""
    d = lambda a: Tensor(a[i:i+1].copy())
    means, scales = d(model.means.data), d(model.scales.data)
    quats = d(model.quats.data)
    opac_r = Tensor(model.opacities.data[i:i+1].copy())
    colors = d(model.colors.data)
    R, t = view["R"], view["t"]
    fx, fy, cx, cy, H, W = view["fx"], view["fy"], view["cx"], view["cy"], view["H"], view["W"]

    means_cam = means @ Tensor(R.T) + Tensor(t)
    means2d   = G.t_project_to_pixels(means_cam, fx, fy, cx, cy)
    Sigma3d   = G.t_build_covariance(scales, quats)
    cov2d     = G.t_project_covariance(means_cam, Sigma3d, R, fx, fy)
    conics    = G.t_cov2d_to_conic(cov2d)
    opac      = opac_r.sigmoid()
    colors    = colors.clamp(min=0.0)
    px, py    = G._pixel_grid(H, W)
    image     = G.t_composite(means2d, conics, colors, opac, px, py, H, W)
    stages = dict(means=means, means_cam=means_cam, means2d=means2d, Sigma3d=Sigma3d,
                  cov2d=cov2d, conics=conics, opac=opac, colors=colors, image=image)
    return image, stages


def draw_dot_grad(root, path="gaussian_grad_graph"):
    """Graphviz of the computation graph, each node coloured/labelled by |grad|.
    Use on the loss from a render_one() call (one Gaussian) so the graph is small."""
    from graphviz import Digraph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for c in v._prev:
                edges.add((c, v)); build(c)
    build(root)

    gmax = max((np.abs(n.grad).max() for n in nodes), default=1.0) or 1.0
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})
    for n in nodes:
        gnorm = float(np.linalg.norm(n.grad))
        inten = np.log10(np.abs(n.grad).max() + 1e-12) - np.log10(gmax)   # 0 (strong) .. negative
        shade = int(max(0, min(255, 255 + inten * 40)))                   # darker = stronger grad
        fill = f"#{shade:02x}{shade:02x}ff"
        op = n._op or "leaf"
        dot.node(str(id(n)), label=f"{op}\\n|g|={gnorm:.2e}",
                 shape="box", style="filled", fillcolor=fill,
                 fontcolor="white" if shade < 140 else "black")
    for a, b in edges:
        dot.edge(str(id(a)), str(id(b)))
    dot.render(path, cleanup=True)
    print(f"wrote {path}.svg")
    return dot


def stage_grads(model, view, i, target):
    """|grad| at each named stage for Gaussian i (the data behind the flow DAG)."""
    image, st = render_one(model, view, i)
    G.l1_loss(image, target).backward()
    return {name: float(np.linalg.norm(t.grad)) for name, t in st.items()}


# ======================================================================
# 2. track chosen Gaussians' gradient norm across a training run
# ======================================================================
def track_gaussians(model, view, ids, iters=100, lr=2e-2, target=None):
    if target is None:
        img0 = G.render(model, view)
        target = Tensor(np.full(img0.data.shape, 0.5))
    opt = G.Adam(model.parameters(), lr=lr)
    curves = {i: [] for i in ids}
    for _ in range(iters):
        loss = G.l1_loss(G.render(model, view), target)
        opt.zero_grad(); loss.backward()
        for i in ids:
            curves[i].append(float(np.linalg.norm(np.concatenate([
                model.means.grad[i], model.scales.grad[i], model.quats.grad[i],
                [model.opacities.grad[i]], model.colors.grad[i]]))))
        opt.step()
    return curves


def plot_curves(curves, path="gaussian_grad_curves.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, c in curves.items():
        ax.plot(c, label=f"g{i}", linewidth=1.4)
    ax.set_xlabel("iteration"); ax.set_ylabel("|grad| (L2 over params)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120)
    print("saved", path)


if __name__ == "__main__":
    xyz, rgb = G.read_points3D("points3D.bin")
    cams = G.read_cameras("cameras.bin"); images = G.read_images("images.bin")
    xyz, rgb = G.subsample(xyz, rgb, 1000)
    G.LONG_EDGE = 40
    view = G.build_view(images[0], cams[images[0]["cam_id"]])

    # (1) op-graph with gradients for one visible, contributing Gaussian
    model = G.GaussianModel(xyz, rgb)
    tgt = Tensor(np.full((view["H"], view["W"], 3), 0.5))
    mc = (model.means.data @ view["R"].T) + view["t"]
    visible = np.where(mc[:, 2] > 0.2)[0]
    gi = next((int(i) for i in visible                       # find one that covers pixels
               if stage_grads(model, view, int(i), tgt)["means"] > 0), int(visible[0]))
    image, _ = render_one(model, view, gi)
    loss = G.l1_loss(image, tgt)
    loss.backward()
    print(f"Gaussian #{gi}")
    draw_dot_grad(loss, "gaussian_grad_graph")
    print("stage grads:", {k: round(v, 6) for k, v in stage_grads(model, view, gi, tgt).items()})
    model = G.GaussianModel(xyz, rgb)
    ids = [min(i, len(xyz) - 1) for i in (0, 250, 500, 750, 1000)]
    curves = track_gaussians(model, view, ids, iters=100, target=tgt)
    plot_curves(curves)