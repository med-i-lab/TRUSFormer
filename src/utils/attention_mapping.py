"""
Code for creating attention maps 
# TODO clean this code it is a mess
"""


def get_relevancy_matrix(attentions, score, layer=-1, device="cpu"):
    # compute gradients for classes

    for A in attentions:
        A.retain_grad()

    score.backward()

    # compute matrices A_bar

    A_bars = []
    import torch

    def to_A_bar(A):
        A_bar = A * A.grad
        A_bar = torch.nn.functional.relu(A_bar)
        A_bar = A_bar.mean(dim=1)  # head dim
        return A_bar

    A_bars = [to_A_bar(A) for A in attentions]

    # do relevancy prop
    n_tokens = A_bars[0].shape[-1]
    R_s = []
    R_s.append(torch.eye(n_tokens).to(device))

    for A_bar in A_bars:
        R = R_s[-1]
        R_next = R + torch.matmul(A_bar, R)
        R_s.append(R_next)

    # Relevancy at last level
    R_last = R_s[layer]

    # Relevancy of input tokens to last class token
    # (row 0 of relevancy matrix)
    # If using mean rather than class pooling, take mean of rows
    # instead

    return R_last


from data.exact.dataset import patch_view_to_core_out

from src.utils.attention_mapping import get_relevancy_matrix
from src.data.exact.transforms import TransformV3
from src.data.exact.core import Core
from skimage.transform import resize
from src.data.exact.preprocessing import to_bmode
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch


def add_patch(ax, pos, **kwargs):
    x1, x2, y1, y2 = pos
    h = x2 - x1
    w = y2 - y1
    new_x = y1
    new_y = 28 - x2

    ax.add_patch(Rectangle((new_x, new_y), h, w, **kwargs))


t = TransformV3()


def core_to_model_in(core_specifier):
    core = Core(core_specifier)
    return patch_view_to_core_out(
        core.get_patch_view(
            needle_region_only=True,
            prostate_region_only=True,
            needle_intersection_threshold=0.66,
        ),
        t,
    )


def get_bmode_for_core(core_specifier, size=(512, 512)):
    return resize(to_bmode(Core(core_specifier).rf), size)


def plot_with_relevancy_maps(
    bmode, pos, relevance, base_alpha=0.3, ax=None, thresh=0.4
):
    if ax is None:
        plt.figure()
        ax = plt.gca()

    from matplotlib.cm import get_cmap

    cm = get_cmap("viridis")

    ax.set_axis_off()
    ax.imshow(bmode, extent=(0, 46, 0, 28), cmap="gray")
    for i in range(len(pos)):
        rel = relevance[i]
        if rel < thresh:
            continue
        fc = cm(rel)
        fc = "yellow"
        alpha = base_alpha * rel
        add_patch(ax, pos[i], fc=fc, alpha=alpha)


BENIGN_REFERENCE_CORE = "UVA-0036_LML"
benign_patch, benign_pos = core_to_model_in(BENIGN_REFERENCE_CORE)
CANCER_REFERENCE_CORE = "UVA-0515_RMM"
cancer_patch, cancer_pos = core_to_model_in(CANCER_REFERENCE_CORE)


def attach_ref(view, cancer_or_benign="cancer"):
    ref_patch = cancer_patch if cancer_or_benign == "cancer" else benign_patch
    ref_pos = cancer_pos if cancer_or_benign == "cancer" else benign_pos
    patch, pos = view
    ind = np.arange(len(patch))
    new_patch = torch.concat([patch, ref_patch])
    new_pos = torch.concat([pos, ref_pos])

    return new_patch, new_pos, ind


def make_heatmap(
    model, threshold_for_score, spec, cls_, ax=None, base_alpha=0.4, modifier_factor=1.2
):
    view = core_to_model_in(spec)

    patch, pos = view
    out = model(patch.cuda(), pos[:, 0].cuda(), pos[:, 2].cuda())
    score = out["logits"].softmax(-1)[1] - threshold_for_score
    print("score: ", score)
    print(Core(spec).metadata)

    patch, pos, ind = attach_ref(view, "cancer" if cls_ == 1 else "benign")
    out = model(patch.cuda(), pos[:, 0].cuda(), pos[:, 2].cuda())
    R = (
        get_relevancy_matrix(out["attentions"], out["logits"][cls_], device="cuda")
        .detach()
        .cpu()
    )
    R = np.array(R[0].mean(0))
    R = (R - R.min()) / (R.max() - R.min())

    R = R[ind]
    pos = pos[ind]

    score = score.item()
    if cls_ == 1:
        base_alpha = base_alpha * (1 + score * modifier_factor)
    else:
        base_alpha = base_alpha * (1 - score * modifier_factor)

    plot_with_relevancy_maps(
        get_bmode_for_core(spec), pos, R, ax=ax, base_alpha=base_alpha
    )
