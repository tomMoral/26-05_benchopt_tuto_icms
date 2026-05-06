import numpy as np
from benchopt import BasePlot


class Plot(BasePlot):
    """Display denoised images alongside the noisy input and ground truth."""

    name = "Denoised images"
    type = "image"

    options = {
        "dataset": ...,
        "objective": ...,
    }

    def plot(self, df, dataset, objective):
        traces = []

        rep_col = "idx_rep" if "idx_rep" in df.columns else None

        # Ground truth is constant across repetitions — take from first row
        first = df.iloc[0]
        if "objective_x_true" in df.columns:
            traces.append({
                "image": _to_display(first["objective_x_true"]),
                "label": "Ground truth",
            })

        # Noisy input varies per repetition (different noise seed)
        if "objective_y" in df.columns:
            if rep_col is not None:
                frames_y = [
                    _to_display(row["objective_y"])
                    for _, row in df.sort_values(rep_col)
                    .drop_duplicates(rep_col).iterrows()
                ]
            else:
                frames_y = [_to_display(first["objective_y"])]

            traces.append({
                "image": frames_y if len(frames_y) > 1 else frames_y[0],
                "label": "Noisy input",
            })

        # One entry per solver; collect all repetitions as GIF frames
        for solver_name, grp in df.groupby("solver_name"):
            if rep_col is not None:
                grp = grp.sort_values(rep_col)
            frames = [_to_display(row["objective_x_hat"])
                      for _, row in grp.iterrows()]
            psnr = grp["objective_psnr"].mean()
            traces.append({
                "image": frames if len(frames) > 1 else frames[0],
                "label": f"{solver_name}\n PSNR: {psnr:.2f} dB",
            })

        return traces

    def get_metadata(self, df, dataset, objective):
        return {"title": "Denoised images"}


def _to_display(tensor):
    """Convert a (1, C, H, W) or (C, H, W) tensor to a (H, W) or (H, W, C)
    numpy array in [0, 1] suitable for benchopt's image plot."""
    arr = np.asarray(tensor)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
    return np.clip(arr, 0, 1)
