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

        # Take the last row per solver (run_once → only one row anyway)
        df_last = df.groupby("solver_name").last().reset_index()

        # Ground truth and noisy input are the same across solvers — use first
        first = df_last.iloc[0]
        if "objective_x_true" in df_last.columns:
            traces.append({
                "image": _to_display(first["objective_x_true"]),
                "label": "Ground truth",
            })
        if "objective_y" in df_last.columns:
            traces.append({
                "image": _to_display(first["objective_y"]),
                "label": "Noisy input",
            })

        # One entry per solver
        for _, row in df_last.iterrows():
            traces.append({
                "image": _to_display(row["objective_x_hat"]),
                "label": row["solver_name"],
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
