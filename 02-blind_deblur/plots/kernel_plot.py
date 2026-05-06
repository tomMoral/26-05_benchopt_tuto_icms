from benchopt import BasePlot


class Plot(BasePlot):
    """Display the estimated blur kernel for each solver."""

    name = "Kernel"
    type = "image"

    options = {
        "dataset": ...,
        "objective": ...,
    }

    def normalize(self, k_arr):
        # Normalise to [0, 1] for display
        k_min, k_max = k_arr.min(), k_arr.max()
        if k_max > k_min:
            k_arr = (k_arr - k_min) / (k_max - k_min)
        return k_arr.clip(0, 1)

    def plot(self, df, dataset, objective):
        traces = []

        # Keep only the last row per solver (where final_results is stored)
        df_final = df.dropna(subset=["final_results"])

        k_true = df_final.iloc[0]["final_results"]["k_true"].squeeze()
        traces = [
            {"image": self.normalize(k_true), "label": "Ground truth"}
        ]

        for solver_name, group in df_final.groupby("solver_name"):
            row = group.iloc[-1]
            result = row["final_results"]

            if not isinstance(result, dict) or "k_hat" not in result:
                continue

            k_hat = result["k_hat"].squeeze()
            traces.append(
                {"image": self.normalize(k_hat), "label": solver_name}
            )

        return traces

    def get_metadata(self, df, dataset, objective):
        return {"title": "Estimated blur kernel"}
