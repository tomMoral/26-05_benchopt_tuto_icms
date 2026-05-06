from benchopt import BasePlot


class Plot(BasePlot):
    """Display the estimated blur kernel for each solver."""

    name = "Image"
    type = "image"

    options = {
        "dataset": ...,
        "objective": ...,
    }

    def plot(self, df, dataset, objective):
        traces = []

        # Keep only the last row per solver (where final_results is stored)
        df_final = df.dropna(subset=["final_results"])

        x_true = df_final.iloc[0]["final_results"]["x_true"].squeeze()
        x_true = x_true.transpose(1, 2, 0)
        traces = [
            {"image": x_true, "label": "Ground truth"}
        ]

        for solver_name, group in df_final.groupby("solver_name"):
            row = group.iloc[-1]
            result = row["final_results"]

            if not isinstance(result, dict) or "x_hat" not in result:
                continue

            x_hat = result["x_hat"].squeeze()
            x_hat = x_hat.transpose(1, 2, 0)
            traces.append(
                {"image": x_hat, "label": solver_name}
            )

        return traces

    def get_metadata(self, df, dataset, objective):
        return {"title": "Estimated deblurred image"}
