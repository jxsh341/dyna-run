import plotly.graph_objects as go
import plotly.express as px
import numpy as np


class HeatmapVisualizer:
    def expert_activation_heatmap(self, tracer, n_layers, n_experts):
        mat = tracer.expert_activation_matrix(n_layers, n_experts)
        fig = px.imshow(
            mat,
            labels=dict(x="Expert ID", y="Layer", color="Activations"),
            x=[f"E{e}" for e in range(n_experts)],
            y=[f"L{l}" for l in range(n_layers)],
            title="Expert Activation Heatmap",
            color_continuous_scale="Viridis",
            aspect="auto",
        )
        fig.update_layout(height=400)
        return fig

    def per_token_heatmap(self, tracer, n_layers, n_tokens, n_experts):
        df = tracer.to_dataframe()
        if df.empty:
            return go.Figure()
        mat = np.zeros((n_tokens, n_layers))
        for _, row in df.iterrows():
            tpos = int(row["token_pos"])
            lay = int(row["layer"])
            if tpos < n_tokens and lay < n_layers:
                mat[tpos, lay] += 1
        fig = px.imshow(
            mat.T,
            labels=dict(x="Token Position", y="Layer", color="Experts Used"),
            title="Experts Used per Token per Layer",
            color_continuous_scale="Plasma",
            aspect="auto",
        )
        fig.update_layout(height=400)
        return fig

    def expert_utilization_bar(self, tracer, n_experts):
        util = tracer.expert_utilization(n_experts)
        fig = px.bar(
            x=[f"Expert {e}" for e in range(n_experts)],
            y=[util.get(e, 0) * 100 for e in range(n_experts)],
            title="Expert Utilization (%)",
            labels={"x": "", "y": "Utilization (%)"},
            color=[util.get(e, 0) * 100 for e in range(n_experts)],
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            height=350,
            showlegend=False,
            xaxis={"tickangle": -45},
        )
        return fig
