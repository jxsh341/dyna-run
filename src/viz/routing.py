import plotly.graph_objects as go
import plotly.express as px
import numpy as np


class RoutingVisualizer:
    def sankey_diagram(self, tracer, n_layers, n_experts):
        df = tracer.to_dataframe()
        if df.empty:
            return go.Figure()
        labels = []
        for l in range(n_layers):
            labels.append(f"Layer {l}")
        for e in range(n_experts):
            labels.append(f"Expert {e}")
        source, target, value = [], [], []
        colors = px.colors.qualitative.Set2
        for l in range(n_layers):
            layer_df = df[df["layer"] == l]
            for e in range(n_experts):
                count = len(layer_df[layer_df["expert_id"] == e])
                if count > 0:
                    source.append(l)
                    target.append(n_layers + e)
                    value.append(count)
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15, thickness=20,
                label=labels,
                color=[colors[i % len(colors)] for i in range(len(labels))],
            ),
            link=dict(
                source=source, target=target, value=value,
            )
        )])
        fig.update_layout(title="Token-to-Expert Routing Paths", height=500)
        return fig

    def token_routing_flow(self, tracer, token_texts, n_layers, n_experts):
        df = tracer.to_dataframe()
        if df.empty or not token_texts:
            return go.Figure()
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for e in range(n_experts):
            expert_df = df[df["expert_id"] == e]
            if expert_df.empty:
                continue
            tokens_per_layer = expert_df.groupby("layer").size()
            layers = list(range(n_layers))
            counts = [tokens_per_layer.get(l, 0) for l in layers]
            fig.add_trace(go.Scatter(
                x=layers, y=counts,
                mode="lines+markers",
                name=f"Expert {e}",
                line=dict(color=colors[e % len(colors)]),
                stackgroup="one",
            ))
        fig.update_layout(
            title="Expert Activation Across Layers",
            xaxis_title="Layer",
            yaxis_title="Tokens Routed",
            height=400,
            hovermode="x unified",
        )
        return fig

    def routing_weight_distribution(self, tracer):
        df = tracer.to_dataframe()
        if df.empty:
            return go.Figure()
        fig = px.histogram(
            df, x="weight", nbins=30,
            title="Routing Weight Distribution",
            labels={"weight": "Gate Weight", "count": "Frequency"},
            color_discrete_sequence=["#636EFA"],
        )
        fig.update_layout(height=300, showlegend=False)
        return fig
