import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class BenchmarkVisualizer:
    def speed_comparison(self, benchmark_results):
        lengths = sorted(benchmark_results.keys())
        sparse_tps = [benchmark_results[L]["sparse"]["tokens_per_sec"] for L in lengths]
        dense_tps = [benchmark_results[L]["dense"]["tokens_per_sec"] for L in lengths]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lengths, y=sparse_tps, mode="lines+markers",
            name="Sparse (MoE)", line=dict(color="#636EFA", width=3),
        ))
        fig.add_trace(go.Scatter(
            x=lengths, y=dense_tps, mode="lines+markers",
            name="Dense", line=dict(color="#EF553B", width=3),
        ))
        fig.update_layout(
            title="Inference Speed: Sparse vs Dense",
            xaxis_title="Sequence Length",
            yaxis_title="Tokens / Second",
            height=400,
            hovermode="x unified",
        )
        return fig

    def latency_comparison(self, benchmark_results):
        lengths = sorted(benchmark_results.keys())
        sparse_lat = [benchmark_results[L]["sparse"]["mean_latency_ms"] for L in lengths]
        dense_lat = [benchmark_results[L]["dense"]["mean_latency_ms"] for L in lengths]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[str(L) for L in lengths],
            y=sparse_lat, name="Sparse (MoE)",
            marker_color="#636EFA",
        ))
        fig.add_trace(go.Bar(
            x=[str(L) for L in lengths],
            y=dense_lat, name="Dense",
            marker_color="#EF553B",
        ))
        fig.update_layout(
            title="Latency Comparison (ms)",
            xaxis_title="Sequence Length",
            yaxis_title="Mean Latency (ms)",
            barmode="group",
            height=400,
        )
        return fig

    def memory_comparison(self, benchmark_results):
        lengths = sorted(benchmark_results.keys())
        sparse_ram = [benchmark_results[L]["sparse"]["ram_mb"] for L in lengths]
        dense_ram = [benchmark_results[L]["dense"]["ram_mb"] for L in lengths]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lengths, y=sparse_ram, mode="lines+markers",
            name="Sparse (MoE)", line=dict(color="#636EFA", width=3),
        ))
        fig.add_trace(go.Scatter(
            x=lengths, y=dense_ram, mode="lines+markers",
            name="Dense", line=dict(color="#EF553B", width=3),
        ))
        fig.update_layout(
            title="Memory Usage: Sparse vs Dense",
            xaxis_title="Sequence Length",
            yaxis_title="RAM (MB)",
            height=400,
            hovermode="x unified",
        )
        return fig

    def speedup_chart(self, benchmark_results):
        lengths = sorted(benchmark_results.keys())
        speedups = [benchmark_results[L]["speedup"] for L in lengths]
        fig = px.bar(
            x=[str(L) for L in lengths], y=speedups,
            title="Speedup Factor (Dense Latency / Sparse Latency)",
            labels={"x": "Sequence Length", "y": "Speedup (x)"},
            color=speedups,
            color_continuous_scale="RdYlGn",
            text_auto=".2f",
        )
        fig.update_layout(height=350)
        return fig

    def summary_table(self, benchmark_results):
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=["Metric", "Sparse (MoE)", "Dense", "Improvement"],
                fill_color="paleturquoise",
                align="left",
            ),
            cells=dict(
                values=self._table_values(benchmark_results),
                align="left",
            )
        )])
        fig.update_layout(height=250)
        return fig

    def _table_values(self, benchmark_results):
        if not benchmark_results:
            return [[], [], [], []]
        L = max(benchmark_results.keys())
        b = benchmark_results[L]
        s, d = b["sparse"], b["dense"]
        speed_pct = (d["tokens_per_sec"] - s["tokens_per_sec"]) / d["tokens_per_sec"] * 100
        mem_pct = (1 - s["ram_mb"] / max(d["ram_mb"], 0.1)) * 100
        return [
            ["Speed (tok/s)", "Latency (ms)", "RAM (MB)", "Active Params"],
            [f"{s['tokens_per_sec']:.1f}", f"{s['mean_latency_ms']:.2f}", f"{s['ram_mb']:.1f}", f"{s['active_params']:,}"],
            [f"{d['tokens_per_sec']:.1f}", f"{d['mean_latency_ms']:.2f}", f"{d['ram_mb']:.1f}", f"{d['active_params']:,}"],
            [f"{speed_pct:+.1f}%", f"{-speed_pct:+.1f}%", f"{mem_pct:+.1f}%", f"{s['active_params']/d['active_params']*100:.1f}%"],
        ]

    def dashboard(self, benchmark_results):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Speed (tok/s)", "Latency (ms)", "Memory (MB)", "Speedup"),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]],
        )
        lengths = sorted(benchmark_results.keys())
        sparse_tps = [benchmark_results[L]["sparse"]["tokens_per_sec"] for L in lengths]
        dense_tps = [benchmark_results[L]["dense"]["tokens_per_sec"] for L in lengths]
        sparse_lat = [benchmark_results[L]["sparse"]["mean_latency_ms"] for L in lengths]
        dense_lat = [benchmark_results[L]["dense"]["mean_latency_ms"] for L in lengths]
        sparse_ram = [benchmark_results[L]["sparse"]["ram_mb"] for L in lengths]
        dense_ram = [benchmark_results[L]["dense"]["ram_mb"] for L in lengths]
        speedups = [benchmark_results[L]["speedup"] for L in lengths]
        fig.add_trace(go.Scatter(x=lengths, y=sparse_tps, mode="lines+markers", name="Sparse", line=dict(color="#636EFA")), row=1, col=1)
        fig.add_trace(go.Scatter(x=lengths, y=dense_tps, mode="lines+markers", name="Dense", line=dict(color="#EF553B")), row=1, col=1)
        fig.add_trace(go.Bar(x=[str(L) for L in lengths], y=sparse_lat, name="Sparse", marker_color="#636EFA"), row=1, col=2)
        fig.add_trace(go.Bar(x=[str(L) for L in lengths], y=dense_lat, name="Dense", marker_color="#EF553B"), row=1, col=2)
        fig.add_trace(go.Scatter(x=lengths, y=sparse_ram, mode="lines+markers", name="Sparse", line=dict(color="#636EFA")), row=2, col=1)
        fig.add_trace(go.Scatter(x=lengths, y=dense_ram, mode="lines+markers", name="Dense", line=dict(color="#EF553B")), row=2, col=1)
        fig.add_trace(go.Bar(x=[str(L) for L in lengths], y=speedups, name="Speedup", marker_color="green"), row=2, col=2)
        fig.update_layout(height=600, showlegend=False, title_text="Benchmark Dashboard")
        return fig
