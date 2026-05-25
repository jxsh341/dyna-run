import streamlit as st
import torch
from src.engine.profiler import Profiler
from src.viz.benchmark import BenchmarkVisualizer


def show(moe_model, dense_model, model_config, device, benchmark_results):
    st.header("Benchmark Dashboard")
    st.markdown("Compare inference speed, memory, and activation patterns between **sparse (MoE)** and **dense** models.")

    col1, col2 = st.columns(2)
    with col1:
        n_runs = st.slider("Measurement runs", min_value=3, max_value=30, value=10)
    with col2:
        seq_lengths = st.multiselect(
            "Sequence lengths to benchmark",
            options=[32, 64, 128, 256, 512],
            default=[64, 128, 256],
        )

    if st.button("Run Benchmark", type="primary") and seq_lengths and moe_model is not None:
        profiler = Profiler()
        results = profiler.benchmark_sequence_lengths(
            moe_model, dense_model, seq_lengths,
            device=device, n_runs=n_runs,
        )
        benchmark_results.clear()
        benchmark_results.update(results)
        st.success("Benchmark complete!")

    if benchmark_results:
        bviz = BenchmarkVisualizer()
        tab1, tab2 = st.tabs(["Charts", "Summary Table"])

        with tab1:
            st.plotly_chart(bviz.speed_comparison(benchmark_results), width='stretch')
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(bviz.latency_comparison(benchmark_results), width='stretch')
            with col2:
                st.plotly_chart(bviz.memory_comparison(benchmark_results), width='stretch')
            st.plotly_chart(bviz.speedup_chart(benchmark_results), width='stretch')
            st.plotly_chart(bviz.dashboard(benchmark_results), width='stretch')

        with tab2:
            st.plotly_chart(bviz.summary_table(benchmark_results), width='stretch')
            for length, data in sorted(benchmark_results.items()):
                with st.expander(f"Sequence Length = {length}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Sparse (MoE)")
                        s = data["sparse"]
                        st.json({
                            "Speed (tok/s)": f"{s['tokens_per_sec']:.1f}",
                            "Latency (ms)": f"{s['mean_latency_ms']:.2f} ± {s['std_latency_ms']:.2f}",
                            "RAM (MB)": f"{s['ram_mb']:.1f}",
                            "Active Params": f"{s['active_params']:,} ({s['active_params_ratio']*100:.0f}%)",
                        })
                    with col2:
                        st.subheader("Dense")
                        d = data["dense"]
                        st.json({
                            "Speed (tok/s)": f"{d['tokens_per_sec']:.1f}",
                            "Latency (ms)": f"{d['mean_latency_ms']:.2f} ± {d['std_latency_ms']:.2f}",
                            "RAM (MB)": f"{d['ram_mb']:.1f}",
                            "Active Params": f"{d['active_params']:,} ({d['active_params_ratio']*100:.0f}%)",
                        })
                    st.metric("Speedup", f"{data['speedup']:.2f}x")
                    st.metric("Memory Savings", f"{data['memory_savings_pct']:.1f}%")
    else:
        st.info("Select sequence lengths and click 'Run Benchmark' to compare performance.")
