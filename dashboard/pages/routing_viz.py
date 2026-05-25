import streamlit as st
import torch
import pandas as pd
from src.engine.inference import InferenceEngine
from src.trace.metrics import RoutingMetrics
from src.viz.routing import RoutingVisualizer
from src.viz.heatmap import HeatmapVisualizer
from dashboard.components.metrics_card import metrics_card, metric_row


def show(moe_model, model_config, device):
    st.header("Routing Visualization")
    st.markdown("Enter text to see how the MoE router dispatches tokens to different experts.")

    input_text = st.text_area(
        "Input text",
        value="To be or not to be that is the question",
        height=100,
    )

    run_btn = st.button("Run Routing", type="primary")

    if run_btn and input_text and moe_model is not None:
        moe_model.eval()
        chars = sorted(set(input_text))
        stoi = {ch: i for i, ch in enumerate(chars[:model_config.vocab_size])}
        input_ids_list = [stoi.get(c, 0) for c in input_text]
        if len(input_ids_list) < 2:
            st.warning("Need at least 2 characters")
            return
        input_ids = torch.tensor([input_ids_list], device=device)

        engine = InferenceEngine(moe_model)
        logits, tracer = engine.run_sparse(input_ids)
        metrics = RoutingMetrics(tracer)
        n_layers = model_config.n_layers
        n_experts = model_config.moe.n_experts
        n_tokens = input_ids.shape[1]

        col1, col2, col3, col4 = st.columns(4)
        sm = metrics.summary(n_experts)
        col1.metric("Total Routing Decisions", sm["total_routing_decisions"])
        col2.metric("Unique Experts Used", sm["unique_experts_used"])
        col3.metric("Load Balancing", f"{sm['load_balancing']:.2f}")
        col4.metric("Routing Entropy", f"{sm['routing_entropy']:.2f}")

        rviz = RoutingVisualizer()
        hviz = HeatmapVisualizer()

        tab1, tab2, tab3, tab4 = st.tabs(["Sankey Diagram", "Token Paths", "Heatmap", "Expert Usage"])

        with tab1:
            st.plotly_chart(
                rviz.sankey_diagram(tracer, n_layers, n_experts),
                width='stretch',
            )

        with tab2:
            st.plotly_chart(
                rviz.token_routing_flow(tracer, list(input_text), n_layers, n_experts),
                width='stretch',
            )
            st.plotly_chart(
                rviz.routing_weight_distribution(tracer),
                width='stretch',
            )

        with tab3:
            st.plotly_chart(
                hviz.expert_activation_heatmap(tracer, n_layers, n_experts),
                width='stretch',
            )
            st.plotly_chart(
                hviz.per_token_heatmap(tracer, n_layers, n_tokens, n_experts),
                width='stretch',
            )

        with tab4:
            st.plotly_chart(
                hviz.expert_utilization_bar(tracer, n_experts),
                width='stretch',
            )
            usage_df = tracer.to_dataframe()
            if not usage_df.empty:
                pivot = usage_df.pivot_table(
                    index="layer", columns="expert_id",
                    aggfunc="size", fill_value=0,
                )
                st.dataframe(pivot, width='stretch')

        with st.expander("Raw Routing Trace"):
            st.dataframe(tracer.to_dataframe(), width='stretch')

    elif not input_text:
        st.info("Enter some text and click 'Run Routing' to visualize expert activation.")
