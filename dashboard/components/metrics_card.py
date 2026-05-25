import streamlit as st


def metrics_card(label, value, delta=None, help_text=None):
    col = st.columns(1)[0]
    with col:
        if delta:
            st.metric(label=label, value=value, delta=delta, help=help_text)
        else:
            st.metric(label=label, value=value, help=help_text)


def metric_row(cols, metrics_list):
    for i, (label, value, delta, help_text) in enumerate(metrics_list):
        with cols[i]:
            if delta is not None:
                st.metric(label=label, value=value, delta=delta, help=help_text)
            else:
                st.metric(label=label, value=value, help=help_text)
