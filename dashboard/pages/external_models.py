import streamlit as st


def show():
    st.header("External Model Integration")
    st.markdown(
        "Try routing visualization on models from external sources. "
        "These integrations are partial — they probe model internals where possible."
    )

    integration_type = st.radio(
        "Select integration",
        ["Ollama", "HuggingFace", "llama.cpp"],
        horizontal=True,
    )

    if integration_type == "Ollama":
        _ollama_section()
    elif integration_type == "HuggingFace":
        _huggingface_section()
    elif integration_type == "llama.cpp":
        _llamacpp_section()


def _ollama_section():
    st.subheader("Ollama Integration")
    st.markdown(
        "Query local Ollama models and probe their layer activations. "
        "Requires [Ollama](https://ollama.ai) running locally."
    )

    model_name = st.text_input("Ollama model name", value="llama3.2:1b")
    prompt = st.text_area("Prompt", value="Why is MoE efficient?", height=100)

    if st.button("Query Ollama", type="primary"):
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": False},
                timeout=30,
            )
            if response.status_code == 200:
                result = response.json()
                st.success(f"Response received ({len(result.get('response', ''))} chars)")
                st.text_area("Model Output", result.get("response", ""), height=200)
                with st.expander("Raw Response"):
                    st.json(result)
            else:
                body = response.text[:500]
                st.error(f"Ollama error {response.status_code}: {body}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to Ollama. Make sure it's running on localhost:11434.")
        except Exception as e:
            st.error(f"Error: {e}")


def _huggingface_section():
    st.subheader("HuggingFace Integration")
    st.markdown(
        "Load a HuggingFace transformer model and apply routing probes. "
        "Install with: `pip install transformers`"
    )

    model_name = st.text_input("HuggingFace model ID", value="gpt2")
    prompt = st.text_area("Prompt", value="The future of AI is", height=100)

    if st.button("Load & Probe", type="primary"):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            with st.spinner(f"Loading {model_name}..."):
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                inputs = tokenizer(prompt, return_tensors="pt")
                outputs = model(**inputs, output_attentions=True)
                tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                st.success(f"Model loaded. {len(tokens)} tokens processed.")
                st.text_area("Tokens", ", ".join(tokens), height=80)
                st.info("Note: Full routing visualization for HuggingFace models requires custom adapter layers.")
        except ImportError:
            st.error("transformers not installed. Run: pip install transformers")
        except Exception as e:
            st.error(f"Error: {e}")


def _llamacpp_section():
    st.subheader("llama.cpp Integration")
    st.markdown(
        "Run inference via llama.cpp. Requires llama.cpp compiled and accessible. "
        "Install with: `pip install llama-cpp-python`"
    )

    model_path = st.text_input("GGUF model path", value="models/llama-2-7b.Q4_K_M.gguf")
    prompt = st.text_area("Prompt", value="What is sparse MoE?", height=100)

    if st.button("Run llama.cpp", type="primary"):
        try:
            from llama_cpp import Llama
            with st.spinner(f"Loading model from {model_path}..."):
                llm = Llama(model_path=model_path, n_ctx=512, verbose=False)
                output = llm(prompt, max_tokens=50, echo=False)
                text = output["choices"][0]["text"]
                st.success(f"Generated {len(text.split())} tokens")
                st.text_area("Generated Text", text, height=200)
        except ImportError:
            st.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
        except Exception as e:
            st.error(f"Error: {e}")
