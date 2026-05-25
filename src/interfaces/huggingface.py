class HuggingFaceClient:
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_name}: {e}")

    def generate(self, prompt, max_length=100):
        if self.model is None:
            self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs, max_length=max_length,
            do_sample=True, temperature=0.7,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def probe_activations(self, prompt):
        if self.model is None:
            self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model(**inputs, output_hidden_states=True)
        return {
            "n_layers": len(outputs.hidden_states),
            "n_tokens": inputs["input_ids"].shape[1],
            "hidden_states_shape": outputs.hidden_states[-1].shape,
        }
