class LlamaCppClient:
    def __init__(self, model_path, n_ctx=512, verbose=False):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.verbose = verbose
        self.model = None

    def load(self):
        try:
            from llama_cpp import Llama
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                verbose=self.verbose,
            )
            return True
        except ImportError:
            raise RuntimeError("llama-cpp-python not installed")
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_path}: {e}")

    def generate(self, prompt, max_tokens=100, temperature=0.7):
        if self.model is None:
            self.load()
        output = self.model(
            prompt, max_tokens=max_tokens,
            temperature=temperature, echo=False,
        )
        return output["choices"][0]["text"]

    def is_loaded(self):
        return self.model is not None
