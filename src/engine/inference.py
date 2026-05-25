import torch
from src.trace.tracer import RoutingTracer


class InferenceEngine:
    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.tracer = RoutingTracer()

    def run_sparse(self, input_ids):
        self.model.eval()
        self.tracer.reset()
        with torch.no_grad():
            logits, all_gates, all_indices, aux_loss = self.model(input_ids)
            for layer_idx, (gates, indices) in enumerate(zip(all_gates, all_indices)):
                for t_idx in range(indices.shape[1]):
                    token_indices = indices[:, t_idx, :]
                    token_gates = gates[:, t_idx, :]
                    for b in range(token_indices.shape[0]):
                        for e_idx in range(token_indices.shape[1]):
                            expert_id = token_indices[b, e_idx].item()
                            weight = token_gates[b, expert_id].item()
                            self.tracer.record(
                                layer=layer_idx,
                                token_pos=t_idx,
                                expert_id=expert_id,
                                weight=weight,
                            )
        return logits, self.tracer

    def run_dense(self, input_ids):
        self.model.eval()
        with torch.no_grad():
            logits = input_ids.new_zeros(input_ids.shape[0], input_ids.shape[1], self.model.config.vocab_size)
            for block in self.model.blocks:
                moe = block.moe
                x = moe.forward_dense(input_ids.float())
            logits = self.model.lm_head(self.model.norm(input_ids.float()))
        return logits

    def generate(self, input_ids, max_new_tokens=30, temperature=1.0):
        self.model.eval()
        self.tracer.reset()
        generated = input_ids.clone()
        with torch.no_grad():
            for step in range(max_new_tokens):
                logits, all_gates, all_indices, aux_loss = self.model(generated[:, -self.model.config.max_seq_len:])
                for layer_idx, (gates, indices) in enumerate(zip(all_gates, all_indices)):
                    last_pos = indices.shape[1] - 1
                    token_indices = indices[:, last_pos, :]
                    token_gates = gates[:, last_pos, :]
                    for b in range(token_indices.shape[0]):
                        for e_idx in range(token_indices.shape[1]):
                            expert_id = token_indices[b, e_idx].item()
                            weight = token_gates[b, expert_id].item()
                            self.tracer.record(
                                layer=layer_idx,
                                token_pos=generated.shape[1] - 1,
                                expert_id=expert_id,
                                weight=weight,
                                step=step,
                            )
                next_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
        return generated, self.tracer
