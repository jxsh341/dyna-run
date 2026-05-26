import torch
from pathlib import Path
import numpy as np


class ShardManager:
    def __init__(self, shard_dir="data/experts"):
        self.shard_dir = Path(shard_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=True)

    def save_expert_shard(self, expert_state_dict, layer_idx, expert_idx):
        path = self.shard_dir / f"layer{layer_idx}_expert{expert_idx}.pt"
        torch.save(expert_state_dict, path)
        return path

    def save_shared_shard(self, shared_state_dict, name="shared"):
        path = self.shard_dir / f"{name}.pt"
        torch.save(shared_state_dict, path)
        return path

    def load_expert_shard(self, layer_idx, expert_idx, device="cpu", map_location=None):
        path = self.shard_dir / f"layer{layer_idx}_expert{expert_idx}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Expert shard not found: {path}")
        return torch.load(path, map_location=map_location or device, weights_only=True)

    def load_shared_shard(self, name="shared", device="cpu", map_location=None):
        path = self.shard_dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Shared shard not found: {path}")
        return torch.load(path, map_location=map_location or device, weights_only=True)

    def shard_exists(self, layer_idx, expert_idx):
        return (self.shard_dir / f"layer{layer_idx}_expert{expert_idx}.pt").exists()

    def shard_size_bytes(self, layer_idx, expert_idx):
        path = self.shard_dir / f"layer{layer_idx}_expert{expert_idx}.pt"
        return path.stat().st_size if path.exists() else 0

    def list_expert_shards(self, layer_idx=None):
        if layer_idx is not None:
            pattern = f"layer{layer_idx}_expert*.pt"
        else:
            pattern = "layer*_expert*.pt"
        return sorted(self.shard_dir.glob(pattern))

    def delete_expert_shard(self, layer_idx, expert_idx):
        path = self.shard_dir / f"layer{layer_idx}_expert{expert_idx}.pt"
        if path.exists():
            path.unlink()

    def shard_model(self, model, layer_names=None, shared_expert_map=None):
        saved = []
        expert_buffers = {}
        for name, param in model.named_parameters():
            if "experts.experts" in name:
                parts = name.split("experts.experts.")
                expert_part = parts[1]
                expert_tokens = expert_part.split(".")
                expert_idx = int(expert_tokens[0])
                relative_key = ".".join(expert_tokens[1:])
                layer_part = parts[0].strip(".")
                layer_tokens = layer_part.split(".")
                layer_idx = -1
                for i, t in enumerate(layer_tokens):
                    if t == "blocks" and i + 1 < len(layer_tokens):
                        try:
                            layer_idx = int(layer_tokens[i + 1])
                        except ValueError:
                            pass
                        break
                if layer_idx < 0:
                    layer_idx = 0
                key = (layer_idx, expert_idx)
                if shared_expert_map:
                    canonical = shared_expert_map.get(key, key)
                    if canonical != key:
                        continue
                if key not in expert_buffers:
                    expert_buffers[key] = {}
                expert_buffers[key][relative_key] = param.data.cpu().clone()
            elif "token_embedding" in name or "norm" in name or "lm_head" in name or "attn" in name:
                self.save_shared_shard({name: param.data.cpu().clone()}, name.replace(".", "_"))
                saved.append(("shared", name, str(self.shard_dir / f"{name.replace('.', '_')}.pt")))
        for (layer_idx, expert_idx), state_dict in expert_buffers.items():
            shard_path = self.save_expert_shard(state_dict, layer_idx, expert_idx)
            saved.append((layer_idx, expert_idx, str(shard_path)))
        if shared_expert_map:
            self._save_redirects(shared_expert_map)
        return saved

    def _save_redirects(self, shared_expert_map):
        redirected = {}
        for alias, source in shared_expert_map.items():
            if alias != source:
                group = redirected.get(source, [])
                group.append(alias)
                redirected[source] = group
        for source, aliases in redirected.items():
            meta_path = self.shard_dir / f"shared_{source[0]}_{source[1]}.meta"
            torch.save({"source": source, "aliases": aliases}, meta_path)

    def distribute_shards(self, n_devices):
        manifest = {}
        shards = self.list_expert_shards()
        for i, shard_path in enumerate(shards):
            fname = shard_path.name
            parts = fname.replace("layer", "").replace("expert", "").replace(".pt", "").split("_")
            layer_idx = int(parts[0])
            expert_idx = int(parts[1])
            device_id = i % n_devices
            manifest[(layer_idx, expert_idx)] = device_id
        meta_path = self.shard_dir / "device_manifest.pt"
        torch.save(manifest, meta_path)
        return manifest

    def load_device_manifest(self):
        meta_path = self.shard_dir / "device_manifest.pt"
        if not meta_path.exists():
            return {}
        return torch.load(meta_path, weights_only=True)

    def count_shards(self):
        return len(self.list_expert_shards())

    def total_shard_size_mb(self):
        total = sum(f.stat().st_size for f in self.shard_dir.glob("*.pt"))
        return total / (1024 * 1024)
