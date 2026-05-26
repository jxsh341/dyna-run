import torch
import psutil


class MemoryController:
    def __init__(self, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self._resident = {}
        self._resident_timestamps = {}
        self._sharing_map = {}
        self._refcount = {}
        self._device_map = {}
        self.metrics = {
            "gpu_bytes_loaded": 0,
            "gpu_bytes_evicted": 0,
            "peak_gpu_mb": 0,
        }

    def set_sharing_map(self, mapping):
        self._sharing_map = mapping

    def set_device_map(self, mapping):
        self._device_map = mapping

    def _canonical_key(self, layer_idx, expert_idx):
        return self._sharing_map.get((layer_idx, expert_idx), (layer_idx, expert_idx))

    def _target_device(self, layer_idx, expert_idx):
        if self._device_map:
            key = self._canonical_key(layer_idx, expert_idx)
            did = self._device_map.get(key)
            if did is not None:
                return f"cuda:{did}" if torch.cuda.is_available() else "cpu"
        return self.device

    def load_to_gpu(self, state_dict, layer_idx, expert_idx):
        key = self._canonical_key(layer_idx, expert_idx)
        if key in self._resident:
            self._refcount[key] = self._refcount.get(key, 1) + 1
            return self._resident[key]
        target = self._target_device(layer_idx, expert_idx)
        moved = {}
        for k, tensor in state_dict.items():
            t = tensor.to(target, non_blocking=True)
            moved[k] = t
        self._resident[key] = moved
        self._refcount[key] = 1
        self._resident_timestamps[key] = torch.cuda.Event() if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            self._resident_timestamps[key].record()
        bytes_loaded = sum(t.numel() * t.element_size() for t in moved.values())
        self.metrics["gpu_bytes_loaded"] += bytes_loaded
        current_mb = self.gpu_memory_used_mb()
        if current_mb > self.metrics["peak_gpu_mb"]:
            self.metrics["peak_gpu_mb"] = current_mb
        return moved

    def evict_from_gpu(self, layer_idx, expert_idx):
        key = self._canonical_key(layer_idx, expert_idx)
        if key not in self._resident:
            return
        self._refcount[key] = self._refcount.get(key, 1) - 1
        if self._refcount[key] > 0:
            return
        bytes_evicted = sum(
            t.numel() * t.element_size()
            for t in self._resident[key].values()
        )
        self.metrics["gpu_bytes_evicted"] += bytes_evicted
        del self._resident[key]
        del self._resident_timestamps[key]
        del self._refcount[key]

    def is_resident(self, layer_idx, expert_idx):
        return self._canonical_key(layer_idx, expert_idx) in self._resident

    def get_expert_on_gpu(self, layer_idx, expert_idx):
        key = self._canonical_key(layer_idx, expert_idx)
        return self._resident.get(key)

    def evict_all(self):
        for key in list(self._resident.keys()):
            self._refcount[key] = 1
            self.evict_from_gpu(*key)
        self._refcount.clear()

    def gpu_memory_used_mb(self):
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device) / (1024 * 1024)
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def resident_summary(self):
        return {
            "n_experts_in_gpu": len(self._resident),
            "gpu_memory_mb": self.gpu_memory_used_mb(),
            "peak_gpu_mb": self.metrics["peak_gpu_mb"],
            "total_loaded_mb": self.metrics["gpu_bytes_loaded"] / (1024 * 1024),
            "total_evicted_mb": self.metrics["gpu_bytes_evicted"] / (1024 * 1024),
            "resident_experts": list(self._resident.keys()),
        }
