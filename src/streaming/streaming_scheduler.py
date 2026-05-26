from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import torch
from pathlib import Path


class StreamingScheduler:
    def __init__(self, shard_manager, num_workers=2, prefetch_depth=1, devices=None):
        self.shard_manager = shard_manager
        self.num_workers = num_workers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.prefetch_depth = prefetch_depth
        self._futures = {}
        self._pinned_cache = {}
        self._devices = devices if devices else ["cpu"]
        self._resolve_key = lambda l, e: (l, e)
        self._device_assignments = {}
        self._rr_index = 0
        self.metrics = {
            "disk_read_ms": [],
            "pin_memory_ms": [],
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def set_key_resolver(self, resolver):
        self._resolve_key = resolver

    def set_device_assignments(self, mapping):
        self._device_assignments = mapping

    def _canonical_key(self, layer_idx, expert_idx):
        return self._resolve_key(layer_idx, expert_idx)

    def _assign_device(self, layer_idx, expert_idx):
        ckey = self._canonical_key(layer_idx, expert_idx)
        if ckey in self._device_assignments:
            did = self._device_assignments[ckey]
            return self._devices[did % len(self._devices)]
        did = self._rr_index % len(self._devices)
        self._rr_index += 1
        return self._devices[did]

    def prefetch_expert(self, layer_idx, expert_idx, device=None):
        ckey = self._canonical_key(layer_idx, expert_idx)
        if ckey in self._pinned_cache:
            self.metrics["cache_hits"] += 1
            return
        target = device or self._assign_device(layer_idx, expert_idx)
        future = self.executor.submit(self._load_and_pin, layer_idx, expert_idx, target)
        self._futures[ckey] = future

    def _load_and_pin(self, layer_idx, expert_idx, device):
        sl, se = self._canonical_key(layer_idx, expert_idx)
        t0 = time.perf_counter()
        state_dict = self.shard_manager.load_expert_shard(sl, se, device="cpu")
        disk_ms = (time.perf_counter() - t0) * 1000
        self.metrics["disk_read_ms"].append(disk_ms)
        t1 = time.perf_counter()
        if torch.cuda.is_available() and device != "cpu":
            for k in state_dict:
                state_dict[k] = state_dict[k].pin_memory()
        pin_ms = (time.perf_counter() - t1) * 1000
        self.metrics["pin_memory_ms"].append(pin_ms)
        self.metrics["cache_misses"] += 1
        return (layer_idx, expert_idx, state_dict)

    def get_expert(self, layer_idx, expert_idx, timeout=None):
        ckey = self._canonical_key(layer_idx, expert_idx)
        if ckey in self._pinned_cache:
            state_dict = self._pinned_cache.pop(ckey)
            return state_dict
        if ckey in self._futures:
            future = self._futures.pop(ckey)
            _, _, state_dict = future.result(timeout=timeout)
            return state_dict
        sl, se = self._canonical_key(layer_idx, expert_idx)
        t0 = time.perf_counter()
        state_dict = self.shard_manager.load_expert_shard(sl, se, device="cpu")
        disk_ms = (time.perf_counter() - t0) * 1000
        self.metrics["disk_read_ms"].append(disk_ms)
        self.metrics["cache_misses"] += 1
        return state_dict

    def resolve_experts(self, layer_idx, expert_ids, next_expert_ids=None):
        for eid in expert_ids:
            ckey = self._canonical_key(layer_idx, eid)
            if ckey not in self._futures and ckey not in self._pinned_cache:
                self.metrics["cache_misses"] += 1
                self.prefetch_expert(layer_idx, eid)
        if next_expert_ids is not None:
            for eid in next_expert_ids:
                self.prefetch_expert(layer_idx + 1, eid)
        state_dicts = {}
        for eid in expert_ids:
            state_dicts[eid] = self.get_expert(layer_idx, eid)
        return state_dicts

    def clear_cache(self):
        self._pinned_cache.clear()
        self._futures.clear()

    def reset_metrics(self):
        self.metrics = {
            "disk_read_ms": [],
            "pin_memory_ms": [],
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def cache_hit_rate(self):
        total = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        return self.metrics["cache_hits"] / total if total > 0 else 0.0

    def shutdown(self):
        self.executor.shutdown(wait=True)
