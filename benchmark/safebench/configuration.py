"""Resolve versioned experiment overrides without duplicating base inputs."""
import json
from pathlib import Path


def load_config(path, seen=()):
    path = Path(path).resolve()
    if path in seen: raise ValueError('Cyclic configuration inheritance')
    override = json.loads(path.read_text())
    parent = override.pop('extends', None)
    base = load_config(path.parent / parent, (*seen, path)) if parent else {}
    def merge(left, right):
        result = dict(left)
        for key, value in right.items():
            result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
        return result
    return merge(base, override)
