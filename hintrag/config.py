import yaml


class Config(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = wrap(value)

    def __deepcopy__(self, memo):
        return wrap(to_dict(self))


def wrap(obj):
    if isinstance(obj, Config):
        return obj
    if isinstance(obj, dict):
        return Config({k: wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [wrap(v) for v in obj]
    return obj


def to_dict(obj):
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    return obj


def set_by_path(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = Config()
        node = node[key]
    node[keys[-1]] = wrap(value)


def load_config(path, overrides=None):
    with open(path, "r", encoding="utf-8") as f:
        cfg = wrap(yaml.safe_load(f))
    for item in overrides or []:
        key, value = item.split("=", 1)
        set_by_path(cfg, key.strip(), yaml.safe_load(value))
    return cfg


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False, allow_unicode=True)
