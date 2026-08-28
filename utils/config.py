from typing import Any, NotRequired, TypedDict

import yaml


class ParamsConfig(TypedDict):
    Nx: int
    Ny: int
    Lx: float
    Ly: float
    dt: float
    nt: int
    nt_save: int
    eta: float
    nu: float


class InitialConditionsConfig(TypedDict):
    name: str


class Config(TypedDict):
    params: ParamsConfig
    initial_conditions: InitialConditionsConfig


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg.get("params") is None:
        raise ValueError("Params section not found in config")
    elif cfg.get("initial_conditions") is None:
        raise ValueError("Initial conditions section not found in config")

    return cfg
