"""
Funções utilitárias gerais, sem domínio específico.

Exemplos do que pode conter:
    - funções de log
    - decorators (ex: medir tempo de execução)
    - funções de resumo de DataFrames
    - helpers de I/O
"""

import logging
import yaml
from pathlib import Path

def load_config(caminho: str = "configs/config.yaml") -> dict:
    with open(caminho, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    return logging.getLogger(name)