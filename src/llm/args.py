from dataclasses import dataclass # Added import for dataclass
from pathlib import Path

@dataclass
class ARG:
    model: str
    data_source: str
    rating: str
    diversity: bool = False
    coverage: bool = False
    coherence: bool = False
    repetitive: bool = False
    readability: bool = False
    clustering_complementarity: bool = False

    def __repr__(self):
        flags = []
        if self.coverage: flags.append("coverage")
        if self.diversity: flags.append("diversity")
        if self.coherence: flags.append("coherence")
        if self.repetitive: flags.append("repetitive")
        if self.readability: flags.append("readability")
        if self.clustering_complementarity: flags.append("clustering_complementarity")
        # Handle case where none are selected (might happen if __post_init__ doesn't default)
        flag_str = "+".join(flags) if flags else "no_task"
        # Limit data_source path length in repr for clarity
        ds_repr = Path(self.data_source).name
        return f'{flag_str}+{self.model}+{ds_repr}+{self.rating}'
