"""Predefined visual styles for rendering mazes."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class MazeStyle:
    name: str
    wall_color: str
    solution_color: str
    background_color: str
    wall_width: float = 2.5
    solution_width: float = 3.0


STYLES: Dict[str, MazeStyle] = {
    "classic": MazeStyle(
        name="classic",
        wall_color="#000000",
        solution_color="#c62828",
        background_color="#ffffff",
        wall_width=2.5,
        solution_width=3.0,
    ),
    "blueprint": MazeStyle(
        name="blueprint",
        wall_color="#0d47a1",
        solution_color="#ffeb3b",
        background_color="#e3f2fd",
        wall_width=2.0,
        solution_width=2.5,
    ),
    "night": MazeStyle(
        name="night",
        wall_color="#fafafa",
        solution_color="#80cbc4",
        background_color="#212121",
        wall_width=2.5,
        solution_width=3.2,
    ),
}


def get_style(name: str) -> MazeStyle:
    try:
        return STYLES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown style '{name}'. Available styles: {', '.join(sorted(STYLES))}"
        ) from exc
