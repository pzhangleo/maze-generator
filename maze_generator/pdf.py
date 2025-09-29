"""PDF export helpers for mazes."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .maze import Maze
from .render import export_maze_pdf as _export


def export_maze_pdf(
    maze: Maze,
    pdf_path: Path,
    style_name: str = "classic",
    metadata: Optional[str] = None,
) -> Path:
    """Export the maze and its solution to a PDF file."""

    return _export(maze=maze, style_name=style_name, pdf_path=pdf_path, metadata=metadata)
