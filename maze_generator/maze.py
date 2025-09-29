"""Core maze generation algorithms and data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple

Direction = str

OPPOSITE_WALL: Dict[Direction, Direction] = {"N": "S", "S": "N", "E": "W", "W": "E"}
MOVE_DELTA: Dict[Direction, Tuple[int, int]] = {
    "N": (0, -1),
    "S": (0, 1),
    "E": (1, 0),
    "W": (-1, 0),
}


@dataclass
class Cell:
    """A single cell inside the maze grid."""

    x: int
    y: int
    walls: Dict[Direction, bool] = field(
        default_factory=lambda: {"N": True, "S": True, "E": True, "W": True}
    )

    def knock_down_wall(self, other: "Cell", direction: Direction) -> None:
        """Remove the wall between this cell and a neighbour cell."""

        self.walls[direction] = False
        other.walls[OPPOSITE_WALL[direction]] = False


class Maze:
    """An orthogonal grid maze supporting generation and solving."""

    def __init__(
        self,
        width: int,
        height: int,
        seed: Optional[int] = None,
        loop_factor: float = 0.0,
    ) -> None:
        if width <= 1 or height <= 1:
            raise ValueError("Maze dimensions must be greater than 1x1")

        self.width = width
        self.height = height
        self.random = random.Random(seed)
        self.loop_factor = max(0.0, min(1.0, loop_factor))
        self.grid: List[List[Cell]] = [
            [Cell(x, y) for x in range(width)] for y in range(height)
        ]

    def cell(self, x: int, y: int) -> Cell:
        return self.grid[y][x]

    def neighbours(self, x: int, y: int) -> Iterable[Tuple[Direction, Cell]]:
        for direction, (dx, dy) in MOVE_DELTA.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                yield direction, self.cell(nx, ny)

    def generate(self) -> None:
        """Generate the maze using a randomized depth-first search algorithm."""

        stack: List[Cell] = []
        current = self.cell(0, 0)
        visited = {(0, 0)}
        stack.append(current)

        while stack:
            current = stack[-1]
            x, y = current.x, current.y
            unvisited_neighbors = [
                (direction, neighbour)
                for direction, neighbour in self.neighbours(x, y)
                if (neighbour.x, neighbour.y) not in visited
            ]

            if unvisited_neighbors:
                direction, next_cell = self.random.choice(unvisited_neighbors)
                current.knock_down_wall(next_cell, direction)
                stack.append(next_cell)
                visited.add((next_cell.x, next_cell.y))
            else:
                stack.pop()

        self._add_loops()

        # Open an entrance and an exit for clarity when rendering
        self.cell(0, 0).walls["N"] = False
        self.cell(self.width - 1, self.height - 1).walls["S"] = False

    def _add_loops(self) -> None:
        """Optionally remove additional walls to increase complexity."""

        if self.loop_factor <= 0:
            return

        total_cells = self.width * self.height
        attempts = int(total_cells * self.loop_factor)

        for _ in range(attempts):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            cell = self.cell(x, y)
            direction, neighbour = self.random.choice(list(self.neighbours(x, y)))
            if cell.walls[direction]:
                cell.knock_down_wall(neighbour, direction)

    def ascii_render(self, wall_char: str = "#", path_char: str = " ") -> str:
        """Return a simple ASCII representation of the maze."""

        top_border = [wall_char] * (self.width * 2 + 1)
        if not self.cell(0, 0).walls["N"]:
            top_border[1] = path_char
        lines = ["".join(top_border)]
        for y in range(self.height):
            top = [wall_char]
            bottom = [wall_char]
            for x in range(self.width):
                cell = self.cell(x, y)
                top.append(path_char)
                top.append(wall_char if cell.walls["E"] else path_char)

                bottom.append(wall_char if cell.walls["S"] else path_char)
                bottom.append(wall_char)
            lines.append("".join(top))
            lines.append("".join(bottom))
        if not self.cell(self.width - 1, self.height - 1).walls["S"]:
            last = list(lines[-1])
            last[-2] = path_char
            lines[-1] = "".join(last)
        return "\n".join(lines)

    def solve(self) -> List[Tuple[int, int]]:
        """Find the shortest path from the top-left to the bottom-right cell."""

        start = (0, 0)
        goal = (self.width - 1, self.height - 1)
        queue: Deque[Tuple[int, int]] = deque([start])
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

        while queue:
            current = queue.popleft()
            if current == goal:
                break
            x, y = current
            cell = self.cell(x, y)
            for direction, (dx, dy) in MOVE_DELTA.items():
                if cell.walls[direction]:
                    continue
                nx, ny = x + dx, y + dy
                next_coord = (nx, ny)
                if next_coord not in came_from:
                    queue.append(next_coord)
                    came_from[next_coord] = current

        if goal not in came_from:
            raise RuntimeError("Maze has no solution")

        # Reconstruct path
        path: List[Tuple[int, int]] = []
        current: Optional[Tuple[int, int]] = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path


def generate_maze(
    difficulty: str = "medium",
    width: Optional[int] = None,
    height: Optional[int] = None,
    seed: Optional[int] = None,
) -> Maze:
    """Factory function that creates and returns a generated maze."""

    difficulty_profiles = {
        "easy": {"width": 12, "height": 12, "loop_factor": 0.05, "max_cells": 12 * 12},
        "medium": {"width": 20, "height": 20, "loop_factor": 0.15, "max_cells": 22 * 22},
        "hard": {"width": 28, "height": 28, "loop_factor": 0.25, "max_cells": None},
    }

    profile = difficulty_profiles.get(difficulty)
    if profile is None:
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. Available options: {', '.join(difficulty_profiles)}"
        )

    maze_width = width or profile["width"]
    maze_height = height or profile["height"]

    cell_count = maze_width * maze_height
    # Determine loop density dynamically based on the final maze size so that
    # custom dimensions still produce an appropriate difficulty level.
    for data in difficulty_profiles.values():
        max_cells = data["max_cells"]
        if max_cells is None or cell_count <= max_cells:
            loop_factor = data["loop_factor"]
            break
    else:  # pragma: no cover - logically unreachable because "hard" has max_cells=None
        loop_factor = profile["loop_factor"]

    maze = Maze(
        width=maze_width,
        height=maze_height,
        seed=seed,
        loop_factor=loop_factor,
    )
    maze.generate()
    return maze
