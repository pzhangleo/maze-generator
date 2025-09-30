"""Core maze generation algorithms and data structures."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from collections import deque
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

Direction = str


@dataclass
class Cell:
    """A single cell inside the maze grid."""

    x: int
    y: int
    walls: Dict[Direction, bool]


class Maze:
    """A grid maze supporting both square and hexagonal topologies."""

    def __init__(
        self,
        width: int,
        height: int,
        seed: Optional[int] = None,
        loop_factor: float = 0.0,
        cell_shape: str = "square",
    ) -> None:
        if width <= 1 or height <= 1:
            raise ValueError("Maze dimensions must be greater than 1x1")

        self.width = width
        self.height = height
        self.random = random.Random(seed)
        self.loop_factor = max(0.0, min(1.0, loop_factor))
        self.cell_shape = cell_shape

        if cell_shape == "square":
            self._directions: Tuple[Direction, ...] = ("N", "S", "E", "W")
            self._opposites: Dict[Direction, Direction] = {
                "N": "S",
                "S": "N",
                "E": "W",
                "W": "E",
            }
        elif cell_shape == "hex":
            self._directions = ("NE", "E", "SE", "SW", "W", "NW")
            self._opposites = {
                "NE": "SW",
                "SW": "NE",
                "E": "W",
                "W": "E",
                "SE": "NW",
                "NW": "SE",
            }
            self._hex_direction_deltas: Dict[Direction, Tuple[int, int]] = {
                "NE": (1, -1),
                "E": (1, 0),
                "SE": (0, 1),
                "SW": (-1, 1),
                "W": (-1, 0),
                "NW": (0, -1),
            }
            self._hex_direction_corners: Dict[Direction, Tuple[int, int]] = {
                "NE": (0, 1),
                "E": (1, 2),
                "SE": (2, 3),
                "SW": (3, 4),
                "W": (4, 5),
                "NW": (5, 0),
            }
        else:
            raise ValueError("cell_shape must be either 'square' or 'hex'")

        self.grid: List[List[Cell]] = [
            [Cell(x, y, {direction: True for direction in self._directions}) for x in range(width)]
            for y in range(height)
        ]

        if cell_shape == "square":
            self._layout_origin = (0.0, 0.0)
            self._layout_width = float(self.width)
            self._layout_height = float(self.height)
        else:
            self._precompute_hex_layout()

    def _remove_wall(self, cell: Cell, neighbour: Cell, direction: Direction) -> None:
        opposite = self._opposites[direction]
        cell.walls[direction] = False
        neighbour.walls[opposite] = False

    def _neighbour_coords(self, x: int, y: int) -> Iterator[Tuple[Direction, Tuple[int, int]]]:
        if self.cell_shape == "square":
            deltas: Sequence[Tuple[Direction, Tuple[int, int]]] = (
                ("N", (0, -1)),
                ("S", (0, 1)),
                ("E", (1, 0)),
                ("W", (-1, 0)),
            )
            for direction, (dx, dy) in deltas:
                yield direction, (x + dx, y + dy)
            return

        q, r = self._hex_offset_to_axial(x, y)
        for direction in self._directions:
            dq, dr = self._hex_direction_deltas[direction]
            nq, nr = q + dq, r + dr
            nx, ny = self._hex_axial_to_offset(nq, nr)
            yield direction, (nx, ny)

    def cell(self, x: int, y: int) -> Cell:
        return self.grid[y][x]

    def neighbours(self, x: int, y: int) -> Iterable[Tuple[Direction, Cell]]:
        for direction, (nx, ny) in self._neighbour_coords(x, y):
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
                self._remove_wall(current, next_cell, direction)
                stack.append(next_cell)
                visited.add((next_cell.x, next_cell.y))
            else:
                stack.pop()

        self._add_loops()

        # Open an entrance and an exit for clarity when rendering
        if self.cell_shape == "square":
            self.cell(0, 0).walls["N"] = False
            self.cell(self.width - 1, self.height - 1).walls["S"] = False
        else:
            self.cell(0, 0).walls["W"] = False
            self.cell(self.width - 1, self.height - 1).walls["E"] = False

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
            neighbours = list(self.neighbours(x, y))
            if not neighbours:
                continue
            direction, neighbour = self.random.choice(neighbours)
            if cell.walls[direction]:
                self._remove_wall(cell, neighbour, direction)

    def ascii_render(self, wall_char: str = "#", path_char: str = " ") -> str:
        """Return a simple ASCII representation of the maze."""

        if self.cell_shape == "square":
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

        return self._ascii_render_hex(wall_char, path_char)

    def layout_size(self) -> Tuple[float, float]:
        return self._layout_width, self._layout_height

    def wall_segments(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        if self.cell_shape == "square":
            for y in range(self.height):
                for x in range(self.width):
                    cell = self.cell(x, y)
                    left, right = x, x + 1
                    top, bottom = y, y + 1
                    if cell.walls["N"]:
                        segments.append(((left, top), (right, top)))
                    if cell.walls["S"]:
                        segments.append(((left, bottom), (right, bottom)))
                    if cell.walls["W"]:
                        segments.append(((left, top), (left, bottom)))
                    if cell.walls["E"]:
                        segments.append(((right, top), (right, bottom)))
            return segments

        for y in range(self.height):
            for x in range(self.width):
                cell = self.cell(x, y)
                corners = self._hex_cell_corners(x, y)
                for direction in self._directions:
                    if cell.walls.get(direction, False):
                        i1, i2 = self._hex_direction_corners[direction]
                        segments.append((corners[i1], corners[i2]))
        return segments

    def path_points(self, path: Sequence[Tuple[int, int]]) -> List[Tuple[float, float]]:
        return [self._cell_center(x, y) for x, y in path]

    # --- Hex geometry helpers -------------------------------------------------

    def _precompute_hex_layout(self) -> None:
        self._hex_size = 1.0
        min_x = math.inf
        max_x = -math.inf
        min_y = math.inf
        max_y = -math.inf

        for y in range(self.height):
            for x in range(self.width):
                cx, cy = self._hex_cell_center_raw(x, y)
                for px, py in self._hex_polygon(cx, cy):
                    min_x = min(min_x, px)
                    max_x = max(max_x, px)
                    min_y = min(min_y, py)
                    max_y = max(max_y, py)

        if not math.isfinite(min_x) or not math.isfinite(max_x):
            min_x = min_y = 0.0
            max_x = max_y = 1.0

        self._layout_origin = (min_x, min_y)
        self._layout_width = max(max_x - min_x, 1.0)
        self._layout_height = max(max_y - min_y, 1.0)

    def _hex_cell_corners(self, x: int, y: int) -> List[Tuple[float, float]]:
        cx, cy = self._hex_cell_center_raw(x, y)
        ox, oy = self._layout_origin
        return [(px - ox, py - oy) for px, py in self._hex_polygon(cx, cy)]

    def _hex_cell_center_raw(self, x: int, y: int) -> Tuple[float, float]:
        q, r = self._hex_offset_to_axial(x, y)
        return self._hex_axial_to_pixel(q, r)

    def _hex_cell_center(self, x: int, y: int) -> Tuple[float, float]:
        cx, cy = self._hex_cell_center_raw(x, y)
        ox, oy = self._layout_origin
        return cx - ox, cy - oy

    def _cell_center(self, x: int, y: int) -> Tuple[float, float]:
        if self.cell_shape == "square":
            return x + 0.5, y + 0.5
        return self._hex_cell_center(x, y)

    def _hex_offset_to_axial(self, x: int, y: int) -> Tuple[int, int]:
        q = x
        r = y - (x // 2)
        return q, r

    def _hex_axial_to_offset(self, q: int, r: int) -> Tuple[int, int]:
        x = q
        y = r + (q // 2)
        return x, y

    def _hex_axial_to_pixel(self, q: int, r: int) -> Tuple[float, float]:
        size = self._hex_size
        x = size * math.sqrt(3) * (q + r / 2)
        y = size * 1.5 * r
        return x, y

    def _hex_polygon(self, cx: float, cy: float) -> List[Tuple[float, float]]:
        size = self._hex_size
        half_width = math.sqrt(3) * size / 2
        return [
            (cx, cy - size),
            (cx + half_width, cy - size / 2),
            (cx + half_width, cy + size / 2),
            (cx, cy + size),
            (cx - half_width, cy + size / 2),
            (cx - half_width, cy - size / 2),
        ]

    def _ascii_render_hex(self, wall_char: str, path_char: str) -> str:
        segments = self.wall_segments()
        width, height = self.layout_size()
        scale = 3
        padding = 2
        canvas_width = int(math.ceil(width * scale)) + padding * 2 + 1
        canvas_height = int(math.ceil(height * scale)) + padding * 2 + 1
        canvas: List[List[str]] = [
            [path_char for _ in range(canvas_width)] for _ in range(canvas_height)
        ]

        for (x1, y1), (x2, y2) in segments:
            self._draw_segment(canvas, x1, y1, x2, y2, wall_char, scale, padding)

        return "\n".join("".join(row).rstrip() for row in canvas).rstrip()

    def _draw_segment(
        self,
        canvas: List[List[str]],
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        char: str,
        scale: int,
        padding: int,
    ) -> None:
        ix1 = int(round(x1 * scale)) + padding
        iy1 = int(round(y1 * scale)) + padding
        ix2 = int(round(x2 * scale)) + padding
        iy2 = int(round(y2 * scale)) + padding

        dx = abs(ix2 - ix1)
        dy = -abs(iy2 - iy1)
        sx = 1 if ix1 < ix2 else -1
        sy = 1 if iy1 < iy2 else -1
        err = dx + dy

        while True:
            if 0 <= iy1 < len(canvas) and 0 <= ix1 < len(canvas[iy1]):
                canvas[iy1][ix1] = char
            if ix1 == ix2 and iy1 == iy2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                ix1 += sx
            if e2 <= dx:
                err += dx
                iy1 += sy

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
            for direction, (nx, ny) in self._neighbour_coords(x, y):
                if cell.walls.get(direction, True):
                    continue
                next_coord = (nx, ny)
                if next_coord not in came_from and 0 <= nx < self.width and 0 <= ny < self.height:
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
    cell_shape: str = "square",
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
        cell_shape=cell_shape,
    )
    maze.generate()
    return maze
