"""Core maze generation algorithms and data structures."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from collections import deque
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

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
        branching_chance: float = 0.35,
        cell_shape: str = "square",
        detour_bias: float = 0.0,
        turn_bias: float = 0.35,
        max_straight: int = 3,
        hairpin_chance: float = 0.25,
    ) -> None:
        if width <= 1 or height <= 1:
            raise ValueError("Maze dimensions must be greater than 1x1")

        self.width = width
        self.height = height
        self.random = random.Random(seed)
        self.loop_factor = max(0.0, min(1.0, loop_factor))
        self.cell_shape = cell_shape
        self.branching_chance = max(0.0, min(1.0, branching_chance))
        self.detour_bias = max(0.0, min(1.0, detour_bias))
        self.turn_bias = max(0.0, min(0.95, turn_bias))
        self.max_straight = max(0, max_straight)
        self.hairpin_chance = max(0.0, min(1.0, hairpin_chance))

        if cell_shape == "square":
            self._directions: Tuple[Direction, ...] = ("N", "S", "E", "W")
            self._opposites: Dict[Direction, Direction] = {
                "N": "S",
                "S": "N",
                "E": "W",
                "W": "E",
            }
            self._direction_angles: Dict[Direction, float] = {
                "N": 0.0,
                "E": 90.0,
                "S": 180.0,
                "W": 270.0,
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
            self._direction_angles = {
                "NE": 0.0,
                "E": 60.0,
                "SE": 120.0,
                "SW": 180.0,
                "W": 240.0,
                "NW": 300.0,
            }
        else:
            raise ValueError("cell_shape must be either 'square' or 'hex'")

        self.grid: List[List[Cell]] = [
            [Cell(x, y, {direction: True for direction in self._directions}) for x in range(width)]
            for y in range(height)
        ]

        self._goal = (self.width - 1, self.height - 1)
        if cell_shape == "hex":
            self._goal_axial = self._hex_offset_to_axial(*self._goal)
        else:
            self._goal_axial = None

        if cell_shape == "square":
            self._layout_origin = (0.0, 0.0)
            self._layout_width = float(self.width)
            self._layout_height = float(self.height)
        else:
            self._precompute_hex_layout()

        # 记录每个格子进入时的方向和当前直线长度，生成过程中会动态更新
        self._entry_direction: Dict[Tuple[int, int], Optional[Direction]] = {}
        self._straight_run: Dict[Tuple[int, int], int] = {}

    def _remove_wall(self, cell: Cell, neighbour: Cell, direction: Direction) -> None:
        opposite = self._opposites[direction]
        cell.walls[direction] = False
        neighbour.walls[opposite] = False

    def _can_carve_connection(
        self,
        cell: Cell,
        neighbour: Cell,
        direction: Direction,
        max_openings: int = 2,
    ) -> bool:
        """Return ``True`` when knocking down the wall keeps the maze tight."""

        if not cell.walls.get(direction, True):
            return False

        opposite = self._opposites[direction]
        neighbour_wall = neighbour.walls.get(opposite, True)
        if not neighbour_wall:
            return False

        return (
            self._openings(cell) < max_openings
            and self._openings(neighbour) < max_openings
        )

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
        """Generate the maze using a randomized growing-tree style algorithm."""

        active: List[Cell] = []
        start = self.cell(0, 0)
        visited = {(0, 0)}
        total_cells = self.width * self.height
        active.append(start)
        self._entry_direction[(0, 0)] = None
        self._straight_run[(0, 0)] = 0

        while active:
            if len(active) == 1:
                index = 0
            elif self.random.random() < self.branching_chance:
                index = self.random.randrange(len(active))
            else:
                index = len(active) - 1

            current = active[index]
            x, y = current.x, current.y
            unvisited_neighbors = [
                (direction, neighbour)
                for direction, neighbour in self.neighbours(x, y)
                if (neighbour.x, neighbour.y) not in visited
            ]

            if unvisited_neighbors:
                direction, next_cell = self._select_next_cell(
                    unvisited_neighbors,
                    len(visited),
                    total_cells,
                    self._entry_direction.get((x, y)),
                    self._straight_run.get((x, y), 0),
                )
                self._remove_wall(current, next_cell, direction)
                self._maybe_carve_extra_turn(
                    current,
                    self._entry_direction.get((x, y)),
                    visited,
                    blocked={(next_cell.x, next_cell.y)},
                )
                active.append(next_cell)
                visited.add((next_cell.x, next_cell.y))
                previous_direction = self._entry_direction.get((x, y))
                if previous_direction is not None and direction == previous_direction:
                    straight_length = self._straight_run.get((x, y), 0) + 1
                else:
                    straight_length = 1
                self._entry_direction[(next_cell.x, next_cell.y)] = direction
                self._straight_run[(next_cell.x, next_cell.y)] = straight_length
                self._maybe_carve_extra_turn(
                    next_cell,
                    direction,
                    visited,
                    blocked={(current.x, current.y)},
                )
            else:
                active.pop(index)

        self._add_loops()

        # Open an entrance and an exit for clarity when rendering
        if self.cell_shape == "square":
            self.cell(0, 0).walls["N"] = False
            self.cell(self.width - 1, self.height - 1).walls["S"] = False
        else:
            self.cell(0, 0).walls["W"] = False
            self.cell(self.width - 1, self.height - 1).walls["E"] = False

    def _select_next_cell(
        self,
        options: Sequence[Tuple[Direction, Cell]],
        visited_count: int,
        total_cells: int,
        previous_direction: Optional[Direction],
        straight_run: int,
    ) -> Tuple[Direction, Cell]:
        if not options:
            raise ValueError("_select_next_cell requires at least one option")

        directionally_weighted: List[Tuple[Tuple[Direction, Cell], float]] = []
        for option in options:
            direction, _ = option
            weight = 1.0
            if previous_direction is None:
                weight *= 1.0
            elif direction == previous_direction:
                if self.max_straight > 0 and straight_run >= self.max_straight:
                    # 直线过长时强制尝试转向
                    continue
                penalty = 1.0 - self.turn_bias
                if straight_run > 1:
                    penalty -= min(0.3, 0.1 * (straight_run - 1))
                weight *= max(0.05, penalty)
            else:
                angle_bonus = self._angular_difference(previous_direction, direction)
                weight *= 1.0 + self.turn_bias + (angle_bonus / 180.0) * self.turn_bias
            weight *= self.random.uniform(0.8, 1.2)
            if weight > 0.0:
                directionally_weighted.append((option, weight))

        if not directionally_weighted:
            # 如果所有方向都被限制，则退化为原始的可选方向
            directionally_weighted = [(option, 1.0) for option in options]

        if self.detour_bias <= 0.0:
            return self._weighted_choice(directionally_weighted)

        progress = visited_count / float(total_cells)
        bias_strength = self.detour_bias * max(0.0, 1.0 - progress)
        if bias_strength <= 0.0:
            return self._weighted_choice(directionally_weighted)

        weighted_options: List[Tuple[float, Tuple[Direction, Cell]]] = []
        base_options = [option for option, _ in directionally_weighted]
        base_weights = [weight for _, weight in directionally_weighted]

        distances = [self._distance_to_goal(cell.x, cell.y) for _, cell in base_options]
        max_distance = max(distances)
        min_distance = min(distances)

        if math.isclose(max_distance, min_distance):
            return self._weighted_choice(list(zip(base_options, base_weights)))

        for option, distance, direction_weight in zip(base_options, distances, base_weights):
            weight = direction_weight * (1.0 + bias_strength * (distance - min_distance))
            weighted_options.append((max(weight, 0.0), option))

        weighted_pairs = [(option, weight) for weight, option in weighted_options]
        return self._weighted_choice(weighted_pairs)

    def _weighted_choice(self, options: Sequence[Tuple[Tuple[Direction, Cell], float]]) -> Tuple[Direction, Cell]:
        total_weight = sum(weight for _, weight in options)
        if total_weight <= 0.0:
            return self.random.choice([option for option, _ in options])

        pick = self.random.random() * total_weight
        cumulative = 0.0
        for option, weight in options:
            cumulative += weight
            if pick <= cumulative:
                return option
        return options[-1][0]

    def _angular_difference(self, direction_a: Direction, direction_b: Direction) -> float:
        angle_a = self._direction_angles.get(direction_a)
        angle_b = self._direction_angles.get(direction_b)
        if angle_a is None or angle_b is None:
            return 0.0
        diff = abs(angle_a - angle_b) % 360.0
        return diff if diff <= 180.0 else 360.0 - diff

    def _coordinate_in_direction(
        self, x: int, y: int, direction: Direction
    ) -> Optional[Tuple[int, int]]:
        for neighbour_direction, neighbour in self.neighbours(x, y):
            if neighbour_direction == direction:
                return neighbour.x, neighbour.y
        return None

    def _maybe_carve_extra_turn(
        self,
        cell: Cell,
        incoming_direction: Optional[Direction],
        visited: Set[Tuple[int, int]],
        blocked: Optional[Set[Tuple[int, int]]] = None,
    ) -> None:
        if self.hairpin_chance <= 0.0:
            return
        if self.random.random() > self.hairpin_chance:
            return

        # Avoid opening up already spacious areas. Only consider carving an
        # extra turn when the current cell still resembles a corridor (i.e. it
        # has at most two existing exits). Otherwise repeatedly carving
        # hairpins can produce large rooms with few internal walls, which makes
        # the maze feel sparse.
        if self._openings(cell) >= 3:
            return

        blocked_coords: Set[Tuple[int, int]] = set(blocked or set())
        parent_coord: Optional[Tuple[int, int]] = None
        if incoming_direction is not None:
            parent_direction = self._opposites[incoming_direction]
            parent_coord = self._coordinate_in_direction(cell.x, cell.y, parent_direction)

        candidates: List[Tuple[Tuple[Direction, Cell], float]] = []
        for direction, neighbour in self.neighbours(cell.x, cell.y):
            coord = (neighbour.x, neighbour.y)
            if coord not in visited:
                continue
            if coord == parent_coord or coord in blocked_coords:
                continue
            if not cell.walls.get(direction, True):
                continue
            if self._openings(neighbour) >= 3:
                continue
            if incoming_direction is not None:
                angle_diff = self._angular_difference(incoming_direction, direction)
                weight = 0.5 + (angle_diff / 180.0) * 1.5
            else:
                weight = 1.0
            weight *= self.random.uniform(0.85, 1.15)
            candidates.append(((direction, neighbour), weight))

        if not candidates:
            return

        direction, neighbour = self._weighted_choice(candidates)
        if self._can_carve_connection(cell, neighbour, direction):
            self._remove_wall(cell, neighbour, direction)

    def _openings(self, cell: Cell) -> int:
        """Return the number of open sides for ``cell``."""

        return sum(1 for direction in self._directions if not cell.walls.get(direction, True))

    def _distance_to_goal(self, x: int, y: int) -> float:
        gx, gy = self._goal
        if self.cell_shape == "square":
            return abs(gx - x) + abs(gy - y)

        if self._goal_axial is None:
            return abs(gx - x) + abs(gy - y)

        q1, r1 = self._hex_offset_to_axial(x, y)
        q2, r2 = self._goal_axial
        return max(abs(q1 - q2), abs(r1 - r2), abs((-q1 - r1) - (-q2 - r2)))

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
            if cell.walls[direction] and self._can_carve_connection(cell, neighbour, direction):
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
        "easy": {
            "width": 12,
            "height": 12,
            "loop_factor": 0.05,
            "branching_chance": 0.25,
            "detour_bias": 0.25,
            "max_cells": 12 * 12,
        },
        "medium": {
            "width": 20,
            "height": 20,
            "loop_factor": 0.15,
            "branching_chance": 0.4,
            "detour_bias": 0.45,
            "max_cells": 22 * 22,
        },
        "hard": {
            "width": 28,
            "height": 28,
            "loop_factor": 0.25,
            "branching_chance": 0.55,
            "detour_bias": 0.6,
            "max_cells": None,
        },
    }

    turn_profiles = {
        "easy": {
            "turn_bias": 0.6,
            "max_straight": 3,
            "hairpin_chance": 0.2,
        },
        "medium": {
            "turn_bias": 0.68,
            "max_straight": 2,
            "hairpin_chance": 0.33,
        },
        "hard": {
            "turn_bias": 0.76,
            "max_straight": 1,
            "hairpin_chance": 0.42,
        },
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
    branching_chance = profile["branching_chance"]

    detour_bias = profile["detour_bias"]
    turn_profile = turn_profiles[difficulty]
    turn_bias = turn_profile["turn_bias"]
    max_straight = turn_profile["max_straight"]
    hairpin_chance = turn_profile["hairpin_chance"]

    for data in difficulty_profiles.values():
        max_cells = data["max_cells"]
        if max_cells is None or cell_count <= max_cells:
            loop_factor = data["loop_factor"]
            branching_chance = data["branching_chance"]
            detour_bias = data["detour_bias"]
            break
    else:  # pragma: no cover - logically unreachable because "hard" has max_cells=None
        loop_factor = profile["loop_factor"]

    maze = Maze(
        width=maze_width,
        height=maze_height,
        seed=seed,
        loop_factor=loop_factor,
        branching_chance=branching_chance,
        cell_shape=cell_shape,
        detour_bias=detour_bias,
        turn_bias=turn_bias,
        max_straight=max_straight,
        hairpin_chance=hairpin_chance,
    )
    maze.generate()
    return maze
