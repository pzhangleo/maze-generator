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
        branching_chance: float = 0.35,
        cell_shape: str = "square",
        detour_bias: float = 0.0,
        turn_bias: float = 0.35,
        max_straight: int = 3,
        hairpin_chance: float = 0.25,
        winding: float = 0.6,
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
        self.winding = max(0.0, min(1.0, winding))

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
        """Generate the maze in three passes.

        1. Carve a spanning tree with the randomized growing-tree algorithm.
        2. Stretch the start-to-goal route so the answer winds through the maze
           instead of cutting across it in a near-straight diagonal.
        3. Braid dead ends with loops that cannot shortcut that route.
        """

        self._carve_spanning_tree()
        # 生成本身只会保证「连通」，起点到终点的通路仍然可能几乎是条直线。
        # 这里在保持生成树结构的前提下重连分支，把解答路径拉长成蜿蜒形态。
        if self.winding > 0.0:
            self._elongate_solution()
        self._add_loops()

        # Open an entrance and an exit for clarity when rendering
        if self.cell_shape == "square":
            self.cell(0, 0).walls["N"] = False
            self.cell(self.width - 1, self.height - 1).walls["S"] = False
        else:
            self.cell(0, 0).walls["W"] = False
            self.cell(self.width - 1, self.height - 1).walls["E"] = False

    def _carve_spanning_tree(self) -> None:
        """Carve a spanning tree with the randomized growing-tree algorithm."""

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
                active.append(next_cell)
                visited.add((next_cell.x, next_cell.y))
                previous_direction = self._entry_direction.get((x, y))
                if previous_direction is not None and direction == previous_direction:
                    straight_length = self._straight_run.get((x, y), 0) + 1
                else:
                    straight_length = 1
                self._entry_direction[(next_cell.x, next_cell.y)] = direction
                self._straight_run[(next_cell.x, next_cell.y)] = straight_length
            else:
                active.pop(index)

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

    # --- Solution path shaping -------------------------------------------------

    def _direction_between(
        self, origin: Tuple[int, int], target: Tuple[int, int]
    ) -> Optional[Direction]:
        """Return the direction that steps from ``origin`` to ``target``."""

        for direction, (nx, ny) in self._neighbour_coords(*origin):
            if (nx, ny) == target:
                return direction
        return None

    def _set_wall(self, origin: Tuple[int, int], target: Tuple[int, int], closed: bool) -> None:
        direction = self._direction_between(origin, target)
        if direction is None:
            raise ValueError(f"{origin} and {target} are not adjacent cells")
        ox, oy = origin
        tx, ty = target
        self.grid[oy][ox].walls[direction] = closed
        self.grid[ty][tx].walls[self._opposites[direction]] = closed

    def _flood_from(self, origin: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
        """Breadth-first distances from ``origin`` across currently open walls."""

        distances: Dict[Tuple[int, int], int] = {origin: 0}
        queue: Deque[Tuple[int, int]] = deque([origin])
        while queue:
            current = queue.popleft()
            x, y = current
            step = distances[current] + 1
            cell = self.cell(x, y)
            for direction, (nx, ny) in self._neighbour_coords(x, y):
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if cell.walls.get(direction, True):
                    continue
                nxt = (nx, ny)
                if nxt in distances:
                    continue
                distances[nxt] = step
                queue.append(nxt)
        return distances

    def _shortest_path(self) -> List[Tuple[int, int]]:
        """Shortest start-to-goal route across the open walls."""

        start = (0, 0)
        goal = (self.width - 1, self.height - 1)
        parents: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        queue: Deque[Tuple[int, int]] = deque([start])

        while queue:
            current = queue.popleft()
            if current == goal:
                break
            x, y = current
            cell = self.cell(x, y)
            for direction, (nx, ny) in self._neighbour_coords(x, y):
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if cell.walls.get(direction, True):
                    continue
                nxt = (nx, ny)
                if nxt in parents:
                    continue
                parents[nxt] = current
                queue.append(nxt)

        if goal not in parents:
            raise RuntimeError("Maze has no solution")

        path: List[Tuple[int, int]] = []
        node: Optional[Tuple[int, int]] = goal
        while node is not None:
            path.append(node)
            node = parents[node]
        path.reverse()
        return path

    def _solution_target_length(self) -> int:
        """Target route length (in steps) that reads as a winding, hard answer."""

        total_cells = self.width * self.height
        baseline = self._distance_to_goal(0, 0)
        target = baseline + self.winding * 0.25 * total_cells
        # 极端尺寸下也不要把通路拉长到接近填满整个迷宫
        return int(round(min(target, total_cells * 0.45)))

    def _try_elongate_once(
        self, path: Sequence[Tuple[int, int]]
    ) -> Optional[List[Tuple[int, int]]]:
        """Reconnect the tree so the solution must take a longer detour.

        The tree is cut at a random edge of the current solution (splitting it
        into a start side and a goal side) and re-joined through a wall that
        connects two far-apart leaves of the two halves. The route then has to
        walk deep into one half and back out of the other, which lengthens it
        without ever breaking the perfect-maze property.
        """

        if len(path) < 3:
            return None

        cut = self.random.randrange(len(path) - 1)
        first, second = path[cut], path[cut + 1]
        self._set_wall(first, second, True)

        # 断开后生成树裂成两半。要算准新通路有多长，必须量「起点到 A 侧格子」
        # 和「B 侧格子到终点」的距离；量到断口的距离会高估，因为支路可能挂在
        # 断口之前的路径上。
        start_side = self._flood_from(path[0])
        goal_side = self._flood_from(path[-1])
        total_cells = self.width * self.height
        if len(start_side) + len(goal_side) != total_cells:
            # 迷宫不是生成树（例如外部已加过环），放弃本次重连
            self._set_wall(first, second, False)
            return None

        candidates: List[Tuple[Tuple[int, int], Tuple[int, int], int]] = []
        for cell_coord, distance in start_side.items():
            x, y = cell_coord
            cell = self.cell(x, y)
            for direction, (nx, ny) in self._neighbour_coords(x, y):
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if not cell.walls.get(direction, True):
                    continue
                other = (nx, ny)
                other_distance = goal_side.get(other)
                if other_distance is None:
                    continue
                # 新通路 = 起点到 A 侧格子 + 穿墙一步 + B 侧格子到终点
                candidates.append((cell_coord, other, distance + 1 + other_distance))

        if not candidates:
            self._set_wall(first, second, False)
            return None

        best_length = max(item[2] for item in candidates)
        if best_length <= len(path) - 1:
            # 怎么接都不会更长，说明这段已达上限
            self._set_wall(first, second, False)
            return None

        # 只在接近最优的候选里随机挑，既保证够长又不每次都挑同一个极端角落
        threshold = int(best_length * 0.9)
        picks = [item for item in candidates if item[2] >= threshold]
        anchor, bridge, _ = self.random.choice(picks)
        self._set_wall(anchor, bridge, False)

        new_path = self._shortest_path()
        if len(new_path) > len(path):
            return new_path

        # 没有变长就回滚，保持解答路径单调变长
        self._set_wall(anchor, bridge, True)
        self._set_wall(first, second, False)
        return None

    def _elongate_solution(self) -> None:
        """Iteratively stretch the solution route towards its target length."""

        target = self._solution_target_length()
        path = self._shortest_path()
        if len(path) - 1 >= target:
            return

        total_cells = self.width * self.height
        attempts = max(16, min(300, total_cells // 4))
        for _ in range(attempts):
            if len(path) - 1 >= target:
                break
            stretched = self._try_elongate_once(path)
            if stretched is not None:
                path = stretched

    def _path_attachment(
        self, path: Sequence[Tuple[int, int]]
    ) -> Dict[Tuple[int, int], int]:
        """Map every cell to the index of the solution cell its branch hangs off."""

        attachment: Dict[Tuple[int, int], int] = {}
        queue: Deque[Tuple[Tuple[int, int], int]] = deque()
        for index, coord in enumerate(path):
            attachment[coord] = index
            queue.append((coord, index))

        while queue:
            current, index = queue.popleft()
            x, y = current
            cell = self.cell(x, y)
            for direction, (nx, ny) in self._neighbour_coords(x, y):
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if cell.walls.get(direction, True):
                    continue
                nxt = (nx, ny)
                if nxt in attachment:
                    continue
                attachment[nxt] = index
                queue.append((nxt, index))
        return attachment

    def _add_loops(self) -> None:
        """Open extra walls in dead-end branches to create loops.

        A loop is only opened between two cells that hang off the *same* cell of
        the solution path. Such a loop starts and ends at the same attachment
        point, so it can add alternate routes inside a dead end without ever
        offering a shortcut across the answer. Loops that bridge two different
        parts of the solution are skipped on purpose: those are exactly what
        used to flatten a long answer back into a nearly straight line.
        """
        if self.loop_factor <= 0 and self.hairpin_chance <= 0:
            return

        total_cells = self.width * self.height
        attempts = int(total_cells * (self.loop_factor + 0.15 * self.hairpin_chance))
        if attempts <= 0:
            return

        attachment = self._path_attachment(self._shortest_path())

        for _ in range(attempts):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            cell = self.cell(x, y)
            neighbours = list(self.neighbours(x, y))
            if not neighbours:
                continue
            direction, neighbour = self.random.choice(neighbours)
            if not cell.walls.get(direction, True):
                continue
            if self._openings(cell) >= 3 or self._openings(neighbour) >= 3:
                continue

            if attachment[(x, y)] != attachment[(neighbour.x, neighbour.y)]:
                continue

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
        """Find a shortest path while preferring turns over long straight runs."""

        start = (0, 0)
        goal = (self.width - 1, self.height - 1)

        # Phase 1: reverse BFS from goal to compute shortest distance-to-goal
        # for every reachable cell.
        queue: Deque[Tuple[int, int]] = deque([goal])
        dist_to_goal: Dict[Tuple[int, int], int] = {goal: 0}

        while queue:
            current = queue.popleft()
            x, y = current
            current_dist = dist_to_goal[current]
            cell = self.cell(x, y)
            for direction, (nx, ny) in self._neighbour_coords(x, y):
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if cell.walls.get(direction, True):
                    continue
                next_coord = (nx, ny)
                if next_coord in dist_to_goal:
                    continue
                dist_to_goal[next_coord] = current_dist + 1
                queue.append(next_coord)

        if start not in dist_to_goal:
            raise RuntimeError("Maze has no solution")

        # Phase 2: walk from start to goal only through shortest-path edges
        # and prefer larger direction changes to avoid long straight segments.
        path: List[Tuple[int, int]] = [start]
        current = start
        previous_direction: Optional[Direction] = None

        while current != goal:
            x, y = current
            cell = self.cell(x, y)
            current_dist = dist_to_goal[current]

            candidates: List[Tuple[Direction, Tuple[int, int]]] = []
            for direction, (nx, ny) in self._neighbour_coords(x, y):
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if cell.walls.get(direction, True):
                    continue
                next_coord = (nx, ny)
                if dist_to_goal.get(next_coord) == current_dist - 1:
                    candidates.append((direction, next_coord))

            if not candidates:
                raise RuntimeError("Maze has no solution")

            if previous_direction is None:
                # First step: deterministic tie-break by larger remaining distance
                # from the next cell to start-ish frontier (more meander potential).
                chosen_direction, chosen_coord = max(
                    candidates,
                    key=lambda item: (
                        len(
                            [
                                1
                                for d, (nx, ny) in self._neighbour_coords(item[1][0], item[1][1])
                                if 0 <= nx < self.width
                                and 0 <= ny < self.height
                                and not self.cell(item[1][0], item[1][1]).walls.get(d, True)
                            ]
                        ),
                        item[0],
                    ),
                )
            else:
                chosen_direction, chosen_coord = max(
                    candidates,
                    key=lambda item: (
                        self._angular_difference(previous_direction, item[0]),
                        item[0],
                    ),
                )

            path.append(chosen_coord)
            current = chosen_coord
            previous_direction = chosen_direction

        return path


def generate_maze(
    difficulty: str = "medium",
    width: Optional[int] = None,
    height: Optional[int] = None,
    seed: Optional[int] = None,
    cell_shape: str = "square",
) -> Maze:
    """Factory function that creates and returns a generated maze."""

def generate_maze(
    difficulty: str = "medium",
    width: Optional[int] = None,
    height: Optional[int] = None,
    seed: Optional[int] = None,
    cell_shape: str = "square",
    winding: Optional[float] = None,
) -> Maze:
    """Factory function that creates and returns a generated maze.

    ``winding`` overrides how hard the solution path is stretched (0 keeps the
    shortest possible answer, 1 aims for a long meandering route). When left as
    ``None`` the difficulty profile decides it.
    """

    difficulty_profiles = {
        "easy": {
            "width": 12,
            "height": 12,
            "loop_factor": 0.05,
            "branching_chance": 0.25,
            "detour_bias": 0.25,
            "turn_bias": 0.25,
            "max_straight": 5,
            "hairpin_chance": 0.20,
            "winding": 0.40,
        },
        "medium": {
            "width": 20,
            "height": 20,
            "loop_factor": 0.15,
            "branching_chance": 0.40,
            "detour_bias": 0.45,
            "turn_bias": 0.32,
            "max_straight": 4,
            "hairpin_chance": 0.28,
            "winding": 0.65,
        },
        "hard": {
            "width": 28,
            "height": 28,
            "loop_factor": 0.20,
            "branching_chance": 0.55,
            "detour_bias": 0.55,
            "turn_bias": 0.40,
            "max_straight": 3,
            "hairpin_chance": 0.34,
            "winding": 0.95,
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

    # 环路密度随迷宫规模温和归一化：同样的难度，大迷宫不该被环路塞满。
    size_scale = min(1.0, (400.0 / cell_count) ** 0.25)
    loop_factor = profile["loop_factor"] * size_scale

    maze = Maze(
        width=maze_width,
        height=maze_height,
        seed=seed,
        loop_factor=loop_factor,
        branching_chance=profile["branching_chance"],
        cell_shape=cell_shape,
        detour_bias=profile["detour_bias"],
        turn_bias=profile["turn_bias"],
        max_straight=profile["max_straight"],
        hairpin_chance=profile["hairpin_chance"],
        winding=profile["winding"] if winding is None else winding,
    )
    maze.generate()
    return maze
