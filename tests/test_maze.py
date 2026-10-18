"""Tests for maze generation, solution shape, rendering and export."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from maze_generator import export_maze_pdf, generate_maze, save_preview_images
from maze_generator.maze import Maze

SHAPES = ("square", "hex")
DIFFICULTIES = ("easy", "medium", "hard")


def manhattan(maze: Maze) -> int:
    return (maze.width - 1) + (maze.height - 1)


def reachable_cells(maze: Maze) -> set:
    seen = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        x, y = stack.pop()
        cell = maze.cell(x, y)
        for direction, (nx, ny) in maze._neighbour_coords(x, y):
            if not (0 <= nx < maze.width and 0 <= ny < maze.height):
                continue
            if cell.walls[direction] or (nx, ny) in seen:
                continue
            seen.add((nx, ny))
            stack.append((nx, ny))
    return seen


def open_edges(maze: Maze) -> int:
    count = 0
    for y in range(maze.height):
        for x in range(maze.width):
            cell = maze.cell(x, y)
            for direction, (nx, ny) in maze._neighbour_coords(x, y):
                if 0 <= nx < maze.width and 0 <= ny < maze.height and not cell.walls[direction]:
                    count += 1
    return count // 2


def assert_valid_solution(maze: Maze) -> None:
    path = maze.solve()
    assert path[0] == (0, 0)
    assert path[-1] == (maze.width - 1, maze.height - 1)
    assert len(set(path)) == len(path), "solution revisits a cell"
    for current, following in zip(path, path[1:]):
        steps = {coord: direction for direction, coord in maze._neighbour_coords(*current)}
        assert following in steps, f"{current} -> {following} is not a step"
        assert not maze.cell(*current).walls[steps[following]], f"solution crosses a wall at {current}"


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_every_cell_is_reachable(shape: str, difficulty: str) -> None:
    maze = generate_maze(difficulty=difficulty, seed=1234, cell_shape=shape)
    assert len(reachable_cells(maze)) == maze.width * maze.height


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
@pytest.mark.parametrize("seed", (0, 7, 2024))
def test_solution_stays_inside_the_corridors(shape: str, difficulty: str, seed: int) -> None:
    maze = generate_maze(difficulty=difficulty, seed=seed, cell_shape=shape)
    assert_valid_solution(maze)


@pytest.mark.parametrize("shape", SHAPES)
def test_entrance_and_exit_are_open(shape: str) -> None:
    maze = generate_maze(difficulty="medium", seed=5, cell_shape=shape)
    goal = maze.cell(maze.width - 1, maze.height - 1)
    if shape == "square":
        assert maze.cell(0, 0).walls["N"] is False
        assert goal.walls["S"] is False
    else:
        assert maze.cell(0, 0).walls["W"] is False
        assert goal.walls["E"] is False


def test_same_seed_reproduces_the_same_maze() -> None:
    first = generate_maze(difficulty="hard", seed=99)
    second = generate_maze(difficulty="hard", seed=99)
    for y in range(first.height):
        for x in range(first.width):
            assert first.cell(x, y).walls == second.cell(x, y).walls
    assert first.solve() == second.solve()


def test_different_seeds_produce_different_mazes() -> None:
    first = generate_maze(difficulty="hard", seed=1)
    second = generate_maze(difficulty="hard", seed=2)
    assert first.solve() != second.solve()


@pytest.mark.parametrize(
    ("difficulty", "minimum_ratio"),
    [("easy", 1.8), ("medium", 2.6), ("hard", 3.8)],
)
def test_solution_is_winding_not_a_straight_run(difficulty: str, minimum_ratio: float) -> None:
    """Regression guard: answers used to hug the diagonal at ~1.3x the distance."""

    ratios = []
    for seed in range(5):
        maze = generate_maze(difficulty=difficulty, seed=seed)
        ratios.append((len(maze.solve()) - 1) / manhattan(maze))
    assert sum(ratios) / len(ratios) >= minimum_ratio, ratios


def test_hard_solution_is_clearly_longer_than_the_shortest_route() -> None:
    maze = generate_maze(difficulty="hard", seed=7)
    steps = len(maze.solve()) - 1
    assert steps >= 3 * manhattan(maze), steps


def test_winding_zero_does_not_stretch_the_route() -> None:
    straight = generate_maze(difficulty="hard", seed=7, winding=0.0)
    winding = generate_maze(difficulty="hard", seed=7, winding=1.0)
    assert len(straight.solve()) < len(winding.solve())


def test_loops_never_shorten_the_solution() -> None:
    """Loops may only be carved inside a branch, so the answer keeps its length."""

    def build(loop_factor: float) -> Maze:
        maze = Maze(
            20,
            20,
            seed=11,
            loop_factor=loop_factor,
            branching_chance=0.5,
            detour_bias=0.5,
            turn_bias=0.3,
            max_straight=4,
            hairpin_chance=0.0,
            winding=0.9,
        )
        maze.generate()
        return maze

    without_loops = build(0.0)
    with_loops = build(0.3)

    total = with_loops.width * with_loops.height
    assert open_edges(with_loops) > open_edges(without_loops), "no loop was carved"
    assert open_edges(with_loops) > total - 1
    assert len(with_loops.solve()) == len(without_loops.solve())


@pytest.mark.parametrize("shape", SHAPES)
def test_custom_dimensions(shape: str) -> None:
    maze = generate_maze(difficulty="easy", width=17, height=9, seed=3, cell_shape=shape)
    assert (maze.width, maze.height) == (17, 9)
    assert len(reachable_cells(maze)) == 17 * 9
    assert_valid_solution(maze)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"width": 1, "height": 5}, "dimensions"),
        ({"difficulty": "impossible"}, "Unknown difficulty"),
        ({"cell_shape": "triangle"}, "cell_shape"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        generate_maze(seed=1, **kwargs)


def test_winding_is_clamped() -> None:
    assert Maze(5, 5, winding=-3.0).winding == 0.0
    assert Maze(5, 5, winding=4.0).winding == 1.0


def test_preview_images_and_pdf(tmp_path: Path) -> None:
    maze = generate_maze(difficulty="medium", seed=21)

    preview = save_preview_images(maze, style_name="classic", directory=tmp_path, prefix="demo")
    assert preview.figure_path.name == "demo_classic.svg"
    assert preview.solution_path.name == "demo_classic_solution.svg"
    assert preview.figure_path.read_text(encoding="utf-8").startswith("<?xml")
    solution_markup = preview.solution_path.read_text(encoding="utf-8")
    assert "polyline" in solution_markup

    pdf_path = export_maze_pdf(
        maze, style_name="night", pdf_path=tmp_path / "maze.pdf", metadata="Unit Test"
    )
    payload = pdf_path.read_bytes()
    assert payload.startswith(b"%PDF")
    assert payload.count(b"/Type /Page ") == 2, "expected a puzzle page and a solution page"


def test_layout_size_is_positive() -> None:
    for shape in SHAPES:
        maze = generate_maze(difficulty="easy", seed=1, cell_shape=shape)
        width, height = maze.layout_size()
        assert width > 0 and height > 0
        assert math.isfinite(width) and math.isfinite(height)
