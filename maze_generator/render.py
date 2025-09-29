"""Rendering helpers that convert maze data structures into SVG previews and PDFs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from .maze import Maze
from .styles import MazeStyle, get_style

CELL_SIZE = 24  # points/pixels per maze cell
MARGIN = CELL_SIZE


@dataclass
class RenderResult:
    figure_path: Path
    solution_path: Path


def _wall_segments(maze: Maze) -> Iterable[Tuple[Tuple[float, float], Tuple[float, float]]]:
    for y in range(maze.height):
        for x in range(maze.width):
            cell = maze.cell(x, y)
            left, right = x, x + 1
            top, bottom = y, y + 1
            if cell.walls["N"]:
                yield (left, top), (right, top)
            if cell.walls["S"]:
                yield (left, bottom), (right, bottom)
            if cell.walls["W"]:
                yield (left, top), (left, bottom)
            if cell.walls["E"]:
                yield (right, top), (right, bottom)


def _path_points(path: Sequence[Tuple[int, int]]) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for x, y in path:
        xs.append(x + 0.5)
        ys.append(y + 0.5)
    return xs, ys


def _scale_point(x: float, y: float) -> Tuple[float, float]:
    return MARGIN + x * CELL_SIZE, MARGIN + y * CELL_SIZE


def _svg_header(width: float, height: float, background: str) -> List[str]:
    return [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">",
        f"  <rect width=\"100%\" height=\"100%\" fill=\"{background}\" />",
    ]


def _svg_lines(segments: Iterable[Tuple[Tuple[float, float], Tuple[float, float]]], style: MazeStyle) -> List[str]:
    lines: List[str] = []
    for (x1, y1), (x2, y2) in segments:
        sx1, sy1 = _scale_point(x1, y1)
        sx2, sy2 = _scale_point(x2, y2)
        lines.append(
            f"  <line x1=\"{sx1}\" y1=\"{sy1}\" x2=\"{sx2}\" y2=\"{sy2}\" stroke=\"{style.wall_color}\" stroke-width=\"{style.wall_width}\" stroke-linecap=\"square\" />"
        )
    return lines


def _svg_solution(path: Sequence[Tuple[int, int]], style: MazeStyle) -> str:
    xs, ys = _path_points(path)
    points = ["{:.2f},{:.2f}".format(*_scale_point(x, y)) for x, y in zip(xs, ys)]
    return (
        f"  <polyline points=\"{' '.join(points)}\" "
        f"fill=\"none\" stroke=\"{style.solution_color}\" stroke-width=\"{style.solution_width}\" "
        f"stroke-linecap=\"round\" stroke-linejoin=\"round\" />"
    )


def _svg_footer() -> List[str]:
    return ["</svg>"]


def render_maze_svg(maze: Maze, style: MazeStyle, show_solution: bool = False) -> str:
    width = maze.width * CELL_SIZE + MARGIN * 2
    height = maze.height * CELL_SIZE + MARGIN * 2
    svg_parts = _svg_header(width, height, style.background_color)
    svg_parts.extend(_svg_lines(_wall_segments(maze), style))
    if show_solution:
        svg_parts.append(_svg_solution(maze.solve(), style))
    svg_parts.extend(_svg_footer())
    return "\n".join(svg_parts)


def save_preview_images(maze: Maze, style_name: str, directory: Path, prefix: str = "maze") -> RenderResult:
    """Render the maze and its solution to SVG files for quick previews."""

    directory.mkdir(parents=True, exist_ok=True)
    style = get_style(style_name)
    maze_path = directory / f"{prefix}_{style.name}.svg"
    solution_path = directory / f"{prefix}_{style.name}_solution.svg"

    maze_path.write_text(render_maze_svg(maze, style, show_solution=False), encoding="utf-8")
    solution_path.write_text(render_maze_svg(maze, style, show_solution=True), encoding="utf-8")

    return RenderResult(figure_path=maze_path, solution_path=solution_path)


def _pdf_stream_for_page(
    maze: Maze,
    style: MazeStyle,
    show_solution: bool,
    metadata: str | None = None,
) -> Tuple[str, float, float]:
    width = maze.width * CELL_SIZE + MARGIN * 2
    height = maze.height * CELL_SIZE + MARGIN * 2
    commands: List[str] = []

    # Background
    commands.append("q")
    bg_r, bg_g, bg_b = _hex_to_rgb(style.background_color)
    commands.append(f"{bg_r} {bg_g} {bg_b} rg")
    commands.append(f"0 0 {width} {height} re f")
    commands.append("Q")

    # Walls
    wall_r, wall_g, wall_b = _hex_to_rgb(style.wall_color)
    commands.append("q")
    commands.append(f"{style.wall_width} w")
    commands.append(f"{wall_r} {wall_g} {wall_b} RG")
    for (x1, y1), (x2, y2) in _wall_segments(maze):
        sx1, sy1 = _scale_point(x1, y1)
        sx2, sy2 = _scale_point(x2, y2)
        sy1_pdf = height - sy1
        sy2_pdf = height - sy2
        commands.append(f"{sx1:.2f} {sy1_pdf:.2f} m {sx2:.2f} {sy2_pdf:.2f} l S")
    commands.append("Q")

    if show_solution:
        sol_r, sol_g, sol_b = _hex_to_rgb(style.solution_color)
        xs, ys = _path_points(maze.solve())
        if xs and ys:
            commands.append("q")
            commands.append(f"{style.solution_width} w")
            commands.append(f"{sol_r} {sol_g} {sol_b} RG")
            first_x, first_y = _scale_point(xs[0], ys[0])
            commands.append(f"{first_x:.2f} {height - first_y:.2f} m")
            for x, y in zip(xs[1:], ys[1:]):
                sx, sy = _scale_point(x, y)
                commands.append(f"{sx:.2f} {height - sy:.2f} l")
            commands.append("S")
            commands.append("Q")

    if metadata:
        commands.extend(_pdf_title(metadata, width, height))

    stream = "\n".join(commands)
    return stream, width, height


def _pdf_title(text: str, width: float, height: float) -> List[str]:
    y = height - 18
    x = MARGIN
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return [
        "q",
        "/F1 14 Tf",
        "0 0 0 rg",
        f"1 0 0 1 {x:.2f} {y:.2f} Tm",
        f"({escaped}) Tj",
        "Q",
    ]


def _hex_to_rgb(color: str) -> Tuple[float, float, float]:
    color = color.lstrip("#")
    if len(color) == 3:
        color = "".join(c * 2 for c in color)
    r = int(color[0:2], 16) / 255
    g = int(color[2:4], 16) / 255
    b = int(color[4:6], 16) / 255
    return r, g, b


def save_maze_pdf(
    maze: Maze,
    style_name: str,
    pdf_path: Path,
    metadata: str | None = None,
) -> None:
    style = get_style(style_name)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    pages: List[Tuple[str, float, float]] = []
    pages.append(_pdf_stream_for_page(maze, style, show_solution=False, metadata=metadata))
    solution_meta = f"Solution – {metadata}" if metadata else "Solution"
    pages.append(_pdf_stream_for_page(maze, style, show_solution=True, metadata=solution_meta))

    content_objects: List[bytes] = []
    for content, _, _ in pages:
        content_bytes = content.encode("utf-8")
        obj = (
            f"<< /Length {len(content_bytes)} >>\nstream\n".encode("utf-8")
            + content_bytes
            + b"\nendstream\n"
        )
        content_objects.append(obj)

    font_obj = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n"

    # Build PDF objects in order
    objects: List[bytes] = []
    page_count = len(pages)
    kids_refs = " ".join(str(4 + idx) + " 0 R" for idx in range(page_count))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(
        f"<< /Type /Pages /Count {page_count} /Kids [{kids_refs}] >>\nendobj\n".encode("utf-8")
    )
    objects.append(font_obj + b"endobj\n")

    widths = [page[1] for page in pages]
    heights = [page[2] for page in pages]

    # Page objects referencing content streams 6 and 7
    for idx, (width, height) in enumerate(zip(widths, heights), start=0):
        content_num = 4 + page_count + idx
        page_obj = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] ".encode("utf-8")
            + f"/Contents {content_num} 0 R /Resources << /Font << /F1 3 0 R >> >> >>\nendobj\n".encode("utf-8")
        )
        objects.append(page_obj)

    for obj in content_objects:
        objects.append(obj + b"endobj\n")

    header = b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n"
    offsets: List[int] = []
    current_offset = len(header)
    serialized_objects: List[bytes] = []
    for index, obj in enumerate(objects, start=1):
        obj_bytes = f"{index} 0 obj\n".encode("utf-8") + obj
        serialized_objects.append(obj_bytes)
        offsets.append(current_offset)
        current_offset += len(obj_bytes)

    with pdf_path.open("wb") as fh:
        fh.write(header)
        for obj in serialized_objects:
            fh.write(obj)
        xref_pos = fh.tell()
        fh.write(f"xref\n0 {len(serialized_objects)+1}\n".encode("utf-8"))
        fh.write(b"0000000000 65535 f \n")
        for offset in offsets:
            fh.write(f"{offset:010d} 00000 n \n".encode("utf-8"))
        fh.write(
            f"trailer\n<< /Size {len(serialized_objects)+1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode(
                "utf-8"
            )
        )


def render_maze_figure(maze: Maze, style: MazeStyle, show_solution: bool = False):
    """Compatibility helper retained for API parity."""

    return render_maze_svg(maze, style, show_solution)


def export_maze_pdf(
    maze: Maze,
    style_name: str,
    pdf_path: Path,
    metadata: str | None = None,
) -> Path:
    save_maze_pdf(maze, style_name=style_name, pdf_path=pdf_path, metadata=metadata)
    return pdf_path
