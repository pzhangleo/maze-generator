"""Command line interface for generating and exporting mazes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .maze import generate_maze
from .render import save_preview_images
from .styles import STYLES
from .pdf import export_maze_pdf


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Maze generator with preview and PDF export")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"], help="Maze difficulty level")
    common.add_argument("--width", type=int, default=None, help="Custom maze width (overrides difficulty preset)")
    common.add_argument("--height", type=int, default=None, help="Custom maze height (overrides difficulty preset)")
    common.add_argument("--seed", type=int, default=None, help="Random seed for reproducible mazes")
    common.add_argument("--style", default="classic", choices=list(STYLES.keys()), help="Visual style for rendering")
    common.add_argument(
        "--cell-shape",
        default="square",
        choices=["square", "hex"],
        help="Shape of the maze cells",
    )

    preview_parser = subparsers.add_parser("preview", parents=[common], help="Generate preview images of the maze")
    preview_parser.add_argument("--output", type=Path, default=Path("previews"), help="Directory to save preview images")

    generate_parser = subparsers.add_parser("generate", parents=[common], help="Generate a maze and print an ASCII preview")

    pdf_parser = subparsers.add_parser("export-pdf", parents=[common], help="Export the maze and solution to a PDF")
    pdf_parser.add_argument("--pdf", type=Path, default=Path("maze.pdf"), help="Path of the PDF file to create")
    pdf_parser.add_argument("--title", type=str, default=None, help="Optional title to add to the PDF")

    return parser.parse_args(argv)


def _create_maze(args: argparse.Namespace):
    # The difficulty choices for styles were added to reuse argparse choices, filter here.
    difficulty = args.difficulty if args.difficulty in {"easy", "medium", "hard"} else "medium"
    return generate_maze(
        difficulty=difficulty,
        width=args.width,
        height=args.height,
        seed=args.seed,
        cell_shape=args.cell_shape,
    )


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    maze = _create_maze(args)

    if args.command == "generate":
        print(maze.ascii_render())
    elif args.command == "preview":
        result = save_preview_images(maze, style_name=args.style, directory=args.output)
        print(f"Maze preview saved to {result.figure_path}")
        print(f"Solution preview saved to {result.solution_path}")
    elif args.command == "export-pdf":
        pdf_path = export_maze_pdf(
            maze,
            pdf_path=args.pdf,
            style_name=args.style,
            metadata=args.title,
        )
        print(f"Maze PDF saved to {pdf_path}")
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
