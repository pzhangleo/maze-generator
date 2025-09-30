import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from maze_generator.maze import Maze
from maze_generator.render import save_preview_images


class PreviewGenerationTest(unittest.TestCase):
    def test_preview_images_are_created_and_non_empty(self) -> None:
        maze = Maze(
            width=20,
            height=20,
            seed=42,
            loop_factor=0.2,
            branching_chance=0.45,
        )
        maze.generate()

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            result = save_preview_images(
                maze,
                style_name="classic",
                directory=tmp_path,
                prefix="test_maze",
            )

            self.assertTrue(result.figure_path.exists(), "Maze preview SVG should be created")
            self.assertTrue(result.solution_path.exists(), "Solution preview SVG should be created")

            self.assertGreater(result.figure_path.stat().st_size, 0, "Maze preview should not be empty")
            self.assertGreater(result.solution_path.stat().st_size, 0, "Solution preview should not be empty")

            figure_markup = result.figure_path.read_text(encoding="utf-8")
            solution_markup = result.solution_path.read_text(encoding="utf-8")

            self.assertIn("<svg", figure_markup)
            self.assertIn("<svg", solution_markup)
            self.assertIn("<polyline", solution_markup, "Solution preview should draw the solution path")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
