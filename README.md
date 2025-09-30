# Maze Generator

这个项目提供一个命令行迷宫生成器，可以自定义迷宫难度、样式并生成迷宫的预览图和带解答的 PDF。同时也可以作为一个 Python 包引入，在代码中直接生成迷宫、渲染 SVG 或导出 PDF。

## 特性

- 支持 `easy`、`medium`、`hard` 三种难度，或通过自定义宽高生成任意尺寸的迷宫。
- 内置 `classic`、`blueprint`、`night` 三种渲染样式，可扩展自定义主题。
- 支持 `square` 与 `hex` 两种格子形状，对应方形和六边形迷宫。
- 生成 ASCII 预览、SVG 预览图，以及包含迷宫和解答两页的 PDF。
- 支持使用随机种子复现同一个迷宫。
- 完全纯 Python 实现，无第三方依赖。

## 安装依赖

```bash
pip install -r requirements.txt
```

> `requirements.txt` 当前不包含第三方库，可直接使用系统自带 Python 运行。

## 命令行使用

命令行提供三个子命令：

```bash
python -m maze_generator.cli generate [参数]
python -m maze_generator.cli preview [参数]
python -m maze_generator.cli export-pdf [参数]
```

### 通用参数

所有子命令都支持以下参数：

- `--difficulty {easy,medium,hard}`：选择难度，默认 `medium`。难度会根据最终的迷宫格子数量自动调整循环密度，因此自定义尺寸也能得到合适复杂度。
- `--width --height`：自定义迷宫宽和高，传入任意正整数（大于 1）。如果指定则覆盖难度默认尺寸。
- `--seed`：设置随机种子以便复现。
- `--style {classic,blueprint,night}`：选择渲染样式（对 `preview`、`export-pdf` 两个子命令生效）。
- `--cell-shape {square,hex}`：选择迷宫格子形状，默认为 `square`。

### 自定义宽高

可以单独通过宽度和高度配置迷宫尺寸，而不使用预设难度的默认值：

```bash
python -m maze_generator.cli generate --width 40 --height 24 --seed 99
```

上例会生成一个 40×24 的迷宫；若同时提供 `--difficulty`，尺寸仍以 `--width` 与 `--height` 为准，仅使用难度控制循环密度。

同样的尺寸参数也适用于 `preview` 与 `export-pdf` 子命令，可组合 `--style`、`--cell-shape` 等选项一起使用。

### 切换迷宫格子形状

迷宫同时支持方格与六边形拓扑。可以通过 `--cell-shape` 在命令行中切换：

```bash
python -m maze_generator.cli preview --cell-shape hex --style night --output previews
```

上面的命令会生成六边形迷宫的 SVG 预览与解答图。同样的参数也适用于 `generate`（ASCII）与 `export-pdf`。当选择 `hex` 时，ASCII 输出会使用适配六边形的字符排布，而 PDF 与 SVG 则采用相应的六边形几何布局。

### 生成 ASCII 预览

```bash
python -m maze_generator.cli generate --difficulty easy --seed 42
```

终端会直接输出 ASCII 版迷宫，适合快速验证参数效果。

提示：若使用 `--cell-shape hex`，生成的 ASCII 会自动转为适配六边形网格的字符画格式。

### 生成 SVG 预览

```bash
python -m maze_generator.cli preview --difficulty hard --style blueprint --output previews
```

该命令会在指定目录（默认 `previews/`）生成两份文件：

- `maze_<style>.svg`：迷宫图。
- `maze_<style>_solution.svg`：带解答路径的迷宫图。

命令执行后会在终端打印两个 SVG 文件的路径，方便脚本自动化处理或调试。

### 导出 PDF

```bash
python -m maze_generator.cli export-pdf --difficulty medium --style night --pdf output/maze.pdf --title "Team Maze Challenge"
```

导出的 PDF 含两页，第一页为迷宫，第二页为解答页。若提供 `--title`，第一页会显示标题，第二页自动添加 `Solution – <标题>`。

`--pdf` 默认值为 `maze.pdf`，成功导出后命令行会打印生成的 PDF 路径。

### 选择输出格式

三个子命令分别对应不同的输出格式，可根据需求挑选：

- `generate`：直接在终端输出 ASCII 迷宫，便于快速测试参数或脚本处理。
- `preview`：生成 SVG 图片，可在浏览器或矢量软件中查看，输出目录默认 `previews/`，可通过 `--output` 更改。
- `export-pdf`：导出两页 PDF，适合打印或分享，可用 `--pdf` 自定义文件名。

你可以针对不同格式重复使用相同的尺寸、样式与格子形状参数，实现一致配置下的多种导出方式。

## 以库形式使用

除了命令行，亦可在其他 Python 项目中引入使用：

```python
from pathlib import Path

from maze_generator import generate_maze, save_preview_images, export_maze_pdf

maze = generate_maze(difficulty="medium", seed=123, cell_shape="hex")
preview = save_preview_images(maze, style_name="classic", directory=Path("previews"))
export_maze_pdf(maze, style_name="night", pdf_path=Path("maze.pdf"), metadata="My Maze")
```

- `generate_maze` 返回一个已经生成好的迷宫对象，包含求解和 ASCII 渲染等实用方法。
- `save_preview_images` 返回包含迷宫与解答 SVG 文件路径的结果对象，属性为 `figure_path` 与 `solution_path`。
- `export_maze_pdf` 生成两页 PDF，并返回输出文件路径。

## 目录结构

- `maze_generator/maze.py`：迷宫生成逻辑、求解与 ASCII 渲染。
- `maze_generator/render.py`：SVG 渲染与 PDF 导出实现。
- `maze_generator/cli.py`：命令行接口及参数解析。
- `maze_generator/styles.py`：预设的渲染样式配置。

欢迎根据需要扩展更多难度配置或渲染主题！
