# Maze Generator

这个项目提供一个命令行迷宫生成器，可以自定义迷宫难度、样式并生成迷宫的预览图和带解答的 PDF。

## 特性

- 支持 `easy`、`medium`、`hard` 三种难度，亦可自定义宽度与高度。
- 提供 `classic`、`blueprint`、`night` 三种渲染样式。
- 可生成 ASCII 预览、SVG 预览图以及包含迷宫与解答的 PDF。
- 支持通过随机种子复现同一迷宫。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

命令行提供三个子命令：

```bash
python -m maze_generator.cli generate [参数]
python -m maze_generator.cli preview [参数]
python -m maze_generator.cli export-pdf [参数]
```

常用参数：

- `--difficulty {easy,medium,hard}`：选择难度，默认 `medium`。
- `--width --height`：自定义迷宫尺寸，覆盖难度默认尺寸。
- `--style {classic,blueprint,night}`：选择渲染样式。
- `--seed`：设置随机种子以便复现。

### 生成 ASCII 预览

```bash
python -m maze_generator.cli generate --difficulty easy --seed 42
```

### 生成 SVG 预览

```bash
python -m maze_generator.cli preview --difficulty hard --style blueprint --output previews
```

生成的预览图和解答图会保存在 `previews/` 目录下，以 SVG 文件呈现。

### 导出 PDF

```bash
python -m maze_generator.cli export-pdf --difficulty medium --style night --pdf output/maze.pdf --title "Team Maze Challenge"
```

PDF 文件包含两页：第一页是迷宫，第二页为迷宫答案。

## 开发

- `maze_generator/maze.py`：迷宫的生成逻辑与求解算法。
- `maze_generator/render.py`：自定义的 SVG/PDF 渲染与导出工具。
- `maze_generator/cli.py`：命令行接口。

欢迎根据需要扩展难度和样式配置。
