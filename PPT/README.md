# PPTX文件切分工具

这是一个用于将PPTX文件按章节或指定页面范围切分成多个独立PPTX文件的工具。

## 功能特点

- 支持按章节自动切分（通过识别包含"章"或"节"的标题）
- 支持按指定页面范围手动切分
- 保留原PPTX文件的母版和样式
- 自动创建输出目录

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 按章节自动切分

```bash
python split_pptx.py input.pptx output_directory
```

### 按指定页面范围切分

```bash
python split_pptx.py input.pptx output_directory --page-ranges "1-3,5-8,9-10"
```

### 参数说明

- `input.pptx`: 输入的PPTX文件路径
- `output_directory`: 输出目录路径
- `--page-ranges`: 可选的页面范围参数，格式为"起始页-结束页"，多个范围用逗号分隔

## 输出文件

切分后的文件将保存在指定的输出目录中，文件名格式为：`原文件名_section_序号.pptx`

## 注意事项

1. 确保输入的PPTX文件存在且可访问
2. 程序会自动创建输出目录（如果不存在）
3. 如果使用页面范围切分，请确保指定的页面范围有效
4. 按章节切分时，程序会识别包含"章"或"节"的标题作为章节分隔点 