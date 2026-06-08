# translate — TL 实时屏幕翻译工具

基于 Python 的实时屏幕翻译工具，可截取屏幕指定区域的英文文本，通过 OCR 识别后自动翻译为中文，并在透明悬浮窗中显示。

## 功能特性

- 🖥️ **实时屏幕翻译**：自动截取屏幕区域，OCR 识别英文并翻译为中文
- 🎯 **两种检测模式**：鼠标周围区域 / 自定义固定区域
- 🪟 **透明悬浮窗**：无边框、置顶、鼠标穿透，翻译结果浮于屏幕上方
- 🎨 **极简 macOS 风格 UI**：圆角卡片、轻柔阴影、自定义标题栏
- 🔤 **多 OCR 引擎**：Tesseract / EasyOCR 可选
- 🌐 **多翻译引擎**：Google 翻译（免费）、DeepL、OpenAI
- ⏯️ **全局热键**：Ctrl+Shift+T 随时暂停/恢复翻译
- 📋 **系统托盘**：最小化到托盘，后台持续翻译

## 系统依赖

### Tesseract OCR（推荐）
1. 下载安装：[Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki)
2. 安装时勾选中文语言包
3. 确保 `tesseract.exe` 在系统 PATH 中

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
python TL.py
```

## 项目结构

```
translate/
├── TL.py              # 主程序（单文件）
├── requirements.txt   # Python 依赖
└── README.md
```
