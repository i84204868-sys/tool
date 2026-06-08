# -*- coding: utf-8 -*-
"""
TL — 实时屏幕翻译工具 (Screen Translator)
===========================================

依赖库 (pip install):
    PySide6>=6.5.0          # GUI 框架
    mss>=9.0.0              # 高速屏幕截图
    pytesseract>=0.3.10     # Tesseract OCR 封装
    EasyOCR>=1.7.0          # 备用 OCR 引擎 (含 PyTorch 依赖)
    deep-translator>=1.11.0 # Google / DeepL 翻译
    keyboard>=0.13.5        # 全局热键
    Pillow>=10.0.0          # 图像处理
    opencv-python>=4.8.0    # 可选，EasyOCR 依赖

系统依赖:
    Tesseract-OCR: 下载安装 https://github.com/UB-Mannheim/tesseract/wiki
    安装后确保 tesseract.exe 在 PATH 中，或在代码中手动指定路径。

运行方式:
    python TL.py

作者: YY
"""

import sys
import os
import time
import threading
from pathlib import Path

# ── Qt / GUI ──────────────────────────────────────────────────────
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QComboBox, QSlider, QSpinBox, QHBoxLayout, QVBoxLayout,
    QGroupBox, QGridLayout, QSystemTrayIcon, QMenu, QColorDialog,
    QMessageBox, QSizePolicy, QFrame, QLineEdit,
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QTimer, QRect, QPoint, QSize, QEvent,
    QMetaObject, Q_ARG, Slot,
)
from PySide6.QtGui import (
    QAction, QIcon, QColor, QFont, QPainter, QPen, QBrush,
    QPixmap, QImage, QMouseEvent, QKeySequence, QShortcut,
)

# ── 截图 / OCR / 翻译 ─────────────────────────────────────────────
try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

try:
    from deep_translator import GoogleTranslator
    HAS_GOOGLE_TRANSLATE = True
except ImportError:
    HAS_GOOGLE_TRANSLATE = False

try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False

from PIL import Image
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# 自动检测 Tesseract 路径
# ═══════════════════════════════════════════════════════════════════

def _find_tesseract() -> str:
    """自动查找 tesseract.exe 路径。"""
    # 常见安装路径
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # 尝试 PATH
    import shutil
    found = shutil.which("tesseract")
    return found or ""


_TESSERACT_PATH = _find_tesseract()
if _TESSERACT_PATH and HAS_TESSERACT:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_PATH


# ═══════════════════════════════════════════════════════════════════
# 翻译核心模块
# ═══════════════════════════════════════════════════════════════════

class TranslationCore:
    """负责 OCR 识别和文本翻译，封装不同引擎的切换逻辑。"""

    def __init__(self, ocr_engine: str = "tesseract",
                 translate_engine: str = "google",
                 tesseract_path: str = "",
                 deepl_key: str = "",
                 openai_key: str = ""):
        self.ocr_engine = ocr_engine
        self.translate_engine = translate_engine
        self._easyocr_reader = None
        self._translator = None

        # 配置 Tesseract 路径
        if tesseract_path and os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # 初始化 EasyOCR（延迟加载）
        if ocr_engine == "easyocr" and HAS_EASYOCR:
            self._init_easyocr()

    def _init_easyocr(self):
        """延迟初始化 EasyOCR（首次加载较慢）。"""
        if self._easyocr_reader is None and HAS_EASYOCR:
            self._easyocr_reader = easyocr.Reader(["en"], gpu=True)

    def set_ocr_engine(self, engine: str):
        self.ocr_engine = engine
        if engine == "easyocr" and HAS_EASYOCR:
            self._init_easyocr()

    def ocr(self, image: np.ndarray) -> list[tuple[str, tuple]]:
        """
        对图像执行 OCR，返回 [(文本, 边界框), ...]。
        边界框格式: (x, y, w, h)
        """
        results = []

        if self.ocr_engine == "tesseract" and HAS_TESSERACT:
            # 转 PIL 图像
            pil_img = Image.fromarray(image)
            # 获取每个词块的位置信息
            data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
            n = len(data["text"])
            for i in range(n):
                text = data["text"][i].strip()
                if text and len(text) > 1:  # 过滤单字符噪声
                    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    results.append((text, (x, y, w, h)))

        elif self.ocr_engine == "easyocr" and HAS_EASYOCR:
            self._init_easyocr()
            if self._easyocr_reader:
                raw = self._easyocr_reader.readtext(image)
                for (bbox, text, conf) in raw:
                    if conf > 0.3 and text.strip():
                        x1, y1 = int(bbox[0][0]), int(bbox[0][1])
                        x2, y2 = int(bbox[2][0]), int(bbox[2][1])
                        results.append((text.strip(), (x1, y1, x2 - x1, y2 - y1)))

        # 按 y 坐标排序，同行按 x 排序
        results.sort(key=lambda r: (r[1][1], r[1][0]))
        return results

    def translate(self, text: str, source: str = "auto", target: str = "zh-CN") -> str:
        """翻译单段文本，支持重试。"""
        if not text or not text.strip():
            return ""

        if self.translate_engine == "google" and HAS_GOOGLE_TRANSLATE:
            for attempt in range(3):
                try:
                    result = GoogleTranslator(source=source, target=target).translate(text)
                    return result or ""
                except Exception as e:
                    if attempt == 2:
                        return f"[翻译失败: {e}]"
                    time.sleep(1.0)
        else:
            return "[翻译引擎不可用]"

        return ""


# ═══════════════════════════════════════════════════════════════════
# 工作线程
# ═══════════════════════════════════════════════════════════════════

class ScreenshotWorker(QThread):
    """截图 → OCR → 翻译 的工作线程，通过信号与 UI 层通信。"""

    # 信号: (原文列表, 译文列表, 边界框列表)
    translation_ready = Signal(list, list, list)
    # 信号: 状态文本
    status_update = Signal(str)
    # 信号: 截图预览 (用于测试)
    preview_ready = Signal(np.ndarray)

    def __init__(self, core: TranslationCore, parent=None):
        super().__init__(parent)
        self.core = core
        self._running = False
        self._paused = False

        # 检测区域配置
        self.mode = "mouse"          # "mouse" | "fixed"
        self.mouse_w = 500           # 鼠标周围宽度
        self.mouse_h = 300           # 鼠标周围高度
        self.fixed_x, self.fixed_y = 0, 0
        self.fixed_w, self.fixed_h = 400, 300

        # 刷新间隔 (ms)
        self.interval_ms = 800

        # 悬浮窗偏移
        self.overlay_offset_x = 0
        self.overlay_offset_y = 30

    def capture_region(self) -> tuple[np.ndarray, QRect]:
        """根据当前模式截取屏幕区域，返回 (图像数组, 区域矩形)。"""
        if self.mode == "mouse":
            from PySide6.QtGui import QCursor
            pos = QCursor.pos()
            x = max(0, pos.x() - self.mouse_w // 2)
            y = max(0, pos.y() - self.mouse_h // 2)
            w, h = self.mouse_w, self.mouse_h
        else:
            x, y = self.fixed_x, self.fixed_y
            w, h = self.fixed_w, self.fixed_h

        rect = QRect(x, y, w, h)

        if HAS_MSS:
            with mss.mss() as sct:
                monitor = {"left": x, "top": y, "width": w, "height": h}
                img = np.array(sct.grab(monitor))
                # mss 返回 BGRA → RGB
                img = img[:, :, :3][:, :, ::-1]
        else:
            from PIL import ImageGrab
            pil_img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            img = np.array(pil_img)

        return img, rect

    def run(self):
        """主循环：截图 → OCR → 翻译 → 发射信号。"""
        self._running = True
        self.status_update.emit("运行中")

        while self._running:
            if self._paused:
                self.status_update.emit("已暂停")
                time.sleep(0.3)
                continue

            try:
                img, rect = self.capture_region()
                ocr_results = self.core.ocr(img)

                originals, translations, boxes = [], [], []
                for text, (ox, oy, ow, oh) in ocr_results:
                    # 坐标转换为屏幕绝对坐标
                    abs_box = (
                        rect.x() + ox,
                        rect.y() + oy,
                        ow, oh,
                    )
                    translated = self.core.translate(text)
                    originals.append(text)
                    translations.append(translated)
                    boxes.append(abs_box)

                if originals:
                    self.translation_ready.emit(originals, translations, boxes)

            except Exception as e:
                self.status_update.emit(f"错误: {e}")

            time.sleep(self.interval_ms / 1000.0)

    def capture_once(self) -> tuple[list, list, list, np.ndarray]:
        """执行一次截图+OCR+翻译（用于测试按钮），返回结果。"""
        img, rect = self.capture_region()
        ocr_results = self.core.ocr(img)

        originals, translations, boxes = [], [], []
        for text, (ox, oy, ow, oh) in ocr_results:
            abs_box = (
                rect.x() + ox,
                rect.y() + oy,
                ow, oh,
            )
            translated = self.core.translate(text)
            originals.append(text)
            translations.append(translated)
            boxes.append(abs_box)

        return originals, translations, boxes, img

    def stop(self):
        self._running = False

    def pause(self):
        self._paused = True
        self.status_update.emit("已暂停")

    def resume(self):
        self._paused = False
        self.status_update.emit("运行中")

    def toggle_pause(self):
        if self._paused:
            self.resume()
        else:
            self.pause()


# ═══════════════════════════════════════════════════════════════════
# 悬浮翻译窗
# ═══════════════════════════════════════════════════════════════════

class FloatingWindow(QWidget):
    """透明悬浮翻译窗：无边框、置顶、鼠标穿透。"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # 窗口属性
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint |
            Qt.FramelessWindowHint |
            Qt.Tool |
            Qt.WindowTransparentForInput  # 鼠标穿透
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        # 翻译数据
        self.originals: list[str] = []
        self.translations: list[str] = []
        self.boxes: list[tuple] = []

        # 显示设置
        self.font_size = 14
        self.font_color = QColor(255, 255, 255)        # 默认白色
        self.bg_opacity = 0.80                          # 背景透明度 80%

        # 窗口大小
        self.setMinimumSize(100, 40)

        # 定时刷新显示
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self.update)
        self._update_timer.start(200)

    @Slot(list, list, list)
    def update_texts(self, originals: list, translations: list, boxes: list):
        """从工作线程接收翻译结果并刷新。"""
        self.originals = originals
        self.translations = translations
        self.boxes = boxes
        self._reposition()

    def _reposition(self):
        """根据翻译区域重新定位悬浮窗。"""
        if not self.boxes:
            return
        # 悬浮窗覆盖所有检测框的范围
        min_x = min(b[0] for b in self.boxes)
        max_x = max(b[0] + b[2] for b in self.boxes)
        min_y = min(b[1] for b in self.boxes)
        max_y = max(b[1] + b[3] for b in self.boxes)

        margin = 10
        self.setGeometry(
            min_x - margin,
            max_y + 5,           # 显示在原区域下方
            max_x - min_x + margin * 2 + 200,
            (len(self.originals) * (self.font_size + 6) + 20) * 2 + 20
        )

    def paintEvent(self, event):
        """自绘文本和半透明背景。"""
        if not self.originals:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 半透明背景
        bg_color = QColor(30, 30, 30, int(self.bg_opacity * 255))
        painter.setBrush(QBrush(bg_color))
        painter.setPen(Qt.NoPen)
        r = 10  # 圆角半径
        painter.drawRoundedRect(self.rect(), r, r)

        # 绘制文本
        font = QFont()
        font.setPointSize(self.font_size)
        font.setStyleHint(QFont.SansSerif)
        painter.setFont(font)

        line_h = self.font_size + 6
        y = 15

        for i, (orig, trans) in enumerate(zip(self.originals, self.translations)):
            painter.setPen(QPen(QColor(200, 200, 200)))
            painter.drawText(10, y, self.width() - 20, line_h,
                             Qt.AlignLeft, orig)

            painter.setPen(QPen(self.font_color))
            painter.drawText(10, y + line_h, self.width() - 20, line_h,
                             Qt.AlignLeft, trans)

            y += line_h * 2 + 5

        painter.end()


# ═══════════════════════════════════════════════════════════════════
# 主窗口 UI
# ═══════════════════════════════════════════════════════════════════

STYLE_QSS = """
/* ── 全局 ── */
* {
    font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}

/* ── 主窗口 ── */
QMainWindow {
    background: #f5f5f7;
    border-radius: 12px;
}

/* ── 卡片式分组框 ── */
QGroupBox {
    background: #ffffff;
    border: 1px solid #e5e5ea;
    border-radius: 10px;
    margin-top: 16px;
    padding: 16px 14px 14px 14px;
    font-weight: 600;
    color: #1d1d1f;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: #6e6e73;
}

/* ── 按钮 ── */
QPushButton {
    background: #007aff;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 14px;
}
QPushButton:hover {
    background: #0066d6;
}
QPushButton:pressed {
    background: #0055b3;
}
QPushButton#stopBtn {
    background: #ff3b30;
}
QPushButton#stopBtn:hover {
    background: #d62d20;
}
QPushButton#testBtn {
    background: #f2f2f7;
    color: #007aff;
    border: 1px solid #c6c6c8;
}
QPushButton#testBtn:hover {
    background: #e5e5ea;
}

/* ── 下拉框 ── */
QComboBox {
    background: #ffffff;
    border: 1px solid #c6c6c8;
    border-radius: 6px;
    padding: 6px 10px;
    min-width: 120px;
}
QComboBox:hover {
    border-color: #007aff;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}

/* ── 滑块 ── */
QSlider::groove:horizontal {
    background: #e5e5ea;
    height: 4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #007aff;
    width: 18px;
    height: 18px;
    margin: -7px 0;
    border-radius: 9px;
}

/* ── 输入框 ── */
QLineEdit {
    background: #ffffff;
    border: 1px solid #c6c6c8;
    border-radius: 6px;
    padding: 6px 10px;
}
QLineEdit:focus {
    border-color: #007aff;
}

/* ── 状态栏 ── */
QStatusBar {
    background: #f5f5f7;
    color: #8e8e93;
    border-top: 1px solid #e5e5ea;
    padding: 4px 12px;
}

/* ── 自定义标题栏 ── */
#titleBar {
    background: transparent;
    padding: 8px 12px;
}
#titleLabel {
    font-size: 14px;
    font-weight: 600;
    color: #1d1d1f;
}
#closeBtn, #minBtn {
    background: transparent;
    border: none;
    padding: 4px 10px;
    font-size: 16px;
    color: #8e8e93;
    border-radius: 6px;
}
#closeBtn:hover {
    background: #ff3b30;
    color: #ffffff;
}
#minBtn:hover {
    background: #e5e5ea;
    color: #1d1d1f;
}

/* ── 数字输入框 ── */
QSpinBox {
    background: #ffffff;
    border: 1px solid #c6c6c8;
    border-radius: 6px;
    padding: 4px 8px;
    min-width: 70px;
}
QSpinBox:focus {
    border-color: #007aff;
}
"""


class MainWindow(QMainWindow):
    """主窗口：设置区域 + 控制按钮 + 状态栏。"""

    def __init__(self, app: QApplication):
        super().__init__()
        self._app = app

        # 核心引擎
        self.core = TranslationCore()
        self.worker: ScreenshotWorker | None = None
        self.floating: FloatingWindow | None = None

        # 翻译运行状态
        self._translating = False

        # ── 窗口基础设置 ──
        self.setWindowTitle("屏幕翻译")
        self.setFixedSize(480, 620)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # 窗口拖动
        self._drag_pos: QPoint | None = None

        # ── 构建 UI ──
        self._setup_ui()
        self._apply_style()

        # ── 系统托盘 ──
        self._setup_tray()

        # ── 全局热键 Ctrl+Shift+T ──
        if HAS_KEYBOARD:
            keyboard.add_hotkey("ctrl+shift+t", self._on_global_hotkey)

        # ── 悬浮窗 ──
        self.floating = FloatingWindow()

        # ── 显示主窗口 ──
        self.show()

    # ═══════════════════════════════════════════════════════════════
    # UI 构建
    # ═══════════════════════════════════════════════════════════════

    def _setup_ui(self):
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 0, 14, 14)
        root.setSpacing(10)

        # ── 自定义标题栏 ──
        title_bar = QWidget()
        title_bar.setObjectName("titleBar")
        tb_layout = QHBoxLayout(title_bar)
        tb_layout.setContentsMargins(8, 4, 8, 4)

        title_label = QLabel("屏幕翻译")
        title_label.setObjectName("titleLabel")

        min_btn = QPushButton("—")
        min_btn.setObjectName("minBtn")
        min_btn.setFixedSize(32, 28)
        min_btn.clicked.connect(self._minimize)

        close_btn = QPushButton("✕")
        close_btn.setObjectName("closeBtn")
        close_btn.setFixedSize(32, 28)
        close_btn.clicked.connect(self._on_close)

        tb_layout.addWidget(title_label)
        tb_layout.addStretch()
        tb_layout.addWidget(min_btn)
        tb_layout.addWidget(close_btn)
        root.addWidget(title_bar)

        # ── 设置区域 ──
        settings_group = QGroupBox("检测与引擎设置")
        sg_layout = QVBoxLayout(settings_group)

        # 检测模式
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("检测区域:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["鼠标周围区域", "自定义区域"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        sg_layout.addLayout(mode_row)

        # 自定义区域坐标
        self.fixed_grid = QGridLayout()
        self.fixed_grid.addWidget(QLabel("X:"), 0, 0)
        self.fixed_x = QSpinBox()
        self.fixed_x.setRange(0, 9999)
        self.fixed_grid.addWidget(self.fixed_x, 0, 1)
        self.fixed_grid.addWidget(QLabel("Y:"), 0, 2)
        self.fixed_y = QSpinBox()
        self.fixed_y.setRange(0, 9999)
        self.fixed_grid.addWidget(self.fixed_y, 0, 3)
        self.fixed_grid.addWidget(QLabel("宽:"), 0, 4)
        self.fixed_w = QSpinBox()
        self.fixed_w.setRange(100, 3000)
        self.fixed_w.setValue(400)
        self.fixed_grid.addWidget(self.fixed_w, 0, 5)
        self.fixed_grid.addWidget(QLabel("高:"), 0, 6)
        self.fixed_h = QSpinBox()
        self.fixed_h.setRange(100, 3000)
        self.fixed_h.setValue(300)
        self.fixed_grid.addWidget(self.fixed_h, 0, 7)
        # 默认隐藏
        for i in range(self.fixed_grid.count()):
            w = self.fixed_grid.itemAt(i).widget()
            if w:
                w.setVisible(False)
        sg_layout.addLayout(self.fixed_grid)

        # 鼠标模式宽高
        self.mouse_grid = QGridLayout()
        self.mouse_grid.addWidget(QLabel("检测宽度:"), 0, 0)
        self.mouse_w = QSpinBox()
        self.mouse_w.setRange(100, 2000)
        self.mouse_w.setValue(500)
        self.mouse_grid.addWidget(self.mouse_w, 0, 1)
        self.mouse_grid.addWidget(QLabel("检测高度:"), 0, 2)
        self.mouse_h = QSpinBox()
        self.mouse_h.setRange(100, 2000)
        self.mouse_h.setValue(300)
        self.mouse_grid.addWidget(self.mouse_h, 0, 3)
        sg_layout.addLayout(self.mouse_grid)

        # OCR 引擎
        ocr_row = QHBoxLayout()
        ocr_row.addWidget(QLabel("OCR 引擎:"))
        self.ocr_combo = QComboBox()
        self.ocr_combo.addItems(
            ["Tesseract" if HAS_TESSERACT else "Tesseract (未安装)",
             "EasyOCR" if HAS_EASYOCR else "EasyOCR (未安装)"]
        )
        ocr_row.addWidget(self.ocr_combo)
        ocr_row.addStretch()
        sg_layout.addLayout(ocr_row)

        # 翻译引擎
        trans_row = QHBoxLayout()
        trans_row.addWidget(QLabel("翻译引擎:"))
        self.trans_combo = QComboBox()
        items = ["Google 翻译 (免费)" if HAS_GOOGLE_TRANSLATE else "Google 翻译 (未安装)"]
        items.append("DeepL (需 API Key)")
        items.append("OpenAI (需 API Key)")
        self.trans_combo.addItems(items)
        trans_row.addWidget(self.trans_combo)
        trans_row.addStretch()
        sg_layout.addLayout(trans_row)

        # API key
        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("可选，使用 Google 翻译无需填写")
        key_row.addWidget(self.api_key_input)
        sg_layout.addLayout(key_row)

        root.addWidget(settings_group)

        # ── 翻译显示设置 ──
        display_group = QGroupBox("翻译显示设置")
        dg_layout = QVBoxLayout(display_group)

        # 字体大小
        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("字体大小:"))
        self.font_slider = QSlider(Qt.Horizontal)
        self.font_slider.setRange(10, 30)
        self.font_slider.setValue(14)
        self.font_label = QLabel("14")
        self.font_slider.valueChanged.connect(
            lambda v: self.font_label.setText(str(v))
        )
        font_row.addWidget(self.font_slider)
        font_row.addWidget(self.font_label)
        dg_layout.addLayout(font_row)

        # 字体颜色
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("字体颜色:"))
        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(32, 32)
        self.color_btn.setStyleSheet(
            "background: #ffffff; border: 2px solid #c6c6c8; border-radius: 16px;"
        )
        self.color_btn.clicked.connect(self._pick_color)
        self._current_font_color = QColor(255, 255, 255)
        color_row.addWidget(self.color_btn)
        color_row.addStretch()
        dg_layout.addLayout(color_row)

        # 背景透明度
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("悬浮窗透明度:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(80)
        self.opacity_label = QLabel("80%")
        self.opacity_slider.valueChanged.connect(
            lambda v: self.opacity_label.setText(f"{v}%")
        )
        opacity_row.addWidget(self.opacity_slider)
        opacity_row.addWidget(self.opacity_label)
        dg_layout.addLayout(opacity_row)

        root.addWidget(display_group)

        # ── 控制区域 ──
        ctrl_widget = QWidget()
        ctrl = QHBoxLayout(ctrl_widget)
        ctrl.setContentsMargins(0, 0, 0, 0)

        self.start_btn = QPushButton("开始翻译")
        self.start_btn.setMinimumHeight(48)
        self.start_btn.clicked.connect(self._toggle_translation)
        ctrl.addWidget(self.start_btn, 3)

        self.test_btn = QPushButton("测试截图区域")
        self.test_btn.setObjectName("testBtn")
        self.test_btn.setMinimumHeight(48)
        self.test_btn.clicked.connect(self._test_capture)
        ctrl.addWidget(self.test_btn, 1)

        root.addWidget(ctrl_widget)

        # ── 状态栏 ──
        self.status_bar = self.statusBar()
        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label)

        root.addStretch()

    def _apply_style(self):
        self.setStyleSheet(STYLE_QSS)

    # ═══════════════════════════════════════════════════════════════
    # 系统托盘
    # ═══════════════════════════════════════════════════════════════

    def _setup_tray(self):
        """创建系统托盘图标和右键菜单。"""
        self.tray = QSystemTrayIcon(self)
        # 使用简单的纯色图标
        pixmap = QPixmap(32, 32)
        pixmap.fill(QColor(0, 122, 255))
        self.tray.setIcon(QIcon(pixmap))
        self.tray.setToolTip("TL 屏幕翻译")

        # 右键菜单
        menu = QMenu()

        show_action = QAction("显示主窗口", self)
        show_action.triggered.connect(self._show_from_tray)
        menu.addAction(show_action)

        toggle_action = QAction("开始/停止翻译", self)
        toggle_action.triggered.connect(self._toggle_translation)
        menu.addAction(toggle_action)

        menu.addSeparator()

        quit_action = QAction("退出", self)
        quit_action.triggered.connect(self._quit_app)
        menu.addAction(quit_action)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    # ═══════════════════════════════════════════════════════════════
    # 窗口拖动
    # ═══════════════════════════════════════════════════════════════

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            # 只在标题栏区域可拖动
            if event.position().y() < 40:
                self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            else:
                self._drag_pos = None

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_pos = None

    # ═══════════════════════════════════════════════════════════════
    # 模式切换
    # ═══════════════════════════════════════════════════════════════

    def _on_mode_changed(self, idx: int):
        is_fixed = (idx == 1)  # "自定义区域"
        for i in range(self.fixed_grid.count()):
            w = self.fixed_grid.itemAt(i).widget()
            if w:
                w.setVisible(is_fixed)
        for i in range(self.mouse_grid.count()):
            w = self.mouse_grid.itemAt(i).widget()
            if w:
                w.setVisible(not is_fixed)

    # ═══════════════════════════════════════════════════════════════
    # 颜色选择
    # ═══════════════════════════════════════════════════════════════

    def _pick_color(self):
        color = QColorDialog.getColor(self._current_font_color, self, "选择悬浮窗字体颜色")
        if color.isValid():
            self._current_font_color = color
            self.color_btn.setStyleSheet(
                f"background: {color.name()}; border: 2px solid #c6c6c8; border-radius: 16px;"
            )

    # ═══════════════════════════════════════════════════════════════
    # 翻译控制
    # ═══════════════════════════════════════════════════════════════

    def _toggle_translation(self):
        if not self._translating:
            self._start_translation()
        else:
            self._stop_translation()

    def _start_translation(self):
        """启动实时翻译。"""
        # 更新 core 配置
        ocr_engine = "tesseract" if "Tesseract" in self.ocr_combo.currentText() else "easyocr"
        self.core.set_ocr_engine(ocr_engine)

        # 创建 worker
        self.worker = ScreenshotWorker(self.core)

        # 同步设置
        idx = self.mode_combo.currentIndex()
        self.worker.mode = "fixed" if idx == 1 else "mouse"
        self.worker.mouse_w = self.mouse_w.value()
        self.worker.mouse_h = self.mouse_h.value()
        self.worker.fixed_x = self.fixed_x.value()
        self.worker.fixed_y = self.fixed_y.value()
        self.worker.fixed_w = self.fixed_w.value()
        self.worker.fixed_h = self.fixed_h.value()

        # 悬浮窗显示设置
        self.floating.font_size = self.font_slider.value()
        self.floating.font_color = self._current_font_color
        self.floating.bg_opacity = self.opacity_slider.value() / 100.0

        # 连接信号
        self.worker.translation_ready.connect(self.floating.update_texts)
        self.worker.status_update.connect(self._on_status)

        # 显示悬浮窗
        self.floating.show()

        # 启动线程
        self.worker.start()

        # 更新 UI
        self._translating = True
        self.start_btn.setText("停止翻译")
        self.start_btn.setObjectName("stopBtn")
        self._apply_style()

    def _stop_translation(self):
        """停止实时翻译。"""
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
            self.worker = None

        if self.floating:
            self.floating.hide()

        self._translating = False
        self.start_btn.setText("开始翻译")
        self.start_btn.setObjectName("")
        self._apply_style()
        self.status_label.setText("就绪")

    def _test_capture(self):
        """测试截图区域：执行一次截图并在悬浮窗显示结果。"""
        # 临时创建 core 和 worker
        ocr_engine = "tesseract" if "Tesseract" in self.ocr_combo.currentText() else "easyocr"
        temp_core = TranslationCore(ocr_engine=ocr_engine)

        temp_worker = ScreenshotWorker(temp_core)
        idx = self.mode_combo.currentIndex()
        temp_worker.mode = "fixed" if idx == 1 else "mouse"
        temp_worker.mouse_w = self.mouse_w.value()
        temp_worker.mouse_h = self.mouse_h.value()
        temp_worker.fixed_x = self.fixed_x.value()
        temp_worker.fixed_y = self.fixed_y.value()
        temp_worker.fixed_w = self.fixed_w.value()
        temp_worker.fixed_h = self.fixed_h.value()

        self.status_label.setText("测试截图中...")
        QApplication.processEvents()

        try:
            originals, translations, boxes, img = temp_worker.capture_once()

            # 更新悬浮窗显示
            self.floating.font_size = self.font_slider.value()
            self.floating.font_color = self._current_font_color
            self.floating.bg_opacity = self.opacity_slider.value() / 100.0
            self.floating.update_texts(originals, translations, boxes)
            self.floating.show()

            if originals:
                self.status_label.setText(f"测试完成 — 检测到 {len(originals)} 段文本")
            else:
                self.status_label.setText("测试完成 — 未检测到文本，请调整区域")

        except Exception as e:
            self.status_label.setText(f"测试失败: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 窗口操作
    # ═══════════════════════════════════════════════════════════════

    def _on_close(self):
        """关闭按钮：最小化到托盘或退出。"""
        if self._translating:
            # 翻译运行时最小化到托盘
            self.hide()
            self.tray.showMessage(
                "TL 屏幕翻译", "翻译在后台继续运行",
                QSystemTrayIcon.Information, 2000
            )
        else:
            reply = QMessageBox.question(
                self, "退出", "确定要退出程序吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._quit_app()

    def _minimize(self):
        """最小化到托盘。"""
        self.hide()
        if self._translating:
            self.tray.showMessage(
                "TL 屏幕翻译", "翻译在后台运行中",
                QSystemTrayIcon.Information, 2000
            )

    def _show_from_tray(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            self._show_from_tray()

    def _on_global_hotkey(self):
        """全局热键 Ctrl+Shift+T 暂停/恢复。"""
        if self.worker and self._translating:
            self.worker.toggle_pause()
            state = "已暂停" if self.worker._paused else "运行中"
            self.status_label.setText(state)
            self.tray.showMessage("TL 屏幕翻译", state, QSystemTrayIcon.Information, 1000)

    @Slot(str)
    def _on_status(self, text: str):
        self.status_label.setText(text)

    def _quit_app(self):
        """完全退出。"""
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)
        if self.floating:
            self.floating.close()
        if HAS_KEYBOARD:
            try:
                keyboard.remove_hotkey("ctrl+shift+t")
            except Exception:
                pass
        self.tray.hide()
        QApplication.quit()

    def closeEvent(self, event):
        """窗口关闭事件 → 最小化到托盘。"""
        event.ignore()
        self._on_close()


# ═══════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════

def main():
    """应用入口。"""
    # 高 DPI 支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("TL")
    app.setApplicationDisplayName("TL — 屏幕翻译")
    app.setQuitOnLastWindowClosed(False)  # 关闭窗口不退出

    window = MainWindow(app)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
