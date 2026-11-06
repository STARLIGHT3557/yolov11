# -*- coding: utf-8 -*-
"""
基于 Ultralytics YOLO 的目标检测可视化程序
================================================
课程：深度学习综合实践
功能：
    1. 导入图片 / 视频 进行目标检测
    2. 实时显示检测结果（边框 + 类别 + 置信度）
    3. 右侧面板显示各类别检测数量
    4. 视频检测结束后自动导出完整带框视频到 output/
    5. 任意时刻可点"保存当前截图"按钮保存当前帧为图片
    6. 可调置信度阈值、IOU 阈值
    7. 自动使用 GPU（CUDA），无 GPU 时回退 CPU
运行环境：conda 环境 yolo_39
"""

import sys
import os
import time
from datetime import datetime

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from ultralytics import YOLO

# ----------------------------------------------------------------------
# 路径配置
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 默认模型权重（首次运行会自动从 ultralytics 官方仓库下载到 weights/）
DEFAULT_WEIGHTS = os.path.join(WEIGHTS_DIR, "yolo11n.pt")


# ======================================================================
# 检测线程（避免阻塞 GUI 主线程）
# ======================================================================
class DetectionThread(QtCore.QThread):
    """视频/图片检测线程"""
    frame_ready = QtCore.pyqtSignal(np.ndarray, dict)     # 一帧 BGR 图像 + 统计
    finished = QtCore.pyqtSignal()                          # 检测结束
    log_signal = QtCore.pyqtSignal(str)                    # 日志
    video_saved = QtCore.pyqtSignal(str)                   # 完整视频保存完成，传回路径

    def __init__(self, source, model, conf=0.35, iou=0.45, is_video=True):
        super().__init__()
        self.source = source
        self.model = model
        self.conf = conf
        self.iou = iou
        self.is_video = is_video
        self._running = True
        self._paused = False

    def run(self):
        if self.is_video:
            self._run_video()
        else:
            self._run_image()
        self.finished.emit()

    def _run_image(self):
        """单张图片检测"""
        self.log_signal.emit(f"开始检测图片：{self.source}")
        frame = cv2.imdecode(np.fromfile(self.source, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.log_signal.emit("错误：无法读取图片")
            return
        t0 = time.time()
        results = self.model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)
        dt = time.time() - t0
        annotated = results[0].plot()
        stats = self._collect_stats(results[0])
        stats["fps"] = 1.0 / dt if dt > 0 else 0
        stats["frame_count"] = 1
        self.frame_ready.emit(annotated, stats)
        self.log_signal.emit(f"图片检测完成，耗时 {dt:.3f}s，检测到 {stats['total']} 个目标")

    def _run_video(self):
        """视频流检测：边检测边写入输出视频"""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.log_signal.emit("错误：无法打开视频")
            return

        # 读取原视频参数
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 生成输出视频文件名（中文路径用 imencode + 写文件方式，这里直接用英文路径避免编码问题）
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if isinstance(self.source, str):
            base = os.path.splitext(os.path.basename(self.source))[0]
        else:
            base = "camera"
        out_name = f"detected_{base}_{ts}.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        # 用 mp4v 编码器写出
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, src_fps, (src_w, src_h))
        if not writer.isOpened():
            # 回退到 avi
            out_path = out_path.replace(".mp4", ".avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(out_path, fourcc, src_fps, (src_w, src_h))

        self.log_signal.emit(f"开始检测视频，共 {total} 帧，原视频 FPS={src_fps:.1f}")
        self.log_signal.emit(f"检测结果将实时写入：{out_path}")

        frame_idx = 0
        while self._running:
            if self._paused:
                self.msleep(30)
                continue
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            t0 = time.time()
            results = self.model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)
            dt = time.time() - t0
            annotated = results[0].plot()
            # 写入输出视频
            writer.write(annotated)
            stats = self._collect_stats(results[0])
            stats["fps"] = 1.0 / dt if dt > 0 else 0
            stats["frame_count"] = frame_idx
            stats["total_frames"] = total
            self.frame_ready.emit(annotated, stats)

        cap.release()
        writer.release()
        self.log_signal.emit(f"视频检测结束，共处理 {frame_idx} 帧")
        self.video_saved.emit(out_path)

    @staticmethod
    def _collect_stats(result):
        stats = {"classes": {}, "total": 0}
        if result.boxes is None or len(result.boxes) == 0:
            return stats
        names = result.names
        for cls_id in result.boxes.cls.cpu().numpy().astype(int):
            name = names[cls_id]
            stats["classes"][name] = stats["classes"].get(name, 0) + 1
        stats["total"] = int(sum(stats["classes"].values()))
        return stats

    def pause(self):
        self._paused = not self._paused

    def stop(self):
        self._running = False
        self._paused = False


# ======================================================================
# 主窗口
# ======================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 目标检测可视化系统  (环境: yolo_39)")
        self.resize(1280, 800)

        self.model = None
        self.det_thread = None
        self.current_frame = None       # 当前显示的 BGR 帧（用于截图）
        self.current_source = None
        self.is_video = False

        self._build_ui()
        self._load_model()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # 左侧：视频显示区
        left_layout = QtWidgets.QVBoxLayout()
        self.video_label = QtWidgets.QLabel("请导入图片或视频开始检测")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet(
            "QLabel{background:#1e1e1e;color:#888;font-size:18px;border:2px dashed #555;}"
        )
        left_layout.addWidget(self.video_label, stretch=1)

        self.status_label = QtWidgets.QLabel("就绪")
        self.status_label.setStyleSheet("color:#333;padding:4px;")
        left_layout.addWidget(self.status_label)

        main_layout.addLayout(left_layout, stretch=4)

        # 右侧控制面板
        right_panel = QtWidgets.QWidget()
        right_panel.setFixedWidth(320)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setSpacing(10)

        # 1. 文件操作
        gb_file = QtWidgets.QGroupBox("文件操作")
        gb_file_layout = QtWidgets.QVBoxLayout(gb_file)
        self.btn_img = QtWidgets.QPushButton("导入图片")
        self.btn_video = QtWidgets.QPushButton("导入视频")
        self.btn_cam = QtWidgets.QPushButton("打开摄像头")
        self.btn_stop = QtWidgets.QPushButton("停止")
        self.btn_stop.setEnabled(False)
        for b in (self.btn_img, self.btn_video, self.btn_cam, self.btn_stop):
            b.setMinimumHeight(36)
            gb_file_layout.addWidget(b)
        right_layout.addWidget(gb_file)

        # 2. 检测参数
        gb_param = QtWidgets.QGroupBox("检测参数")
        gb_param_layout = QtWidgets.QFormLayout(gb_param)
        self.slider_conf = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_conf.setRange(1, 99)
        self.slider_conf.setValue(35)
        self.lbl_conf = QtWidgets.QLabel("0.35")
        self.slider_conf.valueChanged.connect(
            lambda v: self.lbl_conf.setText(f"{v/100:.2f}"))
        gb_param_layout.addRow("置信度阈值", self.slider_conf)
        gb_param_layout.addRow("", self.lbl_conf)

        self.slider_iou = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_iou.setRange(1, 99)
        self.slider_iou.setValue(45)
        self.lbl_iou = QtWidgets.QLabel("0.45")
        self.slider_iou.valueChanged.connect(
            lambda v: self.lbl_iou.setText(f"{v/100:.2f}"))
        gb_param_layout.addRow("IOU 阈值", self.slider_iou)
        gb_param_layout.addRow("", self.lbl_iou)
        right_layout.addWidget(gb_param)

        # 3. 权重选择
        gb_weight = QtWidgets.QGroupBox("模型权重")
        gb_weight_layout = QtWidgets.QVBoxLayout(gb_weight)
        self.weight_combo = QtWidgets.QComboBox()
        self._refresh_weight_list()
        self.btn_refresh = QtWidgets.QPushButton("刷新权重列表")
        self.btn_load_weight = QtWidgets.QPushButton("加载选中权重")
        gb_weight_layout.addWidget(self.weight_combo)
        gb_weight_layout.addWidget(self.btn_refresh)
        gb_weight_layout.addWidget(self.btn_load_weight)
        right_layout.addWidget(gb_weight)

        # 4. 检测结果统计
        gb_result = QtWidgets.QGroupBox("检测结果统计")
        gb_result_layout = QtWidgets.QVBoxLayout(gb_result)
        self.result_list = QtWidgets.QListWidget()
        gb_result_layout.addWidget(self.result_list)
        right_layout.addWidget(gb_result, stretch=2)

        # 5. 保存按钮（两个）
        self.btn_save_img = QtWidgets.QPushButton("保存当前截图（图片）")
        self.btn_save_img.setMinimumHeight(36)
        self.btn_save_img.setStyleSheet(
            "QPushButton{background:#2d7dd2;color:white;font-weight:bold;}"
            "QPushButton:hover{background:#1e6bb8;}")
        right_layout.addWidget(self.btn_save_img)

        self.lbl_video_hint = QtWidgets.QLabel(
            "视频检测时会自动导出完整带框视频\n到 output/ 目录")
        self.lbl_video_hint.setStyleSheet("color:#666;font-size:11px;")
        self.lbl_video_hint.setAlignment(QtCore.Qt.AlignCenter)
        right_layout.addWidget(self.lbl_video_hint)

        # 6. 日志
        gb_log = QtWidgets.QGroupBox("运行日志")
        gb_log_layout = QtWidgets.QVBoxLayout(gb_log)
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        gb_log_layout.addWidget(self.log_text)
        right_layout.addWidget(gb_log)

        main_layout.addWidget(right_panel, stretch=1)

        # 信号绑定
        self.btn_img.clicked.connect(self.open_image)
        self.btn_video.clicked.connect(self.open_video)
        self.btn_cam.clicked.connect(self.open_camera)
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_save_img.clicked.connect(self.save_screenshot)
        self.btn_refresh.clicked.connect(self._refresh_weight_list)
        self.btn_load_weight.clicked.connect(self._on_load_weight)

    def _load_model(self):
        self._log(f"正在加载模型：{DEFAULT_WEIGHTS}")
        try:
            self.model = YOLO(DEFAULT_WEIGHTS)
            import torch
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False)
            device = "GPU(CUDA)" if torch.cuda.is_available() else "CPU"
            self._log(f"模型加载完成，使用 {device} 推理")
        except Exception as e:
            self._log(f"模型加载失败：{e}")
            QtWidgets.QMessageBox.critical(
                self, "错误",
                f"模型加载失败：\n{e}\n\n请检查网络，或把 .pt 权重放入 weights/ 目录")

    def _refresh_weight_list(self):
        self.weight_combo.clear()
        if not os.path.isdir(WEIGHTS_DIR):
            return
        pts = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".pt")]
        self.weight_combo.addItems(pts)
        idx = self.weight_combo.findText("yolo11n.pt")
        if idx >= 0:
            self.weight_combo.setCurrentIndex(idx)

    def _on_load_weight(self):
        name = self.weight_combo.currentText()
        if not name:
            return
        path = os.path.join(WEIGHTS_DIR, name)
        self._log(f"切换模型权重：{path}")
        try:
            self.model = YOLO(path)
            self._log("模型加载成功")
        except Exception as e:
            self._log(f"加载失败：{e}")

    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp)")
        if path:
            self._start_detection(path, is_video=False)

    def open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频 (*.mp4 *.avi *.mov *.mkv *.flv)")
        if path:
            self._start_detection(path, is_video=True)

    def open_camera(self):
        self._start_detection(0, is_video=True)

    def _start_detection(self, source, is_video):
        if self.model is None:
            QtWidgets.QMessageBox.warning(self, "提示", "模型尚未加载完成")
            return
        self.stop_detection()

        self.current_source = source
        self.is_video = is_video
        conf = self.slider_conf.value() / 100.0
        iou = self.slider_iou.value() / 100.0

        self.det_thread = DetectionThread(source, self.model, conf, iou, is_video)
        self.det_thread.frame_ready.connect(self._on_frame)
        self.det_thread.log_signal.connect(self._log)
        self.det_thread.finished.connect(self._on_thread_finished)
        self.det_thread.video_saved.connect(self._on_video_saved)
        self.det_thread.start()

        self.btn_stop.setEnabled(True)
        self._log("检测已启动")

    def stop_detection(self):
        if self.det_thread and self.det_thread.isRunning():
            self.det_thread.stop()
            self.det_thread.wait(2000)
        self.btn_stop.setEnabled(False)

    def _on_frame(self, frame, stats):
        self.current_frame = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        pix = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = pix.scaled(
            self.video_label.width(), self.video_label.height(),
            QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(pix))

        self.result_list.clear()
        if stats.get("total", 0) == 0:
            self.result_list.addItem("当前帧未检测到目标")
        else:
            for cls_name, cnt in sorted(
                stats["classes"].items(), key=lambda x: -x[1]):
                self.result_list.addItem(f"  {cls_name} ：{cnt} 个")
            self.result_list.addItem("----------------")
            self.result_list.addItem(f"  合计：{stats['total']} 个")

        fps = stats.get("fps", 0)
        if self.is_video and "frame_count" in stats:
            total = stats.get("total_frames", 0)
            self.status_label.setText(
                f"FPS={fps:.1f}    帧 {stats['frame_count']}/{total}    "
                f"检测到 {stats.get('total',0)} 个目标")
        else:
            self.status_label.setText(
                f"推理耗时 {1.0/fps:.3f}s    检测到 {stats.get('total',0)} 个目标")

    def _on_thread_finished(self):
        self.btn_stop.setEnabled(False)
        self._log("检测线程结束")

    def _on_video_saved(self, path):
        """视频检测结束，自动导出完整视频后回调"""
        self._log(f"完整视频已自动保存：{path}")
        QtWidgets.QMessageBox.information(
            self, "视频保存成功",
            f"带检测框的完整视频已自动保存到：\n{path}")

    def save_screenshot(self):
        """保存当前帧为图片（任意时刻可点）"""
        if self.current_frame is None:
            QtWidgets.QMessageBox.warning(self, "提示", "当前没有可保存的画面")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.is_video:
            fname = f"screenshot_{ts}.jpg"
        else:
            base = os.path.splitext(os.path.basename(self.current_source))[0]
            fname = f"detected_{base}_{ts}.jpg"
        save_path = os.path.join(OUTPUT_DIR, fname)
        ok, buf = cv2.imencode(".jpg", self.current_frame)
        if not ok:
            self._log("保存失败：编码错误")
            return
        buf.tofile(save_path)
        self._log(f"截图已保存：{save_path}")
        QtWidgets.QMessageBox.information(
            self, "保存成功", f"截图已保存到：\n{save_path}")

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()


def main():
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
