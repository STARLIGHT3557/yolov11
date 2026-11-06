@echo off
chcp 65001 >nul
REM ============================================================
REM  YOLO 目标检测可视化程序 - Windows 一键启动脚本
REM  双击本文件即可运行，无需手动激活 conda 环境
REM ============================================================

REM 切换到脚本所在目录
cd /d "%~dp0"

REM 调用 miniconda 环境激活脚本
call C:\Users\AN040\miniconda3\Scripts\activate.bat yolo_39

echo.
echo ============================================
echo   YOLO 目标检测可视化系统
echo   环境: yolo_39 (Python 3.9 + PyTorch 2.7 + CUDA)
echo ============================================
echo.

python main.py

pause
