#!/usr/bin/env python3
"""在终端运行，用于检查当前 Python 与 playwright 是否一致。"""
import sys
import subprocess

print("=== 环境检查 ===\n")
print("1. 当前 Python 路径:", sys.executable)
print("2. Python 版本:", sys.version)
print("3. 当前 pip 对应的 Python:")
subprocess.run([sys.executable, "-m", "pip", "--version"], check=False)
print("\n4. 尝试导入 playwright:")
try:
    from playwright.sync_api import sync_playwright
    print("   ✓ 导入成功")
except Exception as e:
    print(f"   ✗ 失败: {type(e).__name__}: {e}")
print("\n若导入失败，请用「上面显示的当前 Python」对应的 pip 安装：")
print(f"  {sys.executable} -m pip install playwright")
print("  playwright install chromium")
