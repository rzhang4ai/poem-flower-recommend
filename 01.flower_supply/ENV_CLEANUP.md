# flower_env 里为什么有 3.13 和 3.14？如何只保留一个？

## 当前情况（检查结果）

| 位置 | Python 版本 | 用途 / 内容 |
|------|-------------|-------------|
| **flower_env/bin/python** | 指向 **3.13**（conda base 的 python3.13） | 你运行 `python`、`python crawl_brighten_hk.py` 时用的就是这个 |
| **flower_env/lib/python3.13/site-packages/** | 3.13 的包 | 项目用的包：flask, pandas, playwright, numpy 等（**应保留**） |
| **flower_env/lib/python3.14/site-packages/** | 3.14 的包 | 只有 playwright 及其依赖（greenlet, pyee 等），是之前误装到 3.14 的（**可删**） |
| **flower_env/bin/python3.14** | 3.14 可执行文件 | 存在时可能让某些命令用到 3.14（**可一并删**） |

## 为什么会同时存在 3.13 和 3.14？

常见原因：

1. **创建 venv 时用的解释器不一致**  
   例如先用 `python3.14 -m venv flower_env` 建了环境（于是有 lib/python3.14），后来系统或 conda 默认变成 3.13，`python` 指向 3.13，所以又出现了 lib/python3.13。

2. **PATH 里有多份 pip/python**  
   没使用 `python -m pip install`，而是直接打 `pip install`，可能用到了别的 Python 的 pip，把包装进了 3.14 的 site-packages。

3. **conda 与 venv 混用**  
   `flower_env/bin/python` 实际指向的是 conda base 的 `python3.13`，所以“当前环境”是 3.13，而 3.14 的目录是历史遗留。

## 建议：只保留 3.13，删掉 3.14

你当前跑脚本、用包的都是 **3.13**，所以可以安全地删掉 3.14 相关部分，避免以后再装错。

### 1. 删除 3.14 的包目录（误装的 playwright 等都在这里）

```bash
rm -rf /Users/rzhang/Documents/poem-flower-recommend/flower_env/lib/python3.14
```

### 2. 如有 3.14 的可执行文件，也删掉（避免误用）

```bash
rm -f /Users/rzhang/Documents/poem-flower-recommend/flower_env/bin/python3.14
```

### 3. 验证

```bash
cd /Users/rzhang/Documents/poem-flower-recommend/flower_supply
python check_env.py   # 应显示 3.13，且 playwright 导入成功
```

以后安装包一律用：

```bash
python -m pip install <包名>
```

这样一定会装到当前在用的 3.13，不会装到别的版本。
