"""
诗花雅送 · 数据审核工具 v3
Google Drive 协作版：每人负责不同月份，结果保存为独立文件，最后合并。

运行方式：
    python app.py --csv poems_dataset_v2.csv --ZHANG Rui
    python app.py --csv poems_dataset_v2.csv --WANG Dongni
    python app.py --csv poems_dataset_v2.csv --SHI Zhigang

合并所有审核结果：
    python app.py --merge --csv poems_dataset_v2.csv
"""

import argparse, csv, json, os, shutil, threading, glob
from datetime import datetime
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

CSV_PATH    = None
REVIEWER    = 'A'
ROWS        = []
LOCK        = threading.Lock()

MONTH_ORDER = {'正月':1,'二月':2,'三月':3,'四月':4,'五月':5,'六月':6,
               '七月':7,'八月':8,'九月':9,'十月':10,'十一月':11,'十二月':12}
MONTHS = list(MONTH_ORDER.keys())

FLOWER_WHITELIST = sorted([
    '梅花','红梅','蜡梅','迎春花','樱桃花','玉兰','月季','杏花','李花','桃花',
    '棠梨花','林檎花','郁李花','剪春罗','素馨','莱花','绣球','绣球花','木兰',
    '玫瑰','蔷薇','酴醿','芍药','牡丹','合欢','玉蕊花','玉簪花','含笑','辛夷',
    '木香','山茶花','山茶','瑞香','兰花','秋兰','石竹','杨花','木棉花','紫荆',
    '琼花','萱草','棣棠','棣花','金沙','木芙蓉','莲花','荷花','茉莉','凤仙花',
    '向日葵','木槿','金银花','秋海棠','芦花','桂花','菊花','甘菊','野菊',
    '金钱花','雁来红','秋葵','山丹','栀子花','紫薇','牵牛花','蜀葵','鸡冠花',
    '水仙花','水仙','蓼花','滴滴金','蘋花','山枇杷','杜鹃','桐花','凌霄花',
    '榴花','罂粟','曼陀罗','虞美人',
])
DYNASTIES = ['先秦','汉','两汉','东汉','西汉','魏','晋','东晋','西晋',
             '南朝梁','南朝陈','南朝宋','南朝齐','北朝周','北朝齐','北朝魏',
             '隋','唐','五代','宋','辽','金','元','明','清','近代','现代','当代']

# ── CSV 路径工具 ──────────────────────────────────────────────────────────────
def reviewer_csv_path():
    """每位审核员的独立输出文件，放在原CSV同目录"""
    base = os.path.splitext(CSV_PATH)[0]
    return f"{base}_reviewed_{REVIEWER}.csv"

def load_csv():
    global ROWS
    # 优先加载审核员自己的进度文件
    rpath = reviewer_csv_path()
    src = rpath if os.path.exists(rpath) else CSV_PATH
    with open(src, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault('审核状态', '')
        r.setdefault('审核备注', '')
        r.setdefault('审核员',   REVIEWER)
        r.setdefault('月份数字', str(MONTH_ORDER.get(r.get('月份',''), 0)))
    ROWS = rows
    print(f"✅ 已加载 {len(ROWS)} 条  来源: {src}")

def save_csv():
    rpath = reviewer_csv_path()
    # 备份
    if os.path.exists(rpath):
        shutil.copy2(rpath, rpath + '.bak')
    fieldnames = list(ROWS[0].keys()) if ROWS else []
    with open(rpath, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ROWS)

# ── 合并函数（CLI 调用）──────────────────────────────────────────────────────
def merge_all():
    base = os.path.splitext(CSV_PATH)[0]
    pattern = f"{base}_reviewed_*.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        print("❌ 没有找到审核文件"); return

    # 以原始CSV为底，按ID建字典
    with open(CSV_PATH, encoding='utf-8-sig') as f:
        original = {r['ID']: r for r in csv.DictReader(f)}

    merged = dict(original)  # shallow copy
    conflicts = []

    for fpath in files:
        reviewer_name = fpath.rsplit('_', 1)[-1].replace('.csv','')
        with open(fpath, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                rid = row['ID']
                if rid not in merged:
                    continue
                cur = merged[rid]
                # 检测冲突（同一字段不同审核员改了不同值）
                for field in ['诗名','朝代','作者','花名','月份','正文']:
                    orig_val = original[rid].get(field,'')
                    new_val  = row.get(field,'')
                    cur_val  = cur.get(field,'')
                    if new_val != orig_val:
                        if cur_val != orig_val and cur_val != new_val:
                            conflicts.append({
                                'ID': rid, 'field': field,
                                'original': orig_val,
                                'value_A': cur_val, 'reviewer_A': cur.get('审核员','?'),
                                'value_B': new_val,  'reviewer_B': reviewer_name,
                            })
                        else:
                            cur[field] = new_val
                # 合并审核状态：有人通过就通过，有人标记就标记
                statuses = [cur.get('审核状态',''), row.get('审核状态','')]
                if '⚑' in statuses:     cur['审核状态'] = '⚑'
                elif '✓' in statuses:   cur['审核状态'] = '✓'
                # 合并备注
                notes = [n for n in [cur.get('审核备注',''), row.get('审核备注','')] if n]
                cur['审核备注'] = ' | '.join(notes)
                cur['审核员'] = 'merged'

    # 输出合并结果
    out_path = base + '_merged.csv'
    fieldnames = list(next(iter(merged.values())).keys())
    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged.values())
    print(f"✅ 合并完成 → {out_path}")
    print(f"   处理文件: {[os.path.basename(p) for p in files]}")

    if conflicts:
        cpath = base + '_conflicts.csv'
        with open(cpath, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=conflicts[0].keys())
            writer.writeheader()
            writer.writerows(conflicts)
        print(f"⚠️  发现 {len(conflicts)} 处冲突 → {cpath}（需人工裁决）")
    else:
        print("   无冲突 🎉")

# ── API ───────────────────────────────────────────────────────────────────────
@app.route('/api/rows')
def api_rows():
    month  = request.args.get('month', '')
    status = request.args.get('status', '')
    with LOCK:
        rows = ROWS
    result = []
    for i, r in enumerate(rows):
        if month and r.get('月份','') != month: continue
        if status == 'approved' and r.get('审核状态') != '✓': continue
        if status == 'flagged'  and r.get('审核状态') != '⚑': continue
        if status == 'pending'  and r.get('审核状态','') not in ('',None): continue
        result.append({'_idx': i, **r})
    return jsonify(result)

@app.route('/api/stats')
def api_stats():
    with LOCK: rows = ROWS
    total    = len(rows)
    approved = sum(1 for r in rows if r.get('审核状态') == '✓')
    flagged  = sum(1 for r in rows if r.get('审核状态') == '⚑')
    by_month = {}
    for r in rows:
        m = r.get('月份','未知')
        if m not in by_month:
            by_month[m] = {'total':0,'approved':0,'flagged':0,'pending':0}
        by_month[m]['total'] += 1
        s = r.get('审核状态','')
        if s == '✓':   by_month[m]['approved'] += 1
        elif s == '⚑': by_month[m]['flagged']  += 1
        else:           by_month[m]['pending']  += 1
    return jsonify({'total':total,'approved':approved,'flagged':flagged,
                    'pending':total-approved-flagged,'by_month':by_month,
                    'reviewer': REVIEWER})

@app.route('/api/update', methods=['POST'])
def api_update():
    data  = request.json
    idx   = int(data['idx'])
    field = data['field']
    value = data['value']
    allowed = {'诗名','朝代','作者','花名','月份','月份数字','正文','赏析','审核状态','审核备注'}
    if field not in allowed:
        return jsonify({'ok': False, 'error': 'not allowed'}), 400
    with LOCK:
        ROWS[idx][field] = value
        if field == '月份':
            ROWS[idx]['月份数字'] = str(MONTH_ORDER.get(value, 0))
        ROWS[idx]['审核员'] = REVIEWER
        save_csv()
    return jsonify({'ok': True})

@app.route('/api/batch', methods=['POST'])
def api_batch():
    data    = request.json
    indices = [int(i) for i in data['indices']]
    status  = data['status']
    note    = data.get('note', '')
    with LOCK:
        for i in indices:
            ROWS[i]['审核状态'] = status
            ROWS[i]['审核员']   = REVIEWER
            if note:
                existing = ROWS[i].get('审核备注','')
                ROWS[i]['审核备注'] = (existing + ' ' + note).strip()
        save_csv()
    return jsonify({'ok': True, 'updated': len(indices)})

@app.route('/api/options')
def api_options():
    return jsonify({'flowers': FLOWER_WHITELIST, 'dynasties': DYNASTIES,
                    'months': MONTHS, 'reviewer': REVIEWER})



@app.route('/api/delete_row', methods=['POST'])
def api_delete_row():
    """删除指定行（仅允许删除新建的空白行，或强制删除）"""
    data  = request.json
    idx   = int(data['idx'])
    force = data.get('force', False)
    with LOCK:
        if idx < 0 or idx >= len(ROWS):
            return jsonify({'ok': False, 'error': 'index out of range'}), 400
        row = ROWS[idx]
        # 安全检查：非强制模式下只允许删除正文和赏析都为空的行
        if not force:
            has_content = (row.get('正文','').strip() or row.get('赏析','').strip()
                          or row.get('诗名','').strip())
            if has_content:
                return jsonify({'ok': False, 'error': 'row has content, use force=true'}), 400
        ROWS.pop(idx)
        save_csv()
    return jsonify({'ok': True})

@app.route('/api/add_row', methods=['POST'])
def api_add_row():
    """在指定行之后插入一条新空白行，继承月份和花名"""
    data      = request.json
    after_idx = int(data.get('after_idx', len(ROWS) - 1))
    inherit   = data.get('inherit', {})
    with LOCK:
        try:
            max_id = max(int(r.get('ID', 0)) for r in ROWS)
        except Exception:
            max_id = len(ROWS)
        fieldnames = list(ROWS[0].keys()) if ROWS else [
            'ID','月份','月份数字','花名','诗名','朝代','作者','正文','赏析','审核状态','审核备注','审核员']
        new_row = {f: '' for f in fieldnames}
        new_row['ID']       = str(max_id + 1)
        new_row['月份']     = inherit.get('月份', '')
        new_row['月份数字'] = str(MONTH_ORDER.get(inherit.get('月份',''), 0))
        new_row['花名']     = inherit.get('花名', '')
        new_row['朝代']     = inherit.get('朝代', '')
        new_row['审核状态'] = ''
        new_row['审核员']   = REVIEWER
        ROWS.insert(after_idx + 1, new_row)
        save_csv()
    return jsonify({'ok': True, 'new_idx': after_idx + 1, 'new_id': new_row['ID']})

@app.route('/')
def index():
    return render_template_string(HTML)

# ── 前端 HTML ─────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>诗花雅送 · 数据审核</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --ink:#1a1209;--paper:#faf6ef;--aged:#f0e8d8;--faded:#b8a888;
  --border:#ddd0b8;--accent:#8b3a2a;--green:#2d6a4f;--amber:#b5621e;
  --serif:'Noto Serif SC',serif;--mono:'JetBrains Mono',monospace;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:var(--serif);background:var(--paper);color:var(--ink);display:flex;height:100vh;overflow:hidden;}

/* ── sidebar ── */
#sidebar{width:190px;min-width:190px;background:var(--aged);border-right:1px solid var(--border);display:flex;flex-direction:column;overflow-y:auto;}
#sidebar-top{padding:16px 16px 10px;border-bottom:1px solid var(--border);}
#sidebar-top h2{font-size:14px;font-weight:700;}
#reviewer-badge{display:inline-block;margin-top:6px;padding:2px 10px;border-radius:12px;font-size:11px;font-family:var(--mono);background:var(--accent);color:#fff;}
.month-item{border-bottom:1px solid var(--border);}
.month-btn{padding:9px 14px 4px;cursor:pointer;font-family:var(--serif);font-size:13px;border:none;background:none;text-align:left;width:100%;color:var(--ink);display:flex;justify-content:space-between;align-items:center;transition:background .12s;}
.month-btn:hover{background:#e8dece;}
.month-btn.active{background:var(--accent);color:#fff;}
.month-btn.active .mprog{color:rgba(255,255,255,.65);}
.mprog{font-size:10px;font-family:var(--mono);color:var(--faded);}
.mbar-wrap{height:2px;background:var(--border);margin:0 14px 6px;}
.mbar{height:2px;background:var(--green);transition:width .3s;}
.month-done .month-btn{color:var(--green);}
#sidebar-foot{margin-top:auto;padding:10px 14px;font-size:11px;color:var(--faded);border-top:1px solid var(--border);line-height:1.7;}
.gbar-wrap{height:3px;background:var(--border);border-radius:2px;margin-top:5px;}
.gbar{height:3px;background:var(--green);border-radius:2px;transition:width .3s;}

/* ── main ── */
#main{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0;}

/* ── toolbar ── */
#toolbar{padding:8px 14px;background:var(--aged);border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
#toolbar h1{font-size:14px;font-weight:700;}
.flt-btn{padding:3px 11px;border:1px solid var(--border);border-radius:16px;background:none;cursor:pointer;font-family:var(--serif);font-size:12px;color:var(--ink);transition:all .12s;}
.flt-btn:hover{background:var(--border);}
.flt-btn.active{background:var(--ink);color:var(--paper);border-color:var(--ink);}
.act-btn{padding:3px 11px;border-radius:16px;border:1px solid;cursor:pointer;font-family:var(--serif);font-size:12px;transition:all .12s;}
#btn-sel-all{border-color:var(--faded);background:none;color:var(--ink);}
#btn-sel-all:hover{background:var(--border);}
#btn-approve{border-color:var(--green);background:none;color:var(--green);}
#btn-approve:hover{background:var(--green);color:#fff;}
#btn-flag{border-color:var(--amber);background:none;color:var(--amber);}
#btn-flag:hover{background:var(--amber);color:#fff;}
#btn-clr{border-color:var(--faded);background:none;color:var(--faded);}
#btn-clr:hover{background:var(--faded);color:#fff;}
#sel-count{font-size:11px;font-family:var(--mono);color:var(--faded);margin-left:auto;}
#search-box{padding:3px 10px;border:1px solid var(--border);border-radius:16px;font-family:var(--serif);font-size:12px;background:#fff;outline:none;width:130px;}
#search-box:focus{border-color:var(--accent);}

/* ── resizable table wrapper ── */
#table-wrap{flex:1;overflow:auto;position:relative;}

table{border-collapse:collapse;font-size:13px;table-layout:fixed;width:100%;}

/* column resize handle */
col{}
thead th{
  position:sticky;top:0;background:var(--aged);border-bottom:2px solid var(--border);
  padding:8px 6px 8px 8px;text-align:left;font-weight:600;font-size:11px;
  letter-spacing:.06em;color:var(--faded);white-space:nowrap;z-index:10;
  overflow:hidden;user-select:none;
}
thead th .th-inner{display:flex;align-items:center;gap:2px;}
thead th .resizer{
  width:5px;min-width:5px;height:100%;cursor:col-resize;
  position:absolute;right:0;top:0;bottom:0;
  background:transparent;z-index:11;
}
thead th .resizer:hover,thead th .resizer.dragging{background:var(--accent);opacity:.4;}
thead th{position:sticky;top:0;z-index:10;}

tbody tr{border-bottom:1px solid var(--border);transition:background .08s;}
tbody tr:hover{background:#f5ede0;}
tbody tr.sel{background:#fde8d0;}
tbody tr.approved{border-left:3px solid var(--green);}
tbody tr.flagged{border-left:3px solid var(--amber);}
tbody tr.new-row{background:#f0f8ff;}

td{padding:6px 8px;vertical-align:top;overflow:hidden;}
td.cb{width:32px;min-width:32px;text-align:center;vertical-align:middle;}
td.st{width:44px;min-width:44px;text-align:center;vertical-align:middle;}
.sbadge{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;border-radius:50%;font-size:12px;cursor:pointer;transition:transform .1s;}
.sbadge:hover{transform:scale(1.2);}
.sbadge.approved{background:#d4edda;}
.sbadge.flagged{background:#fde8c8;}
.sbadge.pending{background:var(--border);}
td.id-cell{font-family:var(--mono);font-size:10px;color:var(--faded);vertical-align:middle;}

/* data cells */
.dc{cursor:pointer;display:block;word-break:break-all;}
.dc:hover{color:var(--accent);}
/* short fields: single line */
td.short-cell .dc{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
/* text/analysis: full multiline display */
td.txt-cell{vertical-align:top;}
td.txt-cell .dc{font-size:12px;line-height:1.7;color:#444;white-space:pre-wrap;word-break:break-all;}
td.ana-cell{vertical-align:top;}
td.ana-cell .dc{font-size:12px;line-height:1.7;color:#666;white-space:pre-wrap;word-break:break-all;}

/* inline select */
.inline-sel{font-family:var(--serif);font-size:12px;background:#fff;border:1px solid var(--accent);border-radius:3px;padding:1px 2px;outline:none;width:100%;}

/* new row btn inside table */
.add-row-btn,.del-row-btn{
  display:none;
  background:none;border:1px dashed var(--border);border-radius:3px;
  font-size:11px;cursor:pointer;padding:1px 6px;
  font-family:var(--serif);transition:all .12s;white-space:nowrap;
}
tbody tr:hover .add-row-btn,tbody tr:hover .del-row-btn{display:inline-block;}
.add-row-btn{color:var(--faded);}
.add-row-btn:hover{border-color:var(--accent);color:var(--accent);}
.del-row-btn{color:#c0392b;border-color:#e8b4b4;margin-left:3px;}
.del-row-btn:hover{background:#c0392b;color:#fff;border-color:#c0392b;}

/* ── drawer ── */
#drawer{
  position:fixed;right:0;top:0;bottom:0;width:520px;
  background:var(--paper);border-left:1px solid var(--border);
  box-shadow:-6px 0 24px rgba(0,0,0,.1);
  transform:translateX(100%);transition:transform .22s ease;
  z-index:100;display:flex;flex-direction:column;
}
#drawer.open{transform:translateX(0);}
#dh{padding:12px 16px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;background:var(--aged);}
#dh h3{font-size:14px;font-weight:700;}
#dh-meta{font-size:11px;color:var(--faded);font-family:var(--mono);margin-top:2px;}
#d-close{background:none;border:none;font-size:18px;cursor:pointer;color:var(--faded);}
#d-close:hover{color:var(--ink);}
#db{flex:1;overflow-y:auto;padding:16px;}
.fg{margin-bottom:14px;}
.fl{font-size:10px;font-weight:700;letter-spacing:.12em;color:var(--faded);text-transform:uppercase;margin-bottom:4px;}
textarea.de{width:100%;font-family:var(--serif);font-size:13px;line-height:1.8;border:1px solid var(--border);border-radius:4px;padding:7px 10px;background:#fff;resize:vertical;outline:none;transition:border-color .12s;}
textarea.de:focus{border-color:var(--accent);}
textarea.de.tall{min-height:140px;}
textarea.de.short{min-height:56px;}
select.ds,input.di{font-family:var(--serif);font-size:13px;border:1px solid var(--border);border-radius:4px;padding:5px 9px;background:#fff;width:100%;outline:none;transition:border-color .12s;}
select.ds:focus,input.di:focus{border-color:var(--accent);}
.row2{display:flex;gap:10px;}.row2>.fg{flex:1;}
#df{padding:10px 16px;border-top:1px solid var(--border);display:flex;gap:8px;background:var(--aged);}
.btn{flex:1;padding:7px 0;border-radius:4px;border:none;cursor:pointer;font-family:var(--serif);font-size:13px;font-weight:600;transition:opacity .12s;}
.btn:hover{opacity:.82;}
.btn-g{background:var(--green);color:#fff;}
.btn-a{background:var(--amber);color:#fff;}
.btn-k{background:var(--ink);color:#fff;}
.btn-c{background:var(--border);color:var(--ink);flex:0 0 auto;padding:7px 14px;}

/* ── split helper in drawer ── */
#split-hint{font-size:11px;color:var(--faded);margin-top:4px;line-height:1.5;}

/* ── shortcuts ── */
#shortcuts{position:fixed;bottom:14px;right:14px;background:rgba(26,18,9,.8);color:rgba(250,246,239,.7);font-size:10px;font-family:var(--mono);padding:7px 11px;border-radius:6px;line-height:1.9;pointer-events:none;z-index:50;}

/* ── toast ── */
#toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%) translateY(60px);background:var(--ink);color:var(--paper);padding:8px 18px;border-radius:6px;font-size:12px;transition:transform .25s;z-index:200;pointer-events:none;}
#toast.show{transform:translateX(-50%) translateY(0);}
</style>
</head>
<body>

<!-- sidebar -->
<div id="sidebar">
  <div id="sidebar-top">
    <h2>诗花雅送审核</h2>
    <span id="reviewer-badge">审核员 ?</span>
  </div>
  <div id="month-list"></div>
  <div id="sidebar-foot">
    <div id="gstat">载入中…</div>
    <div class="gbar-wrap"><div class="gbar" id="gbar"></div></div>
    <div style="margin-top:8px;font-size:10px;opacity:.7">自动保存<br>_reviewed_?.csv</div>
  </div>
</div>

<!-- main -->
<div id="main">
  <div id="toolbar">
    <h1>数据审核</h1>
    <button class="flt-btn active" data-s="">全部</button>
    <button class="flt-btn" data-s="pending">待审</button>
    <button class="flt-btn" data-s="approved">已通过</button>
    <button class="flt-btn" data-s="flagged">已标记</button>
    <span style="width:1px;height:16px;background:var(--border)"></span>
    <button class="act-btn" id="btn-sel-all">全选</button>
    <button class="act-btn" id="btn-approve">批量通过 ✓</button>
    <button class="act-btn" id="btn-flag">批量标记 ⚑</button>
    <button class="act-btn" id="btn-clr">清除</button>
    <input type="text" id="search-box" placeholder="搜索诗名/作者…">
    <span id="sel-count"></span>
  </div>

  <div id="table-wrap">
    <table id="main-table">
      <colgroup>
        <col id="col-cb"     style="width:32px;min-width:32px">
        <col id="col-st"     style="width:44px;min-width:44px">
        <col id="col-id"     style="width:44px;min-width:44px">
        <col id="col-title"  style="width:10%">
        <col id="col-dyn"    style="width:6%">
        <col id="col-auth"   style="width:6%">
        <col id="col-flower" style="width:7%">
        <col id="col-month"  style="width:5%">
        <col id="col-text"   style="width:22%">
        <col id="col-ana"    style="width:26%">
        <col id="col-note"   style="width:8%">
        <col id="col-acts"   style="width:72px;min-width:72px">
      </colgroup>
      <thead>
        <tr>
          <th class="cb"><input type="checkbox" id="chk-all"></th>
          <th>状态<div class="resizer" data-col="col-st"></div></th>
          <th>ID<div class="resizer" data-col="col-id"></div></th>
          <th>诗名<div class="resizer" data-col="col-title"></div></th>
          <th>朝代<div class="resizer" data-col="col-dyn"></div></th>
          <th>作者<div class="resizer" data-col="col-auth"></div></th>
          <th>花名<div class="resizer" data-col="col-flower"></div></th>
          <th>月份<div class="resizer" data-col="col-month"></div></th>
          <th>正文<div class="resizer" data-col="col-text"></div></th>
          <th>赏析<div class="resizer" data-col="col-ana"></div></th>
          <th>备注<div class="resizer" data-col="col-note"></div></th>
          <th></th>
        </tr>
      </thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</div>

<!-- drawer -->
<div id="drawer">
  <div id="dh">
    <div>
      <h3 id="d-title">详情</h3>
      <div id="dh-meta"></div>
    </div>
    <button id="d-close">✕</button>
  </div>
  <div id="db"></div>
  <div id="df">
    <button class="btn btn-g" id="d-ok">通过 ✓</button>
    <button class="btn btn-a" id="d-flag">标记 ⚑</button>
    <button class="btn btn-k" id="d-save">保存修改</button>
    <button class="btn btn-c" id="d-cancel">关闭</button>
  </div>
</div>

<div id="shortcuts">A: 批量通过 &nbsp;F: 批量标记 &nbsp;N: 新建行 &nbsp;Esc: 关闭</div>
<div id="toast"></div>

<script>
let allRows=[], curMonth='正月', curFilter='', selectedIdx=new Set(), drawerIdx=null, opts={}, searchQ='';
const MONTHS=['正月','二月','三月','四月','五月','六月','七月','八月','九月','十月','十一月','十二月'];

async function init(){
  opts=await fetch('/api/options').then(r=>r.json());
  document.getElementById('reviewer-badge').textContent='审核员 '+opts.reviewer;
  initResizers();
  await refreshStats();
  selectMonth('正月');
}

/* ── 列宽拖拽 ───────────────────────────────────────────────────────────────── */
function initResizers(){
  document.querySelectorAll('.resizer').forEach(r=>{
    let startX, startW, col;
    r.addEventListener('mousedown', e=>{
      e.preventDefault();
      col=document.getElementById(r.dataset.col);
      startX=e.clientX;
      startW=col.offsetWidth||parseInt(col.style.width)||100;
      r.classList.add('dragging');
      const onMove=e2=>{
        const w=Math.max(40, startW+(e2.clientX-startX));
        col.style.width=w+'px';
      };
      const onUp=()=>{
        r.classList.remove('dragging');
        document.removeEventListener('mousemove',onMove);
        document.removeEventListener('mouseup',onUp);
      };
      document.addEventListener('mousemove',onMove);
      document.addEventListener('mouseup',onUp);
    });
  });
}

/* ── 月份侧栏 ── */
async function refreshStats(){
  const s=await fetch('/api/stats').then(r=>r.json());
  const ml=document.getElementById('month-list');
  ml.innerHTML='';
  MONTHS.forEach(m=>{
    const ms=s.by_month[m]||{total:0,approved:0,flagged:0,pending:0};
    const pct=ms.total?Math.round(ms.approved/ms.total*100):0;
    const done=ms.pending===0&&ms.total>0;
    const wrap=document.createElement('div');
    wrap.className='month-item'+(done?' month-done':'');
    wrap.innerHTML=`
      <button class="month-btn${m===curMonth?' active':''}" data-m="${m}">
        <span>${m}</span><span class="mprog">${ms.approved}/${ms.total}</span>
      </button>
      <div class="mbar-wrap"><div class="mbar" style="width:${pct}%"></div></div>`;
    wrap.querySelector('.month-btn').onclick=()=>selectMonth(m);
    ml.appendChild(wrap);
  });
  const tot=s.total,apr=s.approved;
  document.getElementById('gstat').textContent=
    `总计 ${tot}  通过 ${apr}\n标记 ${s.flagged}  待审 ${s.pending}`;
  document.getElementById('gbar').style.width=(tot?Math.round(apr/tot*100):0)+'%';
}

function selectMonth(m){
  curMonth=m; selectedIdx.clear();
  document.querySelectorAll('.month-btn').forEach(b=>b.classList.toggle('active',b.dataset.m===m));
  loadRows();
}

/* ── 筛选 / 搜索 ── */
document.querySelectorAll('.flt-btn').forEach(b=>{
  b.onclick=()=>{
    document.querySelectorAll('.flt-btn').forEach(x=>x.classList.remove('active'));
    b.classList.add('active'); curFilter=b.dataset.s; selectedIdx.clear(); loadRows();
  };
});
document.getElementById('search-box').oninput=e=>{searchQ=e.target.value.trim();loadRows();};

/* ── 加载表格 ── */
async function loadRows(){
  const url=`/api/rows?month=${encodeURIComponent(curMonth)}&status=${curFilter}`;
  let rows=await fetch(url).then(r=>r.json());
  if(searchQ){
    const q=searchQ.toLowerCase();
    rows=rows.filter(r=>(r['诗名']||'').includes(q)||(r['作者']||'').includes(q)||(r['花名']||'').includes(q));
  }
  allRows=rows; renderTable(); updateSelCount();
}

function renderTable(){
  const tb=document.getElementById('tbody');
  tb.innerHTML='';
  allRows.forEach(row=>{
    const idx=row._idx;
    const st=row['审核状态']||'';
    const stCls=st==='✓'?'approved':st==='⚑'?'flagged':'pending';
    const isNew=!row['正文']&&!row['赏析']&&!row['诗名'];
    const trCls=(selectedIdx.has(idx)?'sel ':'')+(st==='✓'?'approved':st==='⚑'?'flagged':'')+(isNew?' new-row':'');
    const tr=document.createElement('tr');
    tr.className=trCls; tr.dataset.idx=idx;
    tr.innerHTML=`
      <td class="cb"><input type="checkbox" class="rcb" data-idx="${idx}" ${selectedIdx.has(idx)?'checked':''}></td>
      <td class="st"><span class="sbadge ${stCls}" data-idx="${idx}">${st==='✓'?'✓':st==='⚑'?'⚑':'·'}</span></td>
      <td class="id-cell">${esc(row['ID']||'')}</td>
      <td>${mkCell(idx,'诗名',row['诗名'])}</td>
      <td>${mkSelect(idx,'朝代',row['朝代'],opts.dynasties)}</td>
      <td>${mkCell(idx,'作者',row['作者'])}</td>
      <td>${mkSelect(idx,'花名',row['花名'],opts.flowers)}</td>
      <td>${mkSelect(idx,'月份',row['月份'],opts.months)}</td>
      <td class="txt-cell">${mkCell(idx,'正文',row['正文'],true)}</td>
      <td class="ana-cell">${mkCell(idx,'赏析',row['赏析'],true)}</td>
      <td class="short-cell">${mkCell(idx,'审核备注',row['审核备注'])}</td>
      <td>
        <button class="add-row-btn" data-idx="${idx}" title="在此行后新建一行">＋行</button>
        <button class="del-row-btn" data-idx="${idx}" title="删除此行">✕</button>
      </td>`;
    tb.appendChild(tr);
  });
  tb.querySelectorAll('.rcb').forEach(cb=>cb.onchange=()=>togSel(parseInt(cb.dataset.idx),cb.checked));
  tb.querySelectorAll('.sbadge').forEach(b=>b.onclick=e=>{e.stopPropagation();cycleStatus(parseInt(b.dataset.idx));});
  tb.querySelectorAll('.dc[data-idx]').forEach(dc=>{
    dc.onclick=()=>openDrawer(parseInt(dc.dataset.idx));
  });
  tb.querySelectorAll('.add-row-btn').forEach(btn=>btn.onclick=e=>{
    e.stopPropagation();
    addRow(parseInt(btn.dataset.idx));
  });
  tb.querySelectorAll('.del-row-btn').forEach(btn=>btn.onclick=e=>{
    e.stopPropagation();
    deleteRow(parseInt(btn.dataset.idx));
  });
}

function mkCell(idx, field, val, fullText=false){
  const v=val||'';
  if(fullText){
    return `<span class="dc" data-idx="${idx}" data-f="${field}">${esc(v)}</span>`;
  }
  return `<span class="dc" data-idx="${idx}" data-f="${field}" title="${esc(v)}">${esc(v.slice(0,40))}${v.length>40?'…':''}</span>`;
}
function mkSelect(idx,field,val,opts2){
  return `<select class="inline-sel" data-idx="${idx}" data-f="${field}" onchange="saveF(this.dataset.idx,this.dataset.f,this.value)">
    ${(opts2||[]).map(o=>`<option${o===val?' selected':''}>${esc(o)}</option>`).join('')}</select>`;
}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

/* ── 保存单字段 ── */
async function saveF(idx,field,value){
  idx=parseInt(idx);
  await fetch('/api/update',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({idx,field,value})});
  const row=allRows.find(r=>r._idx===idx);
  if(row)row[field]=value;
  toast(`已保存 · ${field}`);
  refreshStats();
}

/* ── 状态循环 ── */
async function cycleStatus(idx){
  const row=allRows.find(r=>r._idx===idx);
  if(!row)return;
  const next=(!row['审核状态']||row['审核状态']==='')? '✓':row['审核状态']==='✓'?'⚑':'';
  await saveF(idx,'审核状态',next);
  loadRows();
}

/* ── 全选 / 批量 ── */
document.getElementById('chk-all').onchange=e=>{
  allRows.forEach(r=>e.target.checked?selectedIdx.add(r._idx):selectedIdx.delete(r._idx));
  renderTable(); updateSelCount();
};
document.getElementById('btn-sel-all').onclick=()=>{
  allRows.forEach(r=>selectedIdx.add(r._idx)); renderTable(); updateSelCount();
};
function togSel(idx,on){
  on?selectedIdx.add(idx):selectedIdx.delete(idx); updateSelCount();
  const tr=document.querySelector(`tr[data-idx="${idx}"]`);
  if(tr)tr.classList.toggle('sel',on);
}
function updateSelCount(){
  const n=selectedIdx.size;
  document.getElementById('sel-count').textContent=n?`已选 ${n} 条`:'';
}
async function batchSet(st){
  if(!selectedIdx.size){toast('请先勾选条目');return;}
  await fetch('/api/batch',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({indices:[...selectedIdx],status:st})});
  const lbl=st==='✓'?'通过':st==='⚑'?'标记':'清除';
  toast(`批量${lbl} ${selectedIdx.size} 条`);
  selectedIdx.clear(); await loadRows(); refreshStats();
}
document.getElementById('btn-approve').onclick=()=>batchSet('✓');
document.getElementById('btn-flag').onclick=()=>batchSet('⚑');
document.getElementById('btn-clr').onclick=()=>batchSet('');

/* ── 新建行 ── */
async function addRow(afterIdx){
  const row=allRows.find(r=>r._idx===afterIdx);
  const inherit={
    月份: row?row['月份']:'',
    花名: row?row['花名']:'',
    朝代: row?row['朝代']:'',
  };
  const res=await fetch('/api/add_row',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({after_idx:afterIdx, inherit})}).then(r=>r.json());
  toast(`新建行 ID=${res.new_id}`);
  await loadRows();
  // 自动打开新行的抽屉
  const newRow=allRows.find(r=>r._idx===res.new_idx);
  if(newRow) openDrawer(res.new_idx);
}

/* ── 删除行 ── */
async function deleteRow(idx){
  const row=allRows.find(r=>r._idx===idx);
  if(!row)return;
  const hasContent=(row['正文']||'').trim()||(row['赏析']||'').trim()||(row['诗名']||'').trim();
  if(hasContent){
    if(!confirm(`确认删除「${row['诗名']||'此行'}」？\n此行有内容，删除后不可恢复。`))return;
  }
  const force=!!hasContent;
  const res=await fetch('/api/delete_row',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({idx,force})}).then(r=>r.json());
  if(res.ok){
    toast(`已删除 ID=${row['ID']}`);
    await loadRows(); refreshStats();
  } else {
    toast('删除失败: '+res.error);
  }
}

/* ── 详情抽屉 ── */
function openDrawer(idx){
  const row=allRows.find(r=>r._idx===idx);
  if(!row)return;
  drawerIdx=idx;
  document.getElementById('d-title').textContent=row['诗名']||'（新建行）';
  document.getElementById('dh-meta').textContent=`ID ${row['ID']} · ${row['朝代']||'—'}·${row['作者']||'—'}`;
  const db=document.getElementById('db');
  db.innerHTML=`
    <div class="fg"><div class="fl">诗名</div>
      <input class="di" id="di-title" value="${esc(row['诗名']||'')}"></div>
    <div class="row2">
      <div class="fg"><div class="fl">朝代</div>
        <select class="ds" id="di-dyn">${opts.dynasties.map(d=>`<option${d===row['朝代']?' selected':''}>${esc(d)}</option>`).join('')}</select></div>
      <div class="fg"><div class="fl">作者</div>
        <input class="di" id="di-auth" value="${esc(row['作者']||'')}"></div>
    </div>
    <div class="row2">
      <div class="fg"><div class="fl">花名</div>
        <select class="ds" id="di-flower">${opts.flowers.map(f=>`<option${f===row['花名']?' selected':''}>${esc(f)}</option>`).join('')}</select></div>
      <div class="fg"><div class="fl">月份</div>
        <select class="ds" id="di-month">${opts.months.map(m=>`<option${m===row['月份']?' selected':''}>${esc(m)}</option>`).join('')}</select></div>
    </div>
    <div class="fg">
      <div class="fl">正文</div>
      <textarea class="de tall" id="di-text">${esc(row['正文']||'')}</textarea>
    </div>
    <div class="fg">
      <div class="fl">赏析（完整内容 · 可编辑）</div>
      <textarea class="de tall" id="di-ana">${esc(row['赏析']||'')}</textarea>
      <div id="split-hint">💡 如赏析里混入了另一首诗的正文，可复制后关闭此窗口，点击该行「＋行」新建一行粘贴。</div>
    </div>
    <div class="fg"><div class="fl">审核备注</div>
      <textarea class="de short" id="di-note">${esc(row['审核备注']||'')}</textarea></div>
    <div class="fg" style="font-size:11px;color:var(--faded)">
      状态: ${row['审核状态']||'待审'} &nbsp;·&nbsp; 审核员: ${row['审核员']||'—'}
    </div>`;
  document.getElementById('drawer').classList.add('open');
}

document.getElementById('d-close').onclick=closeDrawer;
document.getElementById('d-cancel').onclick=closeDrawer;
function closeDrawer(){document.getElementById('drawer').classList.remove('open');drawerIdx=null;}

document.getElementById('d-save').onclick=async()=>{
  if(drawerIdx===null)return;
  const fields={'诗名':'di-title','朝代':'di-dyn','作者':'di-auth','花名':'di-flower',
                '月份':'di-month','正文':'di-text','赏析':'di-ana','审核备注':'di-note'};
  for(const[f,id] of Object.entries(fields)){
    const el=document.getElementById(id);
    if(el) await fetch('/api/update',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({idx:drawerIdx,field:f,value:el.value})});
  }
  toast('保存成功'); closeDrawer(); await loadRows(); refreshStats();
};
document.getElementById('d-ok').onclick=async()=>{
  if(drawerIdx===null)return;
  await saveF(drawerIdx,'审核状态','✓'); closeDrawer(); await loadRows();
};
document.getElementById('d-flag').onclick=async()=>{
  if(drawerIdx===null)return;
  await saveF(drawerIdx,'审核状态','⚑'); closeDrawer(); await loadRows();
};

/* ── 键盘 ── */
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA'||e.target.tagName==='SELECT')return;
  if(e.key==='Escape') closeDrawer();
  if(e.key==='a'||e.key==='A') batchSet('✓');
  if(e.key==='f'||e.key==='F') batchSet('⚑');
  if(e.key==='n'||e.key==='N'){
    // 新建行：在当前月份末尾
    const last=allRows[allRows.length-1];
    if(last) addRow(last._idx);
  }
});

/* ── Toast ── */
let tt=null;
function toast(msg){const el=document.getElementById('toast');el.textContent=msg;el.classList.add('show');clearTimeout(tt);tt=setTimeout(()=>el.classList.remove('show'),2200);}

init();
</script>
</body>
</html>
"""

# ── 入口 ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='诗花雅送数据审核工具')
    parser.add_argument('--csv',      default='poems_dataset_v4.csv')
    parser.add_argument('--reviewer', default='A', help='审核员标识 (A/B/C/名字)')
    parser.add_argument('--port',     type=int, default=5000)
    parser.add_argument('--merge',    action='store_true', help='合并所有审核员结果')
    args = parser.parse_args()

    CSV_PATH = os.path.abspath(args.csv)
    if not os.path.exists(CSV_PATH):
        print(f'❌ 找不到文件: {CSV_PATH}'); exit(1)

    if args.merge:
        merge_all(); exit(0)

    REVIEWER = args.reviewer
    load_csv()

    rpath = reviewer_csv_path()
    print(f'✅ 已加载 {len(ROWS)} 条')
    print(f'👤 审核员: {REVIEWER}')
    print(f'💾 保存至: {rpath}')
    print(f'🌐 浏览器打开: http://127.0.0.1:{args.port}')
    app.run(host='127.0.0.1', port=args.port, debug=False)
