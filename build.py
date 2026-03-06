from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import numpy as np

DATA_DIR = Path('/Users/kosuke/Dropbox/11_space-tech/02_satellite-data/00_Inbox/day1')
DIST_DIR = Path('dist')
IMAGES_DIR = DIST_DIR / 'images'

TASK_DIRS = {
    'L1': DATA_DIR / 'task1_1_ndvi_threshold',
    'L2': DATA_DIR / 'task1_2_resnet50_inference',
    'L3': DATA_DIR / 'task1_3_resnet50_finetune',
    'CMP': DATA_DIR / 'task1_4_comparison',
    'ALL': DATA_DIR / 'task1_5_all_models',
}


def load_npz(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def as_float(value, default=np.nan):
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def pct_ratio(value) -> str:
    v = as_float(value)
    if np.isnan(v):
        return 'N/A'
    return f'{v * 100:.1f}%'


def esc(text) -> str:
    return html.escape(str(text))


def table_html(headers: list[str], rows: list[list]) -> str:
    thead = ''.join(f'<th class="px-4 py-3 text-left text-sm font-semibold text-slate-200">{esc(h)}</th>' for h in headers)
    body_rows = []
    for row in rows:
        tds = ''.join(f'<td class="px-4 py-3 text-sm text-slate-300 align-top border-t border-slate-700">{esc(c)}</td>' for c in row)
        body_rows.append(f'<tr>{tds}</tr>')
    tbody = ''.join(body_rows)
    return (
        '<div class="overflow-x-auto rounded-xl border border-slate-700 bg-slate-900/60">'
        f'<table class="min-w-full">'
        f'<thead class="bg-slate-800/70"><tr>{thead}</tr></thead>'
        f'<tbody>{tbody}</tbody>'
        '</table></div>'
    )


def code_block(code: str, language: str = 'python') -> str:
    return (
        '<div class="rounded-xl border border-slate-700 bg-slate-950/90 shadow-lg overflow-hidden">'
        f'<div class="px-4 py-2 text-xs uppercase tracking-wider text-cyan-300 bg-slate-900/80 border-b border-slate-700">{esc(language)}</div>'
        f'<pre class="p-4 text-sm text-slate-200 overflow-x-auto"><code>{esc(code)}</code></pre>'
        '</div>'
    )


def info_box(text: str) -> str:
    return f'<div class="rounded-xl border border-sky-600/50 bg-sky-900/20 p-4 text-sky-100">{esc(text)}</div>'


def warning_box(text: str) -> str:
    return f'<div class="rounded-xl border border-amber-600/50 bg-amber-900/20 p-4 text-amber-100">{esc(text)}</div>'


def details_block(title: str, inner_html: str) -> str:
    return (
        '<details class="group rounded-xl border border-slate-700 bg-slate-900/60">'
        f'<summary class="cursor-pointer list-none px-4 py-3 text-slate-100 font-medium flex items-center justify-between">{esc(title)}'
        '<span class="text-slate-400 group-open:rotate-180 transition">⌄</span></summary>'
        f'<div class="px-4 pb-4 pt-1 text-slate-300">{inner_html}</div>'
        '</details>'
    )


def section_card(title: str, body_html: str) -> str:
    return (
        '<section class="rounded-2xl border border-slate-700 bg-slate-800/40 shadow-xl p-6 space-y-4">'
        f'<h3 class="text-xl font-semibold text-slate-100">{esc(title)}</h3>'
        f'{body_html}'
        '</section>'
    )


def copy_images() -> dict[str, str]:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    name_map: dict[str, str] = {}
    used_dest: set[str] = set()
    for task_dir in TASK_DIRS.values():
        if not task_dir.exists():
            continue
        for src in sorted(task_dir.glob('*.png')):
            dest_name = src.name
            if dest_name in used_dest:
                dest_name = f'{task_dir.name}_{src.name}'
            used_dest.add(dest_name)
            dest_path = IMAGES_DIR / dest_name
            shutil.copy2(src, dest_path)
            name_map[src.name] = f'images/{dest_name}'
    return name_map


def image_block(image_map: dict[str, str], file_name: str, title: str, what: str, insight: str) -> str:
    if file_name in image_map:
        img = (
            '<div class="rounded-2xl border border-slate-700 bg-slate-900/40 p-4 space-y-3">'
            f'<h4 class="text-lg font-semibold text-slate-100">{esc(title)}</h4>'
            f'<img src="{esc(image_map[file_name])}" alt="{esc(title)}" class="w-full rounded-xl shadow-2xl ring-1 ring-slate-700/60 cursor-zoom-in lightbox-trigger" loading="lazy">'
            f'<p class="text-sm text-slate-300"><span class="text-cyan-300">何を見るべきか:</span> {esc(what)}</p>'
            f'<p class="text-sm text-slate-300"><span class="text-emerald-300">何がわかるか:</span> {esc(insight)}</p>'
            '</div>'
        )
        return img
    return (
        '<div class="rounded-2xl border border-amber-600/50 bg-amber-900/20 p-4 space-y-2">'
        f'<h4 class="text-lg font-semibold text-amber-100">{esc(title)}</h4>'
        f'<p class="text-amber-200">画像なし: {esc(file_name)}</p>'
        f'<p class="text-sm text-amber-100/90">何を見るべきか: {esc(what)}</p>'
        f'<p class="text-sm text-amber-100/90">何がわかるか: {esc(insight)}</p>'
        '</div>'
    )


def overview_plot(l1, l2, l3):
    l1_acc = as_float(l1.get('accuracy') if l1 else np.nan)
    l2_acc10 = as_float(l2.get('accuracy_10class') if l2 else np.nan)
    l2_acc3 = as_float(l2.get('accuracy_3class') if l2 else np.nan)
    l3_acc10 = as_float(l3.get('accuracy_10class_full') if l3 else np.nan)
    l3_acc3 = as_float(l3.get('accuracy_3class') if l3 else np.nan)

    return {
        'data': [
            {
                'type': 'bar',
                'name': '3クラス精度',
                'x': ['L1', 'L2', 'L3'],
                'y': [l1_acc * 100, l2_acc3 * 100, l3_acc3 * 100],
                'marker': {'color': ['#7f8c8d', '#2980b9', '#27ae60']},
                'text': [f'{l1_acc * 100:.1f}', f'{l2_acc3 * 100:.1f}', f'{l3_acc3 * 100:.1f}'],
                'textposition': 'outside',
            },
            {
                'type': 'bar',
                'name': '10クラス精度',
                'x': ['L1', 'L2', 'L3'],
                'y': [None, l2_acc10 * 100, l3_acc10 * 100],
                'marker': {'color': ['#bdc3c7', '#3498db', '#2ecc71']},
                'text': ['N/A', f'{l2_acc10 * 100:.1f}', f'{l3_acc10 * 100:.1f}'],
                'textposition': 'outside',
            },
        ],
        'layout': {
            'barmode': 'group',
            'title': 'L1→L2→L3 精度サマリ',
            'yaxis': {'title': 'Accuracy (%)'},
            'height': 460,
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(15,23,42,0.4)',
            'font': {'color': '#e2e8f0'},
        },
    }


def task13_plot(l3):
    head = np.array(l3.get('hist_head_test_acc', []), dtype=float)
    full = np.array(l3.get('hist_full_test_acc', []), dtype=float)
    half = np.array(l3.get('hist_half_test_acc', []), dtype=float)
    if len(head) == 0:
        epochs = []
    else:
        epochs = list(range(1, len(head) + 1))
    return {
        'data': [
            {
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Head Only',
                'x': epochs,
                'y': (head * 100).tolist(),
                'line': {'color': '#e67e22', 'width': 3},
            },
            {
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Full FT',
                'x': epochs,
                'y': (full * 100).tolist(),
                'line': {'color': '#27ae60', 'width': 3},
            },
            {
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': '50% Data FT',
                'x': epochs,
                'y': (half * 100).tolist(),
                'line': {'color': '#2980b9', 'width': 3},
            },
        ],
        'layout': {
            'title': '3実験のテスト精度推移',
            'xaxis': {'title': 'Epoch'},
            'yaxis': {'title': 'Test Accuracy (%)', 'range': [85, 100]},
            'height': 460,
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(15,23,42,0.4)',
            'font': {'color': '#e2e8f0'},
        },
    }


def task15_plot():
    models = [
        ('L1: Index Threshold', 'L1', 64.4),
        ('Random Forest', 'Classical ML', 90.2),
        ('SVM (RBF)', 'Classical ML', 93.8),
        ('k-NN (k=5)', 'Classical ML', 90.3),
        ('Simple CNN', 'Deep Learning', 94.9),
        ('L2: ResNet50 + LogReg', 'L2', 95.7),
        ('L3: ResNet50 FT', 'L3', 98.5),
        ('ViT-Tiny FT', 'Deep Learning', 98.5),
    ]
    models = sorted(models, key=lambda x: x[2])
    color_map = {
        'L1': '#c0392b',
        'Classical ML': '#f39c12',
        'L2': '#2980b9',
        'L3': '#27ae60',
        'Deep Learning': '#8e44ad',
    }
    return {
        'data': [
            {
                'type': 'bar',
                'orientation': 'h',
                'x': [m[2] for m in models],
                'y': [m[0] for m in models],
                'marker': {'color': [color_map[m[1]] for m in models]},
                'text': [f'{m[2]:.1f}%' for m in models],
                'textposition': 'outside',
                'customdata': [m[1] for m in models],
                'hovertemplate': '%{y}<br>精度: %{x:.1f}%<br>カテゴリ: %{customdata}<extra></extra>',
            }
        ],
        'layout': {
            'title': '全モデル精度ランキング',
            'xaxis': {'title': 'Accuracy (%)'},
            'yaxis': {'title': ''},
            'height': 520,
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(15,23,42,0.4)',
            'font': {'color': '#e2e8f0'},
        },
    }


def build_html(l1, l2, l3, allm, image_map):
    l1_acc = as_float(l1.get('accuracy') if l1 else np.nan)
    l2_acc10 = as_float(l2.get('accuracy_10class') if l2 else np.nan)
    l3_acc10 = as_float(l3.get('accuracy_10class_full') if l3 else np.nan)

    overview_plot_json = json.dumps(overview_plot(l1, l2, l3), ensure_ascii=False)
    task13_plot_json = json.dumps(task13_plot(l3 or {}), ensure_ascii=False)
    task15_plot_json = json.dumps(task15_plot(), ensure_ascii=False)

    allm_table = ''
    if allm is not None:
        rows = []
        names = list(allm['model_names'])
        accs = list(allm['model_accs'])
        kinds = list(allm['model_types'])
        for n, a, k in zip(names, accs, kinds):
            rows.append([str(n), f'{float(a) * 100:.2f}', str(k)])
        allm_table = details_block(
            'all_models_results.npz の実測値',
            table_html(['モデル', '精度', 'タイプ'], rows),
        )

    overview_lessons = ''.join(
        details_block(f'学び {i}', f'<p>{esc(t)}</p>')
        for i, t in enumerate([
            '学習済みモデルを使うだけで L1→L2 で +31.3pt の最大ジャンプ。',
            'ファインチューニング（L2→L3）で +2.8pt。追加コストに対して着実な上積み。',
            'Head Only (92.4%) は L2 (95.7%) を下回る。凍結特徴 + LogReg の最適化効率が高い。',
            'EuroSATのようなきれいなデータでは学習データ50%でも -0.5pt。実務データでは差が拡大しやすい。',
            '実務のボトルネックは画像取得ではなくラベル作成コスト（セグメンテーション1枚30分〜1時間）。',
        ], 1)
    )

    task11_code = """# 1) データ読み込み
rgb, nir, swir = load_sentinel2_bands(...)

# 2) 指数計算
ndvi = (nir - red) / (nir + red + 1e-8)
ndwi = (green - nir) / (green + nir + 1e-8)
ndbi = (swir - nir) / (swir + nir + 1e-8)

# 3) 閾値探索（NDWI）
for th in np.arange(-0.20, 0.30, 0.02):
    pred = rule_based_predict(ndvi_mean, ndwi_mean, ndbi_mean, th)
    score = accuracy(pred, y_true_3)

# 4) 分類と評価
best_th = -0.18
pred = rule_based_predict(..., best_th)
report = confusion_matrix_and_class_accuracy(pred, y_true_3, y_true_10)"""

    task12_code = """# 1) ResNet50で特徴抽出（weights=ImageNet）
features_train = backbone(train_images)   # (21600, 2048)
features_test  = backbone(test_images)    # (5400, 2048)

# 2) ロジスティック回帰で10クラス分類
clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
clf.fit(features_train, y_train)
y_pred = clf.predict(features_test)

# 3) 評価
acc10 = accuracy_score(y_test, y_pred)
acc3  = accuracy_score(y_test_3, y_pred_3)"""

    task13_code = """# === Head Only（FC層のみ学習） ===
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False          # 全層凍結
model.fc = nn.Linear(2048, 10)           # 最終層だけ差し替え

optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# === Full Fine-tuning（全層学習） ===
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(2048, 10)           # 最終層差し替え
# 全パラメータが requires_grad=True（デフォルト）

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

# === 学習ループ（共通） ===
for epoch in range(10):
    model.train()
    for images, labels in train_loader:          # batch_size=64
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    # テスト評価
    model.eval()
    correct = sum((model(x.to(device)).argmax(1) == y.to(device)).sum()
                  for x, y in test_loader)
    test_acc = correct / len(test_dataset)

# === データ拡張（学習時） ===
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])"""

    task14_code = """# === L1/L2/L3 の結果を読み込んで比較 ===
l1 = np.load('L1_results.npz', allow_pickle=True)
l2 = np.load('L2_results.npz', allow_pickle=True)
l3 = np.load('L3_results.npz', allow_pickle=True)

# 精度テーブル作成
results = {
    'L1': {'3class': l1['accuracy'], '10class': None},
    'L2': {'3class': l2['accuracy_3class'], '10class': l2['accuracy_10class']},
    'L3_head': {'10class': l3['accuracy_10class_head']},
    'L3_full': {'3class': l3['accuracy_3class'], '10class': l3['accuracy_10class_full']},
    'L3_half': {'10class': l3['accuracy_10class_half']},
}

# 混同行列の比較（L2 vs L3）
from sklearn.metrics import confusion_matrix, classification_report
cm_l2 = confusion_matrix(y_test, l2['predictions'])
cm_l3 = confusion_matrix(y_test, l3['predictions_full'])

# クラス別F1スコアでレーダーチャート
report_l2 = classification_report(y_test, l2['predictions'], output_dict=True)
report_l3 = classification_report(y_test, l3['predictions_full'], output_dict=True)

# 同一画像での予測比較
for idx in sample_indices:
    print(f"True: {class_names[y_test[idx]]}")
    print(f"  L1: {l1_pred_3class[idx]}")
    print(f"  L2: {class_names[l2['predictions'][idx]]}")
    print(f"  L3: {class_names[l3['predictions_full'][idx]]}")"""

    task15_code_classical = """# === 手作り特徴量の抽出（32次元） ===
def extract_features(images_13band):
    features = []
    for img in images_13band:
        band_mean = img.mean(axis=(1, 2))    # 13バンドの平均
        band_std = img.std(axis=(1, 2))      # 13バンドの標準偏差
        ndvi = (img[7] - img[3]) / (img[7] + img[3] + 1e-8)
        ndwi = (img[2] - img[7]) / (img[2] + img[7] + 1e-8)
        ndbi = (img[11] - img[7]) / (img[11] + img[7] + 1e-8)
        idx_feats = [ndvi.mean(), ndvi.std(), ndwi.mean(),
                     ndwi.std(), ndbi.mean(), ndbi.std()]
        features.append(np.concatenate([band_mean, band_std, idx_feats]))
    return np.array(features)  # (N, 32)

X_train = extract_features(train_images_13band)
X_test = extract_features(test_images_13band)

# === Random Forest ===
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
print(f"RF accuracy: {rf.score(X_test, y_test):.3f}")

# === SVM (RBF kernel) ===
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(scaler.transform(X_train), y_train)
print(f"SVM accuracy: {svm.score(scaler.transform(X_test), y_test):.3f}")

# === k-NN ===
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(scaler.transform(X_train), y_train)
print(f"k-NN accuracy: {knn.score(scaler.transform(X_test), y_test):.3f}")"""

    task15_code_cnn = """# === Simple CNN（4層、ゼロから学習） ===
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2),            # 64→32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2),            # 32→16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.MaxPool2d(2),            # 16→8
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(), nn.AdaptiveAvgPool2d(1),    # 8→1
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)

model = SimpleCNN(num_classes=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# 入力: 64×64px RGB, 10エポック学習"""

    task15_code_vit = """# === ViT-Tiny（timmライブラリで事前学習済みモデル） ===
import timm

model = timm.create_model('vit_tiny_patch16_224',
                           pretrained=True,    # ImageNet事前学習
                           num_classes=10)
model = model.to(device)

# ファインチューニング設定
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

# 入力: 224×224px RGB, 10エポック学習
# パッチサイズ: 16×16 → 14×14 = 196パッチ
# 各パッチをトークンとしてTransformerで処理"""

    html_doc = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Day 1 衛星画像分類 技術レポート</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      background:
        radial-gradient(1200px 600px at 20% 0%, rgba(30, 64, 175, 0.25), transparent 60%),
        radial-gradient(1000px 500px at 80% 20%, rgba(6, 78, 59, 0.2), transparent 65%),
        #020617;
    }}
    .tab-content {{ display: none; }}
    .tab-content.active {{ display: block; }}
    .tab-btn.active {{ background: rgba(14, 116, 144, .35); border-color: rgba(56, 189, 248, .6); color: #e0f2fe; }}
    .lightbox {{ display: none; }}
    .lightbox.show {{ display: flex; }}
  </style>
</head>
<body class="text-slate-200 min-h-screen">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
    <header class="rounded-2xl border border-slate-700 bg-slate-900/70 shadow-2xl p-6">
      <h1 class="text-3xl sm:text-4xl font-bold tracking-tight text-slate-50">Day 1 衛星画像分類 技術レポート</h1>
      <p class="mt-2 text-slate-300">L1数式ルールからL3ファインチューニングまでの精度・考察・実務示唆を統合したシングルページレポート</p>
    </header>

    <nav class="rounded-2xl border border-slate-700 bg-slate-900/70 p-3">
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
        <button class="tab-btn active rounded-xl border border-slate-600 px-3 py-2 text-sm text-slate-200" data-tab="overview">Overview</button>
        <button class="tab-btn rounded-xl border border-slate-600 px-3 py-2 text-sm text-slate-200" data-tab="task11">Task 1-1: L1 Index Threshold</button>
        <button class="tab-btn rounded-xl border border-slate-600 px-3 py-2 text-sm text-slate-200" data-tab="task12">Task 1-2: L2 Pretrained Model</button>
        <button class="tab-btn rounded-xl border border-slate-600 px-3 py-2 text-sm text-slate-200" data-tab="task13">Task 1-3: L3 Fine-tuning</button>
        <button class="tab-btn rounded-xl border border-slate-600 px-3 py-2 text-sm text-slate-200" data-tab="task14">Task 1-4: L1 vs L2 vs L3 比較</button>
        <button class="tab-btn rounded-xl border border-slate-600 px-3 py-2 text-sm text-slate-200" data-tab="task15">Task 1-5: All Models Comparison</button>
      </div>
    </nav>

    <main class="space-y-8">
      <section id="overview" class="tab-content active space-y-6">
        {section_card('このレポートについて', '<p><strong>目的</strong>: 衛星画像の土地被覆分類（Land Use / Land Cover Classification）において、手法の複雑さを段階的に上げたとき、<strong>分類精度がどう変わるか</strong>を検証する。</p><p>具体的には、数式ベースの単純なルール（L1）→ 学習済みモデルの特徴量利用（L2）→ モデル全体のファインチューニング（L3）という3段階で精度を比較し、各手法の<strong>コスト対効果</strong>と<strong>実務への適用可能性</strong>を評価した。</p>')}

        {section_card('データセット: EuroSAT', table_html(['項目', '内容'], [
            ['データ名', 'EuroSAT（Sentinel-2 衛星画像ベンチマーク）'],
            ['画像数', '27,000枚（各クラス 2,000〜3,000枚）'],
            ['画像サイズ', '64×64ピクセル（10m解像度、約640m×640m の地表をカバー）'],
            ['バンド数', '13バンド（可視光〜短波赤外）。ただし今回はRGB 3バンドのみ使用'],
            ['クラス数', '10クラス（AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake）'],
            ['データ分割', 'Train 80% (21,600枚) / Test 20% (5,400枚)、random_state=42 で固定'],
            ['地域', 'ヨーロッパ各地（Sentinel-2 の公開データから切り出し）'],
        ]))}

        {section_card('実行環境', table_html(['項目', '内容'], [
            ['実行環境', 'Google Colab Pro（GPU: T4 / A100）'],
            ['Python', '3.10'],
            ['主要ライブラリ', 'PyTorch 2.x, torchvision, scikit-learn, numpy, matplotlib'],
            ['事前学習モデル', 'ResNet50（ImageNet-1K で事前学習済み）、ViT-Tiny（同）'],
        ]))}

        {section_card('評価指標', '<ul class="list-disc pl-6 space-y-1 text-slate-300"><li>Accuracy（精度）: 正しく分類できた画像の割合。全体の性能を示す最も基本的な指標</li><li>3クラス精度: 10クラスを Vegetation / Water / Built-up の3つに集約して評価（L1との公平な比較用）</li><li>10クラス精度: 10クラスそれぞれを区別する精度（L2/L3の本来の評価）</li><li>混同行列: どのクラスをどのクラスに間違えたかの詳細</li><li>F1スコア: Precision（適合率）とRecall（再現率）の調和平均。クラスごとの性能評価に使用</li></ul>')}

        {section_card('比較する3手法', table_html(['手法', '概要', '計算コスト'], [
            ['L1: 数式ベース', 'NDVI/NDWI/NDBI の閾値ルールで3クラス分類。AIモデルなし', '数秒、GPU不要'],
            ['L2: 事前学習モデル特徴量', 'ResNet50（凍結）で特徴抽出 → ロジスティック回帰で10クラス分類', '数分、GPU推奨'],
            ['L3: ファインチューニング', 'ResNet50の全層を衛星画像で再学習して10クラス分類', '30分、GPU必須'],
        ]))}

        {section_card('結果サマリ', '<p>Day 1（Task 1-1〜1-5）の全結果を統合し、<strong>何をどのコードでやったか → どんな結果が出たか → なぜそうなるか → 実務への示唆</strong>を1画面で追える内部向けレポートにまとめた。</p>')}

        <section class="grid gap-4 sm:grid-cols-3">
          <div class="rounded-2xl border border-cyan-400/30 bg-cyan-900/20 p-5 shadow-xl">
            <p class="text-sm text-cyan-200">L1 (Index Threshold)</p>
            <p class="text-4xl font-bold text-white mt-1">{esc(pct_ratio(l1_acc))}</p>
            <p class="text-xs text-cyan-100 mt-1">3クラス</p>
          </div>
          <div class="rounded-2xl border border-sky-400/30 bg-sky-900/20 p-5 shadow-xl">
            <p class="text-sm text-sky-200">L2 (ResNet50 + LogReg)</p>
            <p class="text-4xl font-bold text-white mt-1">{esc(pct_ratio(l2_acc10))}</p>
            <p class="text-xs text-sky-100 mt-1">10クラス</p>
          </div>
          <div class="rounded-2xl border border-emerald-400/30 bg-emerald-900/20 p-5 shadow-xl">
            <p class="text-sm text-emerald-200">L3 (ResNet50 Full FT)</p>
            <p class="text-4xl font-bold text-white mt-1">{esc(pct_ratio(l3_acc10))}</p>
            <p class="text-xs text-emerald-100 mt-1">10クラス</p>
          </div>
        </section>

        <section class="rounded-2xl border border-slate-700 bg-slate-800/40 p-4 shadow-xl">
          <div id="plot-overview" class="w-full h-[460px]"></div>
        </section>

        {section_card('衛星データの本質', '<ul class="list-disc pl-6 space-y-1 text-slate-300"><li>Sentinel-2は「写真」を撮るカメラではなく、<strong>13バンドの反射率を計測するセンサー</strong>。</li><li>今回のL2/L3はRGB 3バンド（Band 4/3/2）のみ利用し、<strong>元データの23%（3/13）</strong>しか使っていない。</li><li>残り10バンド（近赤外・短波赤外など）を使う余地があり、特に植生系のクラス分離で改善余地がある。</li></ul>')}

        {section_card('Day 1の5つの学び', overview_lessons)}
      </section>

      <section id="task11" class="tab-content space-y-6">
        {section_card('Task 1-1: L1 Index Threshold', '<p><strong>概要</strong>: NDVI/NDWI/NDBI の3指数と閾値ルールだけで 3クラス分類を実施。AIモデルなしのベースラインとして、どこまで分類できるかを確認した。</p><p><strong>結果</strong>: 3クラス精度 <strong>64.4%</strong>（Vegetation 98.7%、Water 73.9%、Built-up 0.0%）</p>')}

        {section_card('目的と手法（指数の意味）', '<ul class="list-disc pl-6 space-y-1 text-slate-300"><li><strong>NDVI</strong> = (NIR - Red) / (NIR + Red): 植生活性を強調</li><li><strong>NDWI</strong> = (Green - NIR) / (Green + NIR): 水域を強調</li><li><strong>NDBI</strong> = (SWIR - NIR) / (SWIR + NIR): 都市域を強調（理論上）</li></ul><p class="mt-3 text-slate-300">分類ロジック（優先順位）:</p><ol class="list-decimal pl-6 space-y-1 text-slate-300"><li>全件を Vegetation と仮置き</li><li><code>NDWI &gt; -0.18</code> を Water に上書き</li><li>残りで <code>NDBI &gt; NDVI</code> を Built-up に上書き</li></ol>')}

        {section_card('コードで何をやったか（要約）', details_block('展開する', code_block(task11_code, 'python')))}

        {section_card('結果テーブル（3クラス）', table_html(['クラス', '正解数 / 全数', '正解率'], [
            ['Vegetation', '13,331 / 13,500', '98.7%'],
            ['Water', '4,064 / 5,500', '73.9%'],
            ['Built-up', '2 / 8,000', '0.0%'],
        ]))}

        {section_card('結果テーブル（10クラス詳細）', table_html(['元クラス', '統合先', '枚数', '正解率'], [
            ['Forest', 'Vegetation', '3000', '100.0%'],
            ['PermanentCrop', 'Vegetation', '2500', '99.9%'],
            ['Pasture', 'Vegetation', '2000', '99.8%'],
            ['AnnualCrop', 'Vegetation', '3000', '98.8%'],
            ['HerbaceousVegetation', 'Vegetation', '3000', '95.8%'],
            ['SeaLake', 'Water', '3000', '100.0%'],
            ['River', 'Water', '2500', '42.6%'],
            ['Highway', 'Built-up', '2500', '0.0%'],
            ['Industrial', 'Built-up', '2500', '0.0%'],
            ['Residential', 'Built-up', '3000', '0.1%'],
        ]))}

        {image_block(image_map, 'index_distributions.png', '指数分布（index_distributions.png）', 'NDWIがWaterを比較的分ける一方、NDVI/NDBIでBuilt-upとVegetationが重なる点。', '閾値分類でBuilt-upが難しいことを学習前に予見できる。')}
        {image_block(image_map, 'L1_confusion_matrix.png', '混同行列（L1_confusion_matrix.png）', 'Built-up行がどこへ流れているか。', 'Built-upの大半がVegetationに流れ、構造的な失敗が確認できる。')}
        {image_block(image_map, 'L1_classification_maps.png', 'ピクセル分類マップ（L1_classification_maps.png）', 'River画像で川と周辺植生がどう塗り分けられているか。', 'Riverの平均値が植生側へ引っ張られる理由が見える。')}
        {image_block(image_map, 'L1_3d_scatter.png', '3D散布図（L1_3d_scatter.png）', 'Built-upとVegetationの点群重なり。', '3指数を同時に使っても線形閾値では分離困難である。')}
        {image_block(image_map, 'eurosat_10classes_rgb.png', '10クラスRGBサンプル（eurosat_10classes_rgb.png）', '見た目が似るクラス（例: AnnualCropとResidential）。', '人間目視でも曖昧なペアが誤分類源になる。')}

        {section_card('考察', '<ul class="list-disc pl-6 space-y-1 text-slate-300"><li><strong>Built-up 0.0%の主因</strong>: 都市パッチ内に緑地が多く混在し、画像平均では <code>NDBI &gt; NDVI</code> がほぼ成立しない。</li><li><strong>River 42.6%の主因</strong>: 河川が細く、640mパッチの大半を周辺植生が占めるため平均NDWIが下がる。</li><li>閾値法は「明確に異なる物質（水 vs 植生）」には有効だが、混在・微差の識別には限界。</li></ul>')}

        {section_card('L2への接続', info_box('L1の限界は『平均値3次元の情報不足』。L2ではResNet50の2048次元特徴により、テクスチャ・形状・配置パターンを使った判別へ移行する。'))}
      </section>

      <section id="task12" class="tab-content space-y-6">
        {section_card('Task 1-2: L2 Pretrained Model', '<p><strong>概要</strong>: ImageNet事前学習済み ResNet50 を特徴抽出器として固定し、抽出された2048次元特徴にロジスティック回帰を学習。</p><p><strong>結果</strong>: 10クラス <strong>95.7%</strong>、3クラス <strong>98.1%</strong></p>')}

        {section_card('バックボーン + 分類器の構造', table_html(['段階', '内容'], [
            ['入力', '224×224×3 = 150,528個の数値'],
            ['バックボーン', 'ResNet50（50層、約2,350万パラメータ）'],
            ['特徴', '2,048次元特徴ベクトル'],
            ['分類器', 'LogReg（2,048×10+10 = 20,490パラメータ）'],
            ['出力', '10クラス確率（softmax）'],
        ]) + '<ul class="list-disc pl-6 mt-4 space-y-1 text-slate-300"><li>バックボーンは画像から「意味のある特徴」を圧縮抽出する装置。</li><li>分類器は2048特徴に重みを掛け、10クラスへ振り分ける。</li><li>L2ではバックボーンを凍結し、分類器のみ学習（転移学習の最小構成）。</li></ul>')}

        {section_card('コードで何をやったか（要約）', details_block('展開する', code_block(task12_code, 'python')))}

        <section class="grid gap-4 sm:grid-cols-2">
          <div class="rounded-2xl border border-sky-400/30 bg-sky-900/20 p-5 shadow-xl">
            <p class="text-sm text-sky-200">10クラス精度</p>
            <p class="text-4xl font-bold text-white mt-1">{esc(pct_ratio(l2.get('accuracy_10class') if l2 else np.nan))}</p>
          </div>
          <div class="rounded-2xl border border-cyan-400/30 bg-cyan-900/20 p-5 shadow-xl">
            <p class="text-sm text-cyan-200">3クラス精度</p>
            <p class="text-4xl font-bold text-white mt-1">{esc(pct_ratio(l2.get('accuracy_3class') if l2 else np.nan))}</p>
          </div>
        </section>

        {image_block(image_map, 'L2_confusion_matrix.png', 'L2_confusion_matrix', '対角線の濃さと誤分類先の集中。', 'L1で壊滅したBuilt-up系が90%超まで回復。')}
        {image_block(image_map, 'L2_tsne.png', 'L2_tsne', 'クラスごとのクラスタ分離と重なり領域。', '3次元指数では見えない分離構造を2048次元特徴が形成している。')}
        {image_block(image_map, 'L2_per_class_accuracy.png', 'L2_per_class_accuracy', 'クラス間ばらつき（最低Pasture〜最高SeaLake）。', '全クラス92%以上で大崩れがない。')}
        {image_block(image_map, 'L2_predictions.png', 'L2_predictions', '高確信度・低確信度の正解サンプル。', '『当てたけど迷った画像』がどのタイプか把握できる。')}
        {image_block(image_map, 'L2_correct_vs_misclassified.png', 'L2_correct_vs_misclassified', '各クラスの正解例と誤分類例の差。', '誤分類が似たクラス間で起きることを視覚確認できる。')}
        {image_block(image_map, 'L2_hard_samples.png', 'L2_hard_samples', '低確信度サンプルの見た目。', 'モデル限界は境界サンプルに集中している。')}

        {section_card('L1→L2で +31%pt ジャンプした理由', '<ul class="list-disc pl-6 space-y-1 text-slate-300"><li>L1: <code>NDVI/NDWI/NDBI</code> の <strong>3次元平均値</strong>で判定</li><li>L2: ResNet50の <strong>2048次元特徴</strong>で判定</li></ul><p class="mt-3 text-slate-300">この「3次元→2048次元」の表現力の跳躍により、平均値では消える空間パターン（道路の直線、住宅地の格子、河川の蛇行）を識別できるようになった。</p>')}
      </section>

      <section id="task13" class="tab-content space-y-6">
        {section_card('Task 1-3: L3 Fine-tuning', '<p><strong>概要</strong>: ResNet50の全層を衛星画像で再学習（ファインチューニング）し、L2の凍結特徴からドメイン適応へ移行。</p><p><strong>結果</strong>: Full FT 10クラス <strong>98.5%</strong>、3クラス <strong>99.6%</strong></p>')}

        {section_card('L2との違い', table_html(['項目', 'L2（特徴抽出）', 'L3（FT）'], [
            ['バックボーン', '凍結', '更新'],
            ['学習対象', '分類器のみ', '全層（またはヘッド）'],
            ['計算負荷', '低', '高'],
            ['期待効果', '即効で高精度', 'ドメイン最適化'],
        ]))}

        {section_card('3実験の詳細', table_html(['実験', '学習対象', '10クラス精度'], [
            ['Head Only', 'FC層のみ', pct_ratio(l3.get('accuracy_10class_head') if l3 else np.nan)],
            ['Full FT', '全23.5Mパラメータ', pct_ratio(l3.get('accuracy_10class_full') if l3 else np.nan)],
            ['50% Data FT', '全23.5M（データ半分）', pct_ratio(l3.get('accuracy_10class_half') if l3 else np.nan)],
        ]))}

        {section_card('学習設定の解説', '<ul class="list-disc pl-6 space-y-1 text-slate-300"><li><strong>エポック</strong>: 全学習データ1周。10エポック = 21,600枚を10周。</li><li><strong>バッチサイズ64</strong>: 21,600枚を337バッチに分割。更新回数は <code>337×10=3,370</code>。</li><li><strong>Adam</strong>: 勢いを考慮してパラメータ更新する最適化手法。</li><li><strong>CosineAnnealing</strong>: 前半は大きく、後半は小さく学習率を下げる。</li><li><strong>データ拡張</strong>: 回転・反転で擬似データを増やして汎化を改善。</li><li><strong>正規化</strong>: <code>(値-平均)/標準偏差</code> でスケールを揃える（偏差値と同じ原理）。</li></ul>')}

        {section_card('コードで何をやったか（要約）', details_block('展開する', code_block(task13_code, 'python')))}

        {section_card('GPUが必要な理由', '<p>CPUは少数コアの逐次処理、GPUは数千コアの並列処理。2350万パラメータ更新を3370回繰り返すFull FTはGPU前提。</p>')}

        {section_card('インタラクティブ学習曲線（Test Accuracy）', '<div id="plot-task13" class="w-full h-[460px]"></div>')}

        {section_card('結果テーブル + 分類レポート（Full FT）', table_html(['手法', '3クラス精度', '10クラス精度'], [
            ['L1', '64.4%', 'N/A'],
            ['L2', '98.1%', '95.7%'],
            ['L3 Full FT', '99.6%', '98.5%'],
        ]) + '<div class="mt-4"></div>' + table_html(['クラス', 'Precision', 'Recall', 'F1'], [
            ['AnnualCrop', '0.967', '0.977', '0.972'],
            ['Forest', '1.000', '0.990', '0.995'],
            ['HerbaceousVegetation', '0.972', '0.985', '0.978'],
            ['Highway', '0.986', '0.990', '0.988'],
            ['Industrial', '0.986', '0.994', '0.990'],
            ['Pasture', '0.987', '0.963', '0.975'],
            ['PermanentCrop', '0.966', '0.970', '0.968'],
            ['Residential', '0.998', '0.990', '0.994'],
            ['River', '0.992', '0.988', '0.990'],
            ['SeaLake', '0.992', '0.993', '0.993'],
        ]))}

        {image_block(image_map, 'L3_learning_curves.png', 'L3_learning_curves', '収束速度の差', 'Full FTは初期から高精度で飽和。')}
        {image_block(image_map, 'L3_confusion_matrix.png', 'L3_confusion_matrix', '対角線集中', '全クラスで高い再現率。')}
        {image_block(image_map, 'L3_per_class_accuracy.png', 'L3_per_class_accuracy', 'Head vs Fullの差', '全層更新の改善幅をクラス別に把握できる。')}
        {image_block(image_map, 'L3_confusion_diff.png', 'L3_confusion_diff', '差分ヒートマップ', 'どの誤分類ペアが減ったかが見える。')}
        {image_block(image_map, 'L3_correct_vs_incorrect.png', 'L3_correct_vs_incorrect', '誤分類サンプルの質', '残存エラーは人間にも難しい境界事例。')}

        {section_card('学習曲線の読み方', '<p>最初の1〜3エポックで急改善し、その後は緩やかに収束。典型的な収穫逓減パターン。</p>')}

        {section_card('なぜ Head Only (92.4%) < L2 (95.7%) か', '<ul class="list-disc pl-6 space-y-1 text-slate-300"><li>L2は全データを一括で使うロジスティック回帰最適化（大域的に解きやすい）。</li><li>Head Onlyは64枚ミニバッチ更新で局所的・反復的に最適化。</li><li>NN向け汎用オプティマイザは、2万パラメータの線形分類には過剰で非効率になることがある。</li></ul>')}
      </section>

      <section id="task14" class="tab-content space-y-6">
        {section_card('Task 1-4: L1 vs L2 vs L3 比較', '<p><strong>概要</strong>: L1/L2/L3の結果を横並びで比較し、「どこで何がどれだけ改善したか」を定量・定性の両面で評価。</p>')}

        {section_card('精度比較（5手法）', table_html(['手法', '3クラス精度', '10クラス精度', '改善幅'], [
            ['L1: NDVI/NDWI/NDBI Threshold', '64.4%', 'N/A', '-'],
            ['L2: ResNet50 + LogReg', '98.1%', '95.7%', '+31.3pt (vs L1)'],
            ['L3: FT Head Only', '-', '92.4%', '-3.3pt (vs L2)'],
            ['L3: FT Full (100% data)', '99.6%', '98.5%', '+2.8pt (vs L2)'],
            ['L3: FT Full (50% data)', '-', '98.0%', '+2.3pt (vs L2)'],
        ]))}

        {section_card('コードで何をやったか（要約）', details_block('展開する', code_block(task14_code, 'python')))}

        {section_card('5つの主要な発見', '<ol class="list-decimal pl-6 space-y-1 text-slate-300"><li>L1→L2 が最大のジャンプ（+31.3pt）。</li><li>L2→L3 Full FT は +2.8pt の上積み。</li><li>Head Only は L2未満で、凍結条件ならLogRegが強い。</li><li>Full FTが最高精度（98.5%）。</li><li>学習データ50%でも98.0%（EuroSATのクリーンさを反映）。</li></ol>')}

        {image_block(image_map, 'L1_L2_L3_comparison_bar.png', 'L1_L2_L3_comparison_bar', 'L1→L2→L3の段差', '最も効く改善ステップが一目でわかる。')}
        {image_block(image_map, 'effort_vs_accuracy.png', 'effort_vs_accuracy', '労力と精度の位置関係', '実務での投資対効果判断に直結する。')}
        {image_block(image_map, 'L2_vs_L3_confusion.png', 'L2_vs_L3_confusion', '誤分類セルの減少', 'FTでどの混同が減ったか確認できる。')}
        {image_block(image_map, 'L2_vs_L3_radar.png', 'L2_vs_L3_radar', 'L2とL3の面積差', 'L3が全方向で上回る傾向を把握できる。')}
        {image_block(image_map, '3class_comparison.png', '3class_comparison', 'L1/L2/L3の3クラス混同行列差', 'Built-up崩壊がL2/L3で解消したことが明確。')}
        {image_block(image_map, 'same_image_comparison.png', 'same_image_comparison', '同一画像の予測変化', '手法進化に伴う予測改善を直感的に追える。')}
        {image_block(image_map, 'comparison_6samples_L1L2L3.png', 'comparison_6samples_L1L2L3', '6サンプル横並び', 'クラス横断の改善傾向を確認できる。')}
        {image_block(image_map, 'comparison_improvement_map.png', 'comparison_improvement_map', 'クラス別改善量', 'FTの恩恵が大きいクラスを特定できる。')}
        {image_block(image_map, 'comparison_L3_failures.png', 'comparison_L3_failures', 'L3でも誤る画像', '残課題がデータ曖昧性にあるとわかる。')}
        {image_block(image_map, 'comparison_L3_error_rate.png', 'comparison_L3_error_rate', 'クラス別エラー率', '次に改善すべきクラス優先度を決められる。')}

        {section_card('きれいなデータ vs 実務データの精度差', table_html(['データ量', 'EuroSAT', '実務データ'], [
            ['1,000枚', '約95%', '約78%'],
            ['5,000枚', '約97%', '約86%'],
            ['20,000枚', '98.5%', '約91%'],
        ]))}

        {section_card('ラベル作りのコスト', '<p>衛星画像自体は無料で取得可能だが、高品質ラベル作成（特にセグメンテーション）は1枚30分〜1時間の人手コストが支配的。</p>')}
      </section>

      <section id="task15" class="tab-content space-y-6">
        {section_card('Task 1-5: All Models Comparison', '<p><strong>概要</strong>: 古典ML〜DLまで全8手法を横並び比較し、モデル選定の実務判断に使える精度・コスト地図を作成。</p>')}

        {section_card('古典ML vs DL の本質', '<ul class="list-disc pl-6 space-y-1 text-slate-300"><li><strong>古典ML</strong>: 人間が特徴量を設計し、モデルは判定だけを担当。</li><li><strong>DL</strong>: モデルが特徴抽出と判定を同時に学習。深い層で段階的に高次特徴を構築。</li></ul>')}

        {section_card('古典ML向け手作り特徴量（32次元）', table_html(['特徴群', '次元', '合計'], [
            ['13バンド平均', '13', ''],
            ['13バンド標準偏差', '13', ''],
            ['NDVI/NDWI/NDBI平均', '3', ''],
            ['NDVI/NDWI/NDBI標準偏差', '3', '32'],
        ]))}

        {section_card('古典ML コード（要約）', details_block('展開する', code_block(task15_code_classical, 'python')))}

        {section_card('Simple CNN コード（要約）', details_block('展開する', code_block(task15_code_cnn, 'python')))}

        {section_card('ViT-Tiny コード（要約）', details_block('展開する', code_block(task15_code_vit, 'python')))}

        {section_card('全8モデル概要', table_html(['モデル', 'カテゴリ', '精度'], [
            ['L1: Index Threshold', 'L1', '64.4'],
            ['Random Forest', 'Classical ML', '90.2'],
            ['SVM (RBF)', 'Classical ML', '93.8'],
            ['k-NN (k=5)', 'Classical ML', '90.3'],
            ['Simple CNN', 'Deep Learning', '94.9'],
            ['L2: ResNet50 + LogReg', 'L2', '95.7'],
            ['L3: ResNet50 FT', 'L3', '98.5'],
            ['ViT-Tiny FT', 'Deep Learning', '98.5'],
        ]))}

        {section_card('精度ランキング（インタラクティブ）', '<div id="plot-task15" class="w-full h-[520px]"></div>')}

        {allm_table}

        {image_block(image_map, 'all_models_comparison.png', 'all_models_comparison', 'カテゴリ別の精度階段', 'L1→Classical→DL/FT で段階的に伸びる。')}
        {image_block(image_map, 'all_models_ranking.png', 'all_models_ranking', '順位の確定', '最終候補モデルを短時間で絞り込める。')}
        {image_block(image_map, 'all_models_class_heatmap.png', 'all_models_class_heatmap', 'クラス別の苦手分布', 'モデル選定時にクラス別リスクを評価できる。')}
        {image_block(image_map, 'all_models_cost_vs_accuracy.png', 'all_models_cost_vs_accuracy', '学習時間と精度の関係', '精度だけでなくコスト制約下の最適点を選べる。')}

        {section_card('事前学習 + FT が強い理由', '<p>ImageNet 120万枚で得た視覚知識を初期値にすることで、EuroSAT 27,000枚だけでは学びきれないパターンを利用できるため。</p>')}
        {section_card('実務でのモデル選定フロー', '<ol class="list-decimal pl-6 space-y-1 text-slate-300"><li>まずResNet50/ViTをFTしてベースライン作成</li><li>要件を満たせば運用へ</li><li>不足する場合はデータ追加・ラベル改善・衛星専用モデルへ拡張</li></ol>')}
      </section>
    </main>
  </div>

  <div id="lightbox" class="lightbox fixed inset-0 bg-black/90 items-center justify-center p-4 z-50">
    <button id="lightbox-close" class="absolute top-4 right-4 text-white text-3xl leading-none">×</button>
    <img id="lightbox-img" src="" alt="expanded" class="max-w-full max-h-full rounded-xl shadow-2xl">
  </div>

  <script>
    const overviewPlot = {overview_plot_json};
    const task13Plot = {task13_plot_json};
    const task15Plot = {task15_plot_json};

    function renderPlots() {{
      Plotly.newPlot('plot-overview', overviewPlot.data, overviewPlot.layout, {{responsive: true, displaylogo: false}});
      Plotly.newPlot('plot-task13', task13Plot.data, task13Plot.layout, {{responsive: true, displaylogo: false}});
      Plotly.newPlot('plot-task15', task15Plot.data, task15Plot.layout, {{responsive: true, displaylogo: false}});
    }}

    function setupTabs() {{
      const buttons = document.querySelectorAll('.tab-btn');
      const sections = document.querySelectorAll('.tab-content');
      buttons.forEach((btn) => {{
        btn.addEventListener('click', () => {{
          const id = btn.dataset.tab;
          buttons.forEach((b) => b.classList.remove('active'));
          btn.classList.add('active');
          sections.forEach((sec) => {{
            sec.classList.toggle('active', sec.id === id);
          }});
          window.scrollTo({{top: 0, behavior: 'smooth'}});
        }});
      }});
    }}

    function setupLightbox() {{
      const modal = document.getElementById('lightbox');
      const modalImg = document.getElementById('lightbox-img');
      const closeBtn = document.getElementById('lightbox-close');
      document.querySelectorAll('.lightbox-trigger').forEach((img) => {{
        img.addEventListener('click', () => {{
          modalImg.src = img.src;
          modal.classList.add('show');
        }});
      }});
      closeBtn.addEventListener('click', () => modal.classList.remove('show'));
      modal.addEventListener('click', (e) => {{
        if (e.target === modal) modal.classList.remove('show');
      }});
      document.addEventListener('keydown', (e) => {{
        if (e.key === 'Escape') modal.classList.remove('show');
      }});
    }}

    window.addEventListener('DOMContentLoaded', () => {{
      renderPlots();
      setupTabs();
      setupLightbox();
    }});
  </script>
</body>
</html>
"""
    return html_doc


def main():
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    l1 = load_npz(TASK_DIRS['L1'] / 'L1_results.npz')
    l2 = load_npz(TASK_DIRS['L2'] / 'L2_results.npz')
    l3 = load_npz(TASK_DIRS['L3'] / 'L3_results.npz')
    allm = load_npz(TASK_DIRS['ALL'] / 'all_models_results.npz')

    image_map = copy_images()
    html_out = build_html(l1, l2, l3, allm, image_map)

    out_path = DIST_DIR / 'index.html'
    out_path.write_text(html_out, encoding='utf-8')

    print(f'Generated: {out_path.resolve()}')
    print(f'Copied images: {len(list(IMAGES_DIR.glob("*.png")))}')


if __name__ == '__main__':
    main()
