from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Day 1 衛星画像解析ダッシュボード", layout="wide")

DATA_DIR = Path('/Users/kosuke/Dropbox/11_space-tech/02_satellite-data/00_Inbox/day1')

TASK_DIRS = {
    "L1": DATA_DIR / "task1_1_ndvi_threshold",
    "L2": DATA_DIR / "task1_2_resnet50_inference",
    "L3": DATA_DIR / "task1_3_resnet50_finetune",
    "CMP": DATA_DIR / "task1_4_comparison",
    "ALL": DATA_DIR / "task1_5_all_models",
}


@st.cache_data
def load_npz(path: Path):
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def pct(v):
    return f"{float(v) * 100:.1f}%"


def show_image(image_path: Path, title: str, what_to_see: str, insight: str):
    st.markdown(f"#### {title}")
    if image_path.exists():
        st.image(str(image_path), use_column_width=True)
    else:
        st.warning(f"画像なし: {image_path.name}")
    st.caption(f"何を見るべきか: {what_to_see}")
    st.caption(f"何がわかるか: {insight}")


def md_table(df: pd.DataFrame):
    st.markdown(df.to_markdown(index=False))


def page_overview(l1, l2, l3):
    st.title("Day 1 技術レポート: 衛星画像分類ダッシュボード")

    st.markdown("---")
    st.markdown("## このレポートについて")
    st.markdown(
        """
**目的**: 衛星画像の土地被覆分類（Land Use / Land Cover Classification）において、
手法の複雑さを段階的に上げたとき、**分類精度がどう変わるか**を検証する。

具体的には、数式ベースの単純なルール（L1）→ 学習済みモデルの特徴量利用（L2）→
モデル全体のファインチューニング（L3）という3段階で精度を比較し、
各手法の**コスト対効果**と**実務への適用可能性**を評価した。
"""
    )

    st.markdown("### データセット: EuroSAT")
    st.markdown(
        """
| 項目 | 内容 |
|---|---|
| **データ名** | EuroSAT（Sentinel-2 衛星画像ベンチマーク） |
| **画像数** | 27,000枚（各クラス 2,000〜3,000枚） |
| **画像サイズ** | 64×64ピクセル（10m解像度、約640m×640m の地表をカバー） |
| **バンド数** | 13バンド（可視光〜短波赤外）。ただし今回はRGB 3バンドのみ使用 |
| **クラス数** | 10クラス（AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake） |
| **データ分割** | Train 80% (21,600枚) / Test 20% (5,400枚)、random_state=42 で固定 |
| **地域** | ヨーロッパ各地（Sentinel-2 の公開データから切り出し） |
"""
    )

    st.markdown("### 実行環境")
    st.markdown(
        """
| 項目 | 内容 |
|---|---|
| **実行環境** | Google Colab Pro（GPU: T4 / A100） |
| **Python** | 3.10 |
| **主要ライブラリ** | PyTorch 2.x, torchvision, scikit-learn, numpy, matplotlib |
| **事前学習モデル** | ResNet50（ImageNet-1K で事前学習済み）、ViT-Tiny（同） |
"""
    )

    st.markdown("### 評価指標")
    st.markdown(
        """
- **Accuracy（精度）**: 正しく分類できた画像の割合。全体の性能を示す最も基本的な指標
- **3クラス精度**: 10クラスを Vegetation / Water / Built-up の3つに集約して評価（L1との公平な比較用）
- **10クラス精度**: 10クラスそれぞれを区別する精度（L2/L3の本来の評価）
- **混同行列**: どのクラスをどのクラスに間違えたかの詳細
- **F1スコア**: Precision（適合率）とRecall（再現率）の調和平均。クラスごとの性能評価に使用
"""
    )

    st.markdown("### 比較する3手法")
    st.markdown(
        """
| 手法 | 概要 | 計算コスト |
|---|---|---|
| **L1: 数式ベース** | NDVI/NDWI/NDBI の閾値ルールで3クラス分類。AIモデルなし | 数秒、GPU不要 |
| **L2: 事前学習モデル特徴量** | ResNet50（凍結）で特徴抽出 → ロジスティック回帰で10クラス分類 | 数分、GPU推奨 |
| **L3: ファインチューニング** | ResNet50の全層を衛星画像で再学習して10クラス分類 | 30分、GPU必須 |
"""
    )
    st.markdown("---")

    st.markdown("## 結果サマリ")
    st.markdown(
        """
Day 1（Task 1-1〜1-5）の全結果を統合し、
**何をどのコードでやったか → どんな結果が出たか → なぜそうなるか → 実務への示唆**
を1画面で追える内部向けレポートにまとめた。
"""
    )

    l1_acc = float(l1["accuracy"]) if l1 is not None else np.nan
    l2_acc10 = float(l2["accuracy_10class"]) if l2 is not None else np.nan
    l2_acc3 = float(l2["accuracy_3class"]) if l2 is not None else np.nan
    l3_acc10 = float(l3["accuracy_10class_full"]) if l3 is not None else np.nan
    l3_acc3 = float(l3["accuracy_3class"]) if l3 is not None else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("L1 (Index Threshold)", pct(l1_acc) if not np.isnan(l1_acc) else "N/A", "3クラス")
    c2.metric("L2 (ResNet50 + LogReg)", pct(l2_acc10) if not np.isnan(l2_acc10) else "N/A", "10クラス")
    c3.metric("L3 (ResNet50 Full FT)", pct(l3_acc10) if not np.isnan(l3_acc10) else "N/A", "10クラス")

    x = ["L1", "L2", "L3"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="3クラス精度",
            x=x,
            y=[l1_acc * 100, l2_acc3 * 100, l3_acc3 * 100],
            marker_color=["#7f8c8d", "#2980b9", "#27ae60"],
            text=[f"{l1_acc*100:.1f}", f"{l2_acc3*100:.1f}", f"{l3_acc3*100:.1f}"],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="10クラス精度",
            x=x,
            y=[None, l2_acc10 * 100, l3_acc10 * 100],
            marker_color=["#bdc3c7", "#3498db", "#2ecc71"],
            text=["N/A", f"{l2_acc10*100:.1f}", f"{l3_acc10*100:.1f}"],
            textposition="outside",
        )
    )
    fig.update_layout(
        barmode="group",
        title="L1→L2→L3 精度サマリ",
        yaxis_title="Accuracy (%)",
        height=460,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 衛星データの本質")
    st.markdown(
        """
- Sentinel-2は「写真」を撮るカメラではなく、**13バンドの反射率を計測するセンサー**。
- 今回のL2/L3はRGB 3バンド（Band 4/3/2）のみ利用し、**元データの23%（3/13）**しか使っていない。
- 残り10バンド（近赤外・短波赤外など）を使う余地があり、特に植生系のクラス分離で改善余地がある。
"""
    )

    st.markdown("### Day 1の5つの学び")
    lessons = [
        "学習済みモデルを使うだけで L1→L2 で +31.3pt の最大ジャンプ。",
        "ファインチューニング（L2→L3）で +2.8pt。追加コストに対して着実な上積み。",
        "Head Only (92.4%) は L2 (95.7%) を下回る。凍結特徴 + LogReg の最適化効率が高い。",
        "EuroSATのようなきれいなデータでは学習データ50%でも -0.5pt。実務データでは差が拡大しやすい。",
        "実務のボトルネックは画像取得ではなくラベル作成コスト（セグメンテーション1枚30分〜1時間）。",
    ]
    for i, txt in enumerate(lessons, 1):
        with st.expander(f"学び {i}"):
            st.markdown(txt)


def page_task11(l1):
    st.title("Task 1-1: L1 Index Threshold")
    st.markdown(
        """
**概要**: NDVI/NDWI/NDBI の3指数と閾値ルールだけで 3クラス分類を実施。
AIモデルなしのベースラインとして、どこまで分類できるかを確認した。

**結果**: 3クラス精度 **64.4%**（Vegetation 98.7%、Water 73.9%、Built-up 0.0%）
"""
    )

    st.markdown("### 目的と手法（指数の意味）")
    st.markdown(
        """
- **NDVI** = (NIR - Red) / (NIR + Red): 植生活性を強調
- **NDWI** = (Green - NIR) / (Green + NIR): 水域を強調
- **NDBI** = (SWIR - NIR) / (SWIR + NIR): 都市域を強調（理論上）

分類ロジック（優先順位）:
1. 全件を Vegetation と仮置き
2. `NDWI > -0.18` を Water に上書き
3. 残りで `NDBI > NDVI` を Built-up に上書き
"""
    )

    with st.expander("コードで何をやったか（要約）"):
        st.code(
            """# 1) データ読み込み
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
report = confusion_matrix_and_class_accuracy(pred, y_true_3, y_true_10)
""",
            language="python",
        )

    st.markdown("### 結果テーブル（3クラス）")
    table_3 = pd.DataFrame(
        {
            "クラス": ["Vegetation", "Water", "Built-up"],
            "正解数 / 全数": ["13,331 / 13,500", "4,064 / 5,500", "2 / 8,000"],
            "正解率": ["98.7%", "73.9%", "0.0%"],
        }
    )
    md_table(table_3)

    st.markdown("### 結果テーブル（10クラス詳細）")
    table_10 = pd.DataFrame(
        {
            "元クラス": [
                "Forest",
                "PermanentCrop",
                "Pasture",
                "AnnualCrop",
                "HerbaceousVegetation",
                "SeaLake",
                "River",
                "Highway",
                "Industrial",
                "Residential",
            ],
            "統合先": ["Vegetation"] * 5 + ["Water", "Water", "Built-up", "Built-up", "Built-up"],
            "枚数": [3000, 2500, 2000, 3000, 3000, 3000, 2500, 2500, 2500, 3000],
            "正解率": ["100.0%", "99.9%", "99.8%", "98.8%", "95.8%", "100.0%", "42.6%", "0.0%", "0.0%", "0.1%"],
        }
    )
    md_table(table_10)

    show_image(
        TASK_DIRS["L1"] / "index_distributions.png",
        "指数分布（index_distributions.png）",
        "NDWIがWaterを比較的分ける一方、NDVI/NDBIでBuilt-upとVegetationが重なる点。",
        "閾値分類でBuilt-upが難しいことを学習前に予見できる。",
    )
    show_image(
        TASK_DIRS["L1"] / "L1_confusion_matrix.png",
        "混同行列（L1_confusion_matrix.png）",
        "Built-up行がどこへ流れているか。",
        "Built-upの大半がVegetationに流れ、構造的な失敗が確認できる。",
    )
    show_image(
        TASK_DIRS["L1"] / "L1_classification_maps.png",
        "ピクセル分類マップ（L1_classification_maps.png）",
        "River画像で川と周辺植生がどう塗り分けられているか。",
        "Riverの平均値が植生側へ引っ張られる理由が見える。",
    )
    show_image(
        TASK_DIRS["L1"] / "L1_3d_scatter.png",
        "3D散布図（L1_3d_scatter.png）",
        "Built-upとVegetationの点群重なり。",
        "3指数を同時に使っても線形閾値では分離困難である。",
    )
    show_image(
        TASK_DIRS["L1"] / "eurosat_10classes_rgb.png",
        "10クラスRGBサンプル（eurosat_10classes_rgb.png）",
        "見た目が似るクラス（例: AnnualCropとResidential）。",
        "人間目視でも曖昧なペアが誤分類源になる。",
    )

    st.markdown("### 考察")
    st.markdown(
        """
- **Built-up 0.0%の主因**: 都市パッチ内に緑地が多く混在し、画像平均では `NDBI > NDVI` がほぼ成立しない。
- **River 42.6%の主因**: 河川が細く、640mパッチの大半を周辺植生が占めるため平均NDWIが下がる。
- 閾値法は「明確に異なる物質（水 vs 植生）」には有効だが、混在・微差の識別には限界。
"""
    )

    st.markdown("### L2への接続")
    st.info(
        "L1の限界は『平均値3次元の情報不足』。L2ではResNet50の2048次元特徴により、"
        "テクスチャ・形状・配置パターンを使った判別へ移行する。"
    )


def page_task12(l2):
    st.title("Task 1-2: L2 Pretrained Model")
    st.markdown(
        """
**概要**: ImageNet事前学習済み ResNet50 を特徴抽出器として固定し、
抽出された2048次元特徴にロジスティック回帰を学習。

**結果**: 10クラス **95.7%**、3クラス **98.1%**
"""
    )

    st.markdown("### バックボーン + 分類器の構造")
    arch_df = pd.DataFrame(
        {
            "段階": ["入力", "バックボーン", "特徴", "分類器", "出力"],
            "内容": [
                "224×224×3 = 150,528個の数値",
                "ResNet50（50層、約2,350万パラメータ）",
                "2,048次元特徴ベクトル",
                "LogReg（2,048×10+10 = 20,490パラメータ）",
                "10クラス確率（softmax）",
            ],
        }
    )
    md_table(arch_df)

    st.markdown(
        """
- バックボーンは画像から「意味のある特徴」を圧縮抽出する装置。
- 分類器は2048特徴に重みを掛け、10クラスへ振り分ける。
- L2ではバックボーンを凍結し、分類器のみ学習（転移学習の最小構成）。
"""
    )

    with st.expander("コードで何をやったか（要約）"):
        st.code(
            """# 1) ResNet50で特徴抽出（weights=ImageNet）
features_train = backbone(train_images)   # (21600, 2048)
features_test  = backbone(test_images)    # (5400, 2048)

# 2) ロジスティック回帰で10クラス分類
clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
clf.fit(features_train, y_train)
y_pred = clf.predict(features_test)

# 3) 評価
acc10 = accuracy_score(y_test, y_pred)
acc3  = accuracy_score(y_test_3, y_pred_3)
""",
            language="python",
        )

    st.markdown("### 結果")
    c1, c2 = st.columns(2)
    c1.metric("10クラス精度", pct(l2["accuracy_10class"]))
    c2.metric("3クラス精度", pct(l2["accuracy_3class"]))

    show_image(
        TASK_DIRS["L2"] / "L2_confusion_matrix.png",
        "L2_confusion_matrix",
        "対角線の濃さと誤分類先の集中。",
        "L1で壊滅したBuilt-up系が90%超まで回復。",
    )
    show_image(
        TASK_DIRS["L2"] / "L2_tsne.png",
        "L2_tsne",
        "クラスごとのクラスタ分離と重なり領域。",
        "3次元指数では見えない分離構造を2048次元特徴が形成している。",
    )
    show_image(
        TASK_DIRS["L2"] / "L2_per_class_accuracy.png",
        "L2_per_class_accuracy",
        "クラス間ばらつき（最低Pasture〜最高SeaLake）。",
        "全クラス92%以上で大崩れがない。",
    )
    show_image(
        TASK_DIRS["L2"] / "L2_predictions.png",
        "L2_predictions",
        "高確信度・低確信度の正解サンプル。",
        "『当てたけど迷った画像』がどのタイプか把握できる。",
    )
    show_image(
        TASK_DIRS["L2"] / "L2_correct_vs_misclassified.png",
        "L2_correct_vs_misclassified",
        "各クラスの正解例と誤分類例の差。",
        "誤分類が似たクラス間で起きることを視覚確認できる。",
    )
    show_image(
        TASK_DIRS["L2"] / "L2_hard_samples.png",
        "L2_hard_samples",
        "低確信度サンプルの見た目。",
        "モデル限界は境界サンプルに集中している。",
    )

    st.markdown("### L1→L2で +31%pt ジャンプした理由")
    st.markdown(
        """
- L1: `NDVI/NDWI/NDBI` の **3次元平均値**で判定
- L2: ResNet50の **2048次元特徴**で判定

この「3次元→2048次元」の表現力の跳躍により、
平均値では消える空間パターン（道路の直線、住宅地の格子、河川の蛇行）を識別できるようになった。
"""
    )


def page_task13(l3):
    st.title("Task 1-3: L3 Fine-tuning")
    st.markdown(
        """
**概要**: ResNet50の全層を衛星画像で再学習（ファインチューニング）し、
L2の凍結特徴からドメイン適応へ移行。

**結果**: Full FT 10クラス **98.5%**、3クラス **99.6%**
"""
    )

    st.markdown("### L2との違い")
    diff_df = pd.DataFrame(
        {
            "項目": ["バックボーン", "学習対象", "計算負荷", "期待効果"],
            "L2（特徴抽出）": ["凍結", "分類器のみ", "低", "即効で高精度"],
            "L3（FT）": ["更新", "全層（またはヘッド）", "高", "ドメイン最適化"],
        }
    )
    md_table(diff_df)

    st.markdown("### 3実験の詳細")
    exp_df = pd.DataFrame(
        {
            "実験": ["Head Only", "Full FT", "50% Data FT"],
            "学習対象": ["FC層のみ", "全23.5Mパラメータ", "全23.5M（データ半分）"],
            "10クラス精度": [
                pct(l3["accuracy_10class_head"]),
                pct(l3["accuracy_10class_full"]),
                pct(l3["accuracy_10class_half"]),
            ],
        }
    )
    md_table(exp_df)

    st.markdown("### 学習設定の解説")
    st.markdown(
        """
- **エポック**: 全学習データ1周。10エポック = 21,600枚を10周。
- **バッチサイズ64**: 21,600枚を337バッチに分割。更新回数は `337×10=3,370`。
- **Adam**: 勢いを考慮してパラメータ更新する最適化手法。
- **CosineAnnealing**: 前半は大きく、後半は小さく学習率を下げる。
- **データ拡張**: 回転・反転で擬似データを増やして汎化を改善。
- **正規化**: `(値-平均)/標準偏差` でスケールを揃える（偏差値と同じ原理）。
"""
    )

    st.markdown("### GPUが必要な理由")
    st.markdown(
        "CPUは少数コアの逐次処理、GPUは数千コアの並列処理。"
        "2350万パラメータ更新を3370回繰り返すFull FTはGPU前提。"
    )

    st.markdown("### インタラクティブ学習曲線（Test Accuracy）")
    epochs = np.arange(1, len(l3["hist_head_test_acc"]) + 1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=np.array(l3["hist_head_test_acc"]) * 100,
            mode="lines+markers",
            name="Head Only",
            line=dict(color="#e67e22", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=np.array(l3["hist_full_test_acc"]) * 100,
            mode="lines+markers",
            name="Full FT",
            line=dict(color="#27ae60", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=np.array(l3["hist_half_test_acc"]) * 100,
            mode="lines+markers",
            name="50% Data FT",
            line=dict(color="#2980b9", width=3),
        )
    )
    fig.update_layout(
        title="3実験のテスト精度推移",
        xaxis_title="Epoch",
        yaxis_title="Test Accuracy (%)",
        yaxis_range=[85, 100],
        height=460,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 結果テーブル + 分類レポート（Full FT）")
    result_df = pd.DataFrame(
        {
            "手法": ["L1", "L2", "L3 Full FT"],
            "3クラス精度": ["64.4%", "98.1%", "99.6%"],
            "10クラス精度": ["N/A", "95.7%", "98.5%"],
        }
    )
    md_table(result_df)

    report_df = pd.DataFrame(
        {
            "クラス": [
                "AnnualCrop",
                "Forest",
                "HerbaceousVegetation",
                "Highway",
                "Industrial",
                "Pasture",
                "PermanentCrop",
                "Residential",
                "River",
                "SeaLake",
            ],
            "Precision": [0.967, 1.000, 0.972, 0.986, 0.986, 0.987, 0.966, 0.998, 0.992, 0.992],
            "Recall": [0.977, 0.990, 0.985, 0.990, 0.994, 0.963, 0.970, 0.990, 0.988, 0.993],
            "F1": [0.972, 0.995, 0.978, 0.988, 0.990, 0.975, 0.968, 0.994, 0.990, 0.993],
        }
    )
    md_table(report_df)

    show_image(TASK_DIRS["L3"] / "L3_learning_curves.png", "L3_learning_curves", "収束速度の差", "Full FTは初期から高精度で飽和。")
    show_image(TASK_DIRS["L3"] / "L3_confusion_matrix.png", "L3_confusion_matrix", "対角線集中", "全クラスで高い再現率。")
    show_image(TASK_DIRS["L3"] / "L3_per_class_accuracy.png", "L3_per_class_accuracy", "Head vs Fullの差", "全層更新の改善幅をクラス別に把握できる。")
    show_image(TASK_DIRS["L3"] / "L3_confusion_diff.png", "L3_confusion_diff", "差分ヒートマップ", "どの誤分類ペアが減ったかが見える。")
    show_image(TASK_DIRS["L3"] / "L3_correct_vs_incorrect.png", "L3_correct_vs_incorrect", "誤分類サンプルの質", "残存エラーは人間にも難しい境界事例。")

    st.markdown("### 学習曲線の読み方")
    st.markdown("最初の1〜3エポックで急改善し、その後は緩やかに収束。典型的な収穫逓減パターン。")

    st.markdown("### なぜ Head Only (92.4%) < L2 (95.7%) か")
    st.markdown(
        """
- L2は全データを一括で使うロジスティック回帰最適化（大域的に解きやすい）。
- Head Onlyは64枚ミニバッチ更新で局所的・反復的に最適化。
- NN向け汎用オプティマイザは、2万パラメータの線形分類には過剰で非効率になることがある。
"""
    )


def page_task14():
    st.title("Task 1-4: L1 vs L2 vs L3 比較")
    st.markdown(
        """
**概要**: L1/L2/L3の結果を横並びで比較し、
「どこで何がどれだけ改善したか」を定量・定性の両面で評価。
"""
    )

    st.markdown("### 精度比較（5手法）")
    comp_df = pd.DataFrame(
        {
            "手法": [
                "L1: NDVI/NDWI/NDBI Threshold",
                "L2: ResNet50 + LogReg",
                "L3: FT Head Only",
                "L3: FT Full (100% data)",
                "L3: FT Full (50% data)",
            ],
            "3クラス精度": ["64.4%", "98.1%", "-", "99.6%", "-"],
            "10クラス精度": ["N/A", "95.7%", "92.4%", "98.5%", "98.0%"],
            "改善幅": ["-", "+31.3pt (vs L1)", "-3.3pt (vs L2)", "+2.8pt (vs L2)", "+2.3pt (vs L2)"],
        }
    )
    md_table(comp_df)

    st.markdown("### 5つの主要な発見")
    findings = [
        "L1→L2 が最大のジャンプ（+31.3pt）。",
        "L2→L3 Full FT は +2.8pt の上積み。",
        "Head Only は L2未満で、凍結条件ならLogRegが強い。",
        "Full FTが最高精度（98.5%）。",
        "学習データ50%でも98.0%（EuroSATのクリーンさを反映）。",
    ]
    for i, f in enumerate(findings, 1):
        st.markdown(f"{i}. {f}")

    show_image(TASK_DIRS["CMP"] / "L1_L2_L3_comparison_bar.png", "L1_L2_L3_comparison_bar", "L1→L2→L3の段差", "最も効く改善ステップが一目でわかる。")
    show_image(TASK_DIRS["CMP"] / "effort_vs_accuracy.png", "effort_vs_accuracy", "労力と精度の位置関係", "実務での投資対効果判断に直結する。")
    show_image(TASK_DIRS["CMP"] / "L2_vs_L3_confusion.png", "L2_vs_L3_confusion", "誤分類セルの減少", "FTでどの混同が減ったか確認できる。")
    show_image(TASK_DIRS["CMP"] / "L2_vs_L3_radar.png", "L2_vs_L3_radar", "L2とL3の面積差", "L3が全方向で上回る傾向を把握できる。")
    show_image(TASK_DIRS["CMP"] / "3class_comparison.png", "3class_comparison", "L1/L2/L3の3クラス混同行列差", "Built-up崩壊がL2/L3で解消したことが明確。")
    show_image(TASK_DIRS["CMP"] / "same_image_comparison.png", "same_image_comparison", "同一画像の予測変化", "手法進化に伴う予測改善を直感的に追える。")
    show_image(TASK_DIRS["CMP"] / "comparison_6samples_L1L2L3.png", "comparison_6samples_L1L2L3", "6サンプル横並び", "クラス横断の改善傾向を確認できる。")
    show_image(TASK_DIRS["CMP"] / "comparison_improvement_map.png", "comparison_improvement_map", "クラス別改善量", "FTの恩恵が大きいクラスを特定できる。")
    show_image(TASK_DIRS["CMP"] / "comparison_L3_failures.png", "comparison_L3_failures", "L3でも誤る画像", "残課題がデータ曖昧性にあるとわかる。")
    show_image(TASK_DIRS["CMP"] / "comparison_L3_error_rate.png", "comparison_L3_error_rate", "クラス別エラー率", "次に改善すべきクラス優先度を決められる。")

    st.markdown("### きれいなデータ vs 実務データの精度差")
    clean_vs_real = pd.DataFrame(
        {
            "データ量": ["1,000枚", "5,000枚", "20,000枚"],
            "EuroSAT": ["約95%", "約97%", "98.5%"],
            "実務データ": ["約78%", "約86%", "約91%"],
        }
    )
    md_table(clean_vs_real)

    st.markdown("### ラベル作りのコスト")
    st.markdown(
        "衛星画像自体は無料で取得可能だが、"
        "高品質ラベル作成（特にセグメンテーション）は1枚30分〜1時間の人手コストが支配的。"
    )


def page_task15(allm):
    st.title("Task 1-5: All Models Comparison")
    st.markdown(
        """
**概要**: 古典ML〜DLまで全8手法を横並び比較し、
モデル選定の実務判断に使える精度・コスト地図を作成。
"""
    )

    st.markdown("### 古典ML vs DL の本質")
    st.markdown(
        """
- **古典ML**: 人間が特徴量を設計し、モデルは判定だけを担当。
- **DL**: モデルが特徴抽出と判定を同時に学習。深い層で段階的に高次特徴を構築。
"""
    )

    st.markdown("### 古典ML向け手作り特徴量（32次元）")
    feat_df = pd.DataFrame(
        {
            "特徴群": ["13バンド平均", "13バンド標準偏差", "NDVI/NDWI/NDBI平均", "NDVI/NDWI/NDBI標準偏差"],
            "次元": [13, 13, 3, 3],
            "合計": ["", "", "", "32"],
        }
    )
    md_table(feat_df)

    st.markdown("### 全8モデル概要")
    all_models_df = pd.DataFrame(
        {
            "モデル": [
                "L1: Index Threshold",
                "Random Forest",
                "SVM (RBF)",
                "k-NN (k=5)",
                "Simple CNN",
                "L2: ResNet50 + LogReg",
                "L3: ResNet50 FT",
                "ViT-Tiny FT",
            ],
            "カテゴリ": ["L1", "Classical ML", "Classical ML", "Classical ML", "Deep Learning", "L2", "L3", "Deep Learning"],
            "精度": [64.4, 90.2, 93.8, 90.3, 94.9, 95.7, 98.5, 98.5],
        }
    )
    md_table(all_models_df)

    st.markdown("### 精度ランキング（インタラクティブ）")
    chart_df = all_models_df.sort_values("精度", ascending=True).copy()
    color_map = {
        "L1": "#c0392b",
        "Classical ML": "#f39c12",
        "L2": "#2980b9",
        "L3": "#27ae60",
        "Deep Learning": "#8e44ad",
    }
    colors = [color_map[c] for c in chart_df["カテゴリ"]]

    fig = go.Figure(
        go.Bar(
            x=chart_df["精度"],
            y=chart_df["モデル"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in chart_df["精度"]],
            textposition="outside",
            hovertemplate="%{y}<br>精度: %{x:.1f}%<br>カテゴリ: %{customdata}<extra></extra>",
            customdata=chart_df["カテゴリ"],
        )
    )
    fig.update_layout(title="全モデル精度ランキング", xaxis_title="Accuracy (%)", yaxis_title="", height=520)
    st.plotly_chart(fig, use_container_width=True)

    if allm is not None:
        with st.expander("all_models_results.npz の実測値"):
            npz_df = pd.DataFrame(
                {
                    "モデル": list(allm["model_names"]),
                    "精度": [round(float(x) * 100, 2) for x in allm["model_accs"]],
                    "タイプ": list(allm["model_types"]),
                }
            )
            md_table(npz_df)

    show_image(TASK_DIRS["ALL"] / "all_models_comparison.png", "all_models_comparison", "カテゴリ別の精度階段", "L1→Classical→DL/FT で段階的に伸びる。")
    show_image(TASK_DIRS["ALL"] / "all_models_ranking.png", "all_models_ranking", "順位の確定", "最終候補モデルを短時間で絞り込める。")
    show_image(TASK_DIRS["ALL"] / "all_models_class_heatmap.png", "all_models_class_heatmap", "クラス別の苦手分布", "モデル選定時にクラス別リスクを評価できる。")
    show_image(TASK_DIRS["ALL"] / "all_models_cost_vs_accuracy.png", "all_models_cost_vs_accuracy", "学習時間と精度の関係", "精度だけでなくコスト制約下の最適点を選べる。")

    st.markdown("### 事前学習 + FT が強い理由")
    st.markdown(
        "ImageNet 120万枚で得た視覚知識を初期値にすることで、"
        "EuroSAT 27,000枚だけでは学びきれないパターンを利用できるため。"
    )

    st.markdown("### 実務でのモデル選定フロー")
    st.markdown(
        "1. まずResNet50/ViTをFTしてベースライン作成\n"
        "2. 要件を満たせば運用へ\n"
        "3. 不足する場合はデータ追加・ラベル改善・衛星専用モデルへ拡張"
    )


def main():
    l1 = load_npz(TASK_DIRS["L1"] / "L1_results.npz")
    l2 = load_npz(TASK_DIRS["L2"] / "L2_results.npz")
    l3 = load_npz(TASK_DIRS["L3"] / "L3_results.npz")
    allm = load_npz(TASK_DIRS["ALL"] / "all_models_results.npz")

    st.sidebar.title("ナビゲーション")
    page = st.sidebar.radio(
        "ページを選択",
        [
            "1. Overview",
            "2. Task 1-1 L1: Index Threshold",
            "3. Task 1-2 L2: Pretrained Model",
            "4. Task 1-3 L3: Fine-tuning",
            "5. Task 1-4 L1 vs L2 vs L3 比較",
            "6. Task 1-5 All Models Comparison",
        ],
    )

    if page.startswith("1"):
        page_overview(l1, l2, l3)
    elif page.startswith("2"):
        page_task11(l1)
    elif page.startswith("3"):
        page_task12(l2)
    elif page.startswith("4"):
        page_task13(l3)
    elif page.startswith("5"):
        page_task14()
    else:
        page_task15(allm)


if __name__ == "__main__":
    main()
