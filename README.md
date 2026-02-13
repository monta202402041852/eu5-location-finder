# EU5 Location Finder (Windows)

Europa Universalis V の MOD 画面にある地名リストを監視し、指定語を検出したら**即時アラート表示**して見逃しを防ぐツールです。

## 特徴

- F9 / F10 で監視範囲をキャリブレーションし `region.json` 保存
- Windows Graphics Capture (`windows-capture`) で指定矩形を安定キャプチャ
- `pytesseract + OpenCV` で日本語 OCR
- 完全一致 → `rapidfuzz.partial_ratio` の順で判定
- 通常モード / 高速モード切替（F5）
- 検出時に最前面オーバーレイを残留表示 + ビープ
- F6 で解除するまで表示を維持（ヒット後は監視停止）

---

## セットアップ手順

### 1. Python の用意

- Python 3.10 以上推奨
- 仮想環境を推奨

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Tesseract OCR の導入（必須）

1. Windows 用 Tesseract をインストール（例: `C:\Program Files\Tesseract-OCR`）
2. `tessdata` に `jpn.traineddata` があることを確認
3. `main.py` は既定で以下を参照します
   - `C:\Program Files\Tesseract-OCR\tesseract.exe`

別パスの場合は `main.py` 内の `DEFAULT_TESSERACT_PATH` を変更してください。

---

## 使い方

1. ゲームを起動（ボーダーレス / 仮想フルスクリーン推奨）
2. ツール起動

```bash
python main.py
```

3. 監視範囲を設定
   - F9: 左上座標を記録
   - F10: 右下座標を記録して `region.json` 保存
4. F7 で検索語を 1 語入力（例: `マンハッタン`）
5. F8 で監視 ON/OFF
6. F5 で高速モード ON/OFF
7. UI の **Test Alert** でアラート経路を強制確認
8. 検出時はオーバーレイ表示 + ビープ、F6 で解除

---

## ホットキー

- **F9**: 範囲左上記録
- **F10**: 範囲右下記録＆保存
- **F7**: 検索語入力
- **F8**: 監視 ON/OFF
- **F5**: 高速モード ON/OFF
- **F6**: アラート解除

---

## 設定・判定ロジック

- 判定順序
  1. OCR トークンと検索語の完全一致
  2. `rapidfuzz.partial_ratio` による類似一致
- 既定閾値: `86`（`FUZZ_THRESHOLD_DEFAULT`）
- 正規化: NFKC、空白除去、中黒ゆれ（`•`, `･` → `・`）吸収
- 高速スクロール対策
  - 高速モード時はキャプチャ間隔短縮
  - 直近複数フレームの高スコアを加味して発火

---

## トラブルシューティング

### `region.json` がない / 読み込めない

- 起動ログに案内が出ます。
- F9 → F10 で再設定し、`region.json` を再生成してください。

### キャプチャが黒画面

- ゲームをボーダーレス（仮想フルスクリーン）にする
- 管理者権限でツールを起動してみる
- オーバーレイ系ツール（Discord/NVIDIA/Steam）を一時停止して確認
- UI スケール固定を前提に再キャリブレーション

### OCR 精度が低い

- 監視範囲を「地名列のみ」に絞る
- 文字サイズが小さい場合、UI スケールを調整
- `FUZZ_THRESHOLD_DEFAULT` を 82〜92 で調整
- `main.py` の `preprocess()` の二値化パラメータを微調整

### ホットキーが効かない

- 他アプリに同じキー割当がないか確認
- 管理者権限起動を試す

---

## exe 化手順（PyInstaller）

```bash
pip install pyinstaller
pyinstaller --noconfirm --onefile --name eu5-location-finder main.py
```

出力先:
- `dist\eu5-location-finder.exe`

> 実行時には Tesseract 本体と `jpn.traineddata` が必要です。必要に応じてインストーラ同梱またはセットアップ手順を配布してください。

---

## 注意

- このツールは Windows 向けです（`winsound`, `windows-capture`, `keyboard` 使用）。
- ゲームアップデートやフォント変化で OCR 精度が変わる場合があります。定期的に閾値と範囲を見直してください。
