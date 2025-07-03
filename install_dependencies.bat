@echo off
echo 📦 依存関係インストール
echo =======================

REM 仮想環境有効化
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo 仮想環境を作成しています...
    python -m venv venv
    call venv\Scripts\activate.bat
)

REM 基本パッケージ更新
echo 📦 基本パッケージ更新中...
python -m pip install --upgrade pip

REM 必要なパッケージインストール
echo 📦 必要なパッケージインストール中...
pip install pygame==2.5.2
pip install moderngl==5.8.2
pip install numpy==1.26.4
pip install rich==13.7.1

REM requirements.txt があればインストール
if exist "requirements.txt" (
    echo 📦 requirements.txt からインストール中...
    pip install -r requirements.txt
)

echo ✅ 依存関係インストール完了
pause
