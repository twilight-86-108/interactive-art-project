# =================================================================
# スクリプト全体の設定
# =================================================================
# 文字化け対策
$OutputEncoding = [System.Text.Encoding]::UTF8
# スクリプトの場所を基準に実行
Set-Location -Path $PSScriptRoot


# =================================================================
# Test 1: 最終統合テスト
# =================================================================
# 独立したPythonファイルを実行する
python final_integration_test.py


# テスト間の区切り
Write-Host "`n" # 改行
Write-Host "---"
Write-Host "`n" # 改行


# =================================================================
# Test 2: 高度アプリケーション短時間実行テスト
# =================================================================
# Start-Jobは確実ですが、今回はより軽量なStart-Processを使い、
# 以前の回答で解説した堅牢なタイムアウト処理を行います。
Write-Host "🚀 高度アプリケーション実行テスト（15秒間）..." -ForegroundColor Green

# python main.py を別プロセスとして起動
$process = Start-Process "python" "main.py" -PassThru -NoNewWindow -ErrorAction SilentlyContinue

if ($process) {
    # 15秒待機し、タイムアウトしたらプロセスを停止
    if ($process.WaitForExit(15000)) {
        # 15秒以内にプロセスが予期せず終了した場合
        Write-Host "⚠️ アプリケーションは15秒以内に終了しました。(終了コード: $($process.ExitCode))" -ForegroundColor Yellow
    } else {
        # 15秒間、正常に実行され続けた場合
        Stop-Process -Id $process.Id -Force
        Write-Host "✅ 高度アプリケーション実行確認完了（15秒タイムアウト）" -ForegroundColor Green
    }
} else {
    # プロセスの起動自体に失敗した場合
    Write-Host "❌ python main.py の起動に失敗しました。" -ForegroundColor Red
}