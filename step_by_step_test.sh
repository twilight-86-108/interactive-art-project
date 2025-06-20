#!/bin/bash
# step_by_step_test.sh - 段階的動作確認スクリプト

echo "🔍 Aqua Mirror 段階的動作確認"
echo "=" * 50

# 環境変数設定（警告抑制）
export TF_CPP_MIN_LOG_LEVEL=2
export SDL_VIDEODRIVER=x11

echo "📋 ステップ1: コンポーネントテスト"
python main.py --test
test_result=$?

if [ $test_result -eq 0 ]; then
    echo "✅ コンポーネントテスト成功"
    
    echo ""
    echo "📋 ステップ2: 5秒デモモード"
    python main.py --demo
    demo_result=$?
    
    if [ $demo_result -eq 0 ]; then
        echo "✅ デモモード成功"
        
        echo ""
        echo "📋 ステップ3: フルアプリケーション（10秒）"
        echo "💡 ESCキーで終了できます"
        
        # 10秒後に自動終了
        timeout 10 python main.py --debug
        app_result=$?
        
        if [ $app_result -eq 0 ] || [ $app_result -eq 124 ]; then  # 124 = timeout
            echo "✅ フルアプリケーション動作確認"
            echo ""
            echo "🎉 すべてのテストが成功しました！"
            echo "🚀 Aqua Mirror の開発を継続できます"
        else
            echo "❌ フルアプリケーションで問題発生"
            echo "🔧 デバッグが必要です"
        fi
    else
        echo "❌ デモモードで問題発生"
        echo "🔧 Pygame または VisionProcessor の確認が必要"
    fi
else
    echo "❌ コンポーネントテストで問題発生"
    echo "🔧 基本モジュールの確認が必要"
fi

echo ""
echo "📋 トラブルシューティング情報:"
echo "   --test : コンポーネント個別確認"
echo "   --demo : 5秒間デモモード"
echo "   --debug: デバッグ情報表示"
echo "   --no-camera: カメラなしモード"