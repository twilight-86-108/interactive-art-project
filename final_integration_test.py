# -*- coding: utf-8 -*-

print('🧪 Week 2 最終統合テスト開始')
print('=' * 80)

try:
    # 全依存関係確認
    import moderngl, cv2, mediapipe, numpy
    print(f'✅ 基本ライブラリ: ModernGL {moderngl.__version__}, OpenCV {cv2.__version__}')

    # 音響ライブラリ確認
    try:
        import sounddevice
        print(f'✅ 音響ライブラリ: SoundDevice {sounddevice.__version__}')
    except ImportError:
        print('⚠️ SoundDevice未インストール（音響機能はスキップ）')

    # 高度コンポーネント初期化テスト
    from src.core.config_loader import ConfigLoader
    from src.effects.advanced_water_simulation import AdvancedWaterSimulation
    from src.audio.advanced_audio_engine import AdvancedAudioEngine

    print('✅ 高度コンポーネントインポート成功')

    # GPU機能確認
    try:
        ctx = moderngl.create_standalone_context()
        print(f'✅ GPU: {ctx.info["GL_RENDERER"]}')
        print(f'✅ OpenGL: {ctx.version_code}')

        # コンピュートシェーダー対応確認
        if ctx.version_code >= 430:
            print('✅ コンピュートシェーダー対応')
        else:
            print('⚠️ コンピュートシェーダー非対応（OpenGL 4.3未満）')

        ctx.release() # リソースを解放

    except Exception as e:
        print(f'⚠️ GPU確認エラー: {e}')

    print('=' * 80)
    print('🎉 Week 2 最終統合テスト完了')

except Exception as e:
    print(f'❌ テストエラー: {e}')