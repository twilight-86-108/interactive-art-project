@echo off
echo 🧪 Aqua Mirror 移行完全テスト
echo ===============================

echo 📦 1. 依存関係確認...
python -c "
packages = ['pygame', 'moderngl', 'numpy', 'opencv-python', 'mediapipe']
missing = []
for pkg in packages:
    try:
        __import__(pkg.replace('-', '_'))
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg}')
        missing.append(pkg)

if missing:
    print(f'\n⚠️ 不足: {missing}')
    print('install_dependencies.bat を実行してください')
else:
    print('\n✅ すべての依存関係OK')
"

echo.
echo 🔍 2. GPU確認...
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo.
echo 🎮 3. Pygame + ModernGL テスト...
python -c "
try:
    import pygame
    import moderngl
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_ACCELERATED_VISUAL, 1)
    screen = pygame.display.set_mode((1, 1), pygame.OPENGL | pygame.HIDDEN)
    ctx = moderngl.create_context()
    gpu = ctx.info.get('GL_RENDERER', 'Unknown')
    print(f'GPU: {gpu}')
    if 'NVIDIA' in gpu:
        print('✅ NVIDIA GPU 検出')
    else:
        print('⚠️ NVIDIA GPU 未検出')
    pygame.quit()
except Exception as e:
    print(f'❌ テスト失敗: {e}')
"

echo.
echo ===============================
pause
