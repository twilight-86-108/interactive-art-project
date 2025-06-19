# ファイル: src/core/app.py
# 時間: 3-4時間 | 優先度: 🔴 最高

import pygame
import sys
import time
import cv2
from camera_manager import CameraManager

class AquaMirrorApp:
    """メインアプリケーション"""
    
    def __init__(self):
        self.running = True
        self.clock = pygame.time.Clock()
        self.target_fps = 30
        
        # 初期化
        self._init_pygame()
        self.camera_manager = CameraManager()
        
    def _init_pygame(self):
        """Pygame初期化"""
        pygame.init()
        
        # 24インチモニター設定
        self.screen_width = 1920
        self.screen_height = 1080
        
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height),
            pygame.DOUBLEBUF | pygame.HWSURFACE
        )
        pygame.display.set_caption("Aqua Mirror - Development")
        
    def initialize(self) -> bool:
        """アプリケーション初期化"""
        try:
            # カメラ初期化
            if not self.camera_manager.initialize():
                return False
            
            self.camera_manager.start_capture()
            print("アプリケーション初期化完了")
            return True
            
        except Exception as e:
            print(f"初期化エラー: {e}")
            return False
    
    def run(self):
        """メインループ"""
        if not self.initialize():
            print("初期化失敗")
            return
        
        try:
            while self.running:
                self._handle_events()
                self._update()
                self._render()
                self.clock.tick(self.target_fps)
                
        except KeyboardInterrupt:
            print("ユーザー停止")
        except Exception as e:
            print(f"実行エラー: {e}")
        finally:
            self._cleanup()
    
    def _handle_events(self):
        """イベント処理"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def _update(self):
        """状態更新"""
        # フレーム取得
        frame = self.camera_manager.get_frame()
        if frame is not None:
            self.current_frame = frame
    
    def _render(self):
        """描画処理"""
        self.screen.fill((0, 0, 0))  # 黒背景
        
        # カメラ画像表示（デバッグ用）
        if hasattr(self, 'current_frame'):
            # OpenCV -> Pygame 変換
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            
            # 画面サイズに合わせてスケール
            scaled_surface = pygame.transform.scale(frame_surface, (self.screen_width, self.screen_height))
            self.screen.blit(scaled_surface, (0, 0))
        
        # FPS表示
        fps = self.clock.get_fps()
        font = pygame.font.Font(None, 36)
        fps_text = font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
        self.screen.blit(fps_text, (10, 10))
        
        pygame.display.flip()
    
    def _cleanup(self):
        """クリーンアップ"""
        self.camera_manager.cleanup()
        pygame.quit()
        print("アプリケーション終了")

# メインエントリーポイント
if __name__ == "__main__":
    app = AquaMirrorApp()
    app.run()