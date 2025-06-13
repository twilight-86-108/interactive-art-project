# src/app.py
import pygame
import time
import sys
from enum import Enum
from vision import VisionProcessor
from scene import SceneManager

class AppState(Enum):
    """アプリケーション状態の定義"""
    STANDBY = "standby"
    RECOGNITION = "recognition"
    INTERACTION = "interaction"
    EXPERIENCE_END = "experience_end"

class App:
    """メインアプリケーションクラス"""
    
    def __init__(self, config):
        self.config = config
        self.running = True
        self.clock = pygame.time.Clock()
        self.current_state = AppState.STANDBY
        self.state_timer = 0
        self.last_detection_time = 0
        
        # デバッグフラグ
        self.debug_mode = False
        
        # 各モジュールの初期化
        self._init_pygame()
        self.vision_processor = VisionProcessor(config)
        self.scene_manager = SceneManager(
            self.config['display']['width'],
            self.config['display']['height'],
            config
        )
        
        print("アプリケーションが初期化されました")
    
    def _init_pygame(self):
        """Pygame初期化"""
        pygame.init()
        pygame.mixer.init()
        
        display_config = self.config['display']
        if display_config['fullscreen']:
            self.screen = pygame.display.set_mode(
                (display_config['width'], display_config['height']),
                pygame.FULLSCREEN
            )
        else:
            self.screen = pygame.display.set_mode(
                (display_config['width'], display_config['height'])
            )
        
        pygame.display.set_caption("Interactive Art - Living Window")
    
    def run(self):
        """メインループ"""
        try:
            while self.running:
                frame_start = time.time()
                
                # イベント処理
                self.handle_events()
                
                # 状態更新
                self.update()
                
                # 描画
                self.draw()
                
                # フレームレート制御
                self.clock.tick(30)
                
        except Exception as e:
            print(f"実行中にエラーが発生しました: {e}")
        finally:
            self.cleanup()
    
    def handle_events(self):
        """イベント処理"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_F1:
                    self.debug_mode = not self.debug_mode
                    print(f"デバッグモード: {'ON' if self.debug_mode else 'OFF'}")
                elif event.key == pygame.K_c:
                    # 設定ファイル再読み込み（実装予定）
                    print("設定ファイル再読み込み（未実装）")
    
    def update(self):
        """状態更新"""
        current_time = time.time()
        
        # 顔・手検出の実行
        detection_result = self.vision_processor.process_frame()
        
        # 状態遷移ロジック
        self._update_state(detection_result, current_time)
        
        # シーンマネージャーに検出結果を渡す
        self.scene_manager.update(detection_result, self.current_state)
    
    def _update_state(self, detection_result, current_time):
        """状態遷移の管理"""
        face_detected = detection_result.get('face_detected', False)
        hands_detected = detection_result.get('hands_detected', False)
        face_distance = detection_result.get('face_distance', float('inf'))
        
        if self.current_state == AppState.STANDBY:
            if face_detected:
                self.current_state = AppState.RECOGNITION
                self.state_timer = current_time
                print("状態変更: STANDBY -> RECOGNITION")
        
        elif self.current_state == AppState.RECOGNITION:
            if not face_detected:
                # 顔が見つからなくなった場合、少し待ってからSTANDBYに戻る
                if current_time - self.last_detection_time > 3.0:
                    self.current_state = AppState.STANDBY
                    print("状態変更: RECOGNITION -> STANDBY")
            else:
                self.last_detection_time = current_time
                
                # インタラクション条件の確認
                approach_threshold = self.config['interaction']['approach_threshold_z']
                if face_distance < approach_threshold or hands_detected:
                    self.current_state = AppState.INTERACTION
                    print("状態変更: RECOGNITION -> INTERACTION")
        
        elif self.current_state == AppState.INTERACTION:
            if not face_detected and not hands_detected:
                self.current_state = AppState.EXPERIENCE_END
                self.state_timer = current_time
                print("状態変更: INTERACTION -> EXPERIENCE_END")
        
        elif self.current_state == AppState.EXPERIENCE_END:
            # 3秒後にSTANDBYに戻る
            if current_time - self.state_timer > 3.0:
                self.current_state = AppState.STANDBY
                print("状態変更: EXPERIENCE_END -> STANDBY")
    
    def draw(self):
        """描画処理"""
        self.scene_manager.draw(self.screen)
        
        # デバッグ情報の描画
        if self.debug_mode:
            self._draw_debug_info()
        
        pygame.display.flip()
    
    def _draw_debug_info(self):
        """デバッグ情報の描画"""
        font = pygame.font.Font(None, 36)
        y_offset = 10
        
        # FPS表示
        fps = self.clock.get_fps()
        fps_text = font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
        self.screen.blit(fps_text, (10, y_offset))
        y_offset += 30
        
        # 現在の状態表示
        state_text = font.render(f"State: {self.current_state.value}", True, (255, 255, 255))
        self.screen.blit(state_text, (10, y_offset))
        y_offset += 30
        
        # 検出結果表示
        detection_info = self.vision_processor.get_debug_info()
        for key, value in detection_info.items():
            info_text = font.render(f"{key}: {value}", True, (255, 255, 255))
            self.screen.blit(info_text, (10, y_offset))
            y_offset += 30
    
    def cleanup(self):
        """クリーンアップ処理"""
        self.vision_processor.cleanup()
        pygame.quit()
        print("アプリケーションが終了しました")