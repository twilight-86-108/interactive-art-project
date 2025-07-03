
# FPS修正パッチ
# パフォーマンス最適化コード

def _optimize_fps_performance(self):
    """FPS最適化設定"""
    
    # OpenGL最適化設定
    if self.ctx:
        # 深度テスト無効化（2D描画のみの場合）
        self.ctx.disable(moderngl.DEPTH_TEST)
        
        # カリング有効化
        self.ctx.enable(moderngl.CULL_FACE)
        
        # ビューポート設定
        self.ctx.viewport = (0, 0, 1920, 1080)
    
    # Pygame最適化
    if hasattr(pygame.display, 'gl_set_swap_interval'):
        pygame.display.gl_set_swap_interval(1)  # VSync
    
    # フレーム制限解除（GPU制限で自然に制限される）
    return 144  # 最大FPS目標

def _update_frame_optimized(self):
    """最適化されたフレーム更新"""
    frame_start = time.perf_counter()
    
    # イベント処理（軽量化）
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
    
    # 描画頻度調整（AI処理は30FPSに制限）
    current_time = time.perf_counter()
    if current_time - getattr(self, '_last_ai_time', 0) > 1.0/30.0:
        self._process_ai_frame()
        self._last_ai_time = current_time
    
    # カメラは60FPS
    self._process_camera_frame()
    
    # レンダリング（最高FPS）
    self._render_frame()
    
    # バッファスワップ
    pygame.display.flip()
    
    # フレーム時間測定
    frame_time = time.perf_counter() - frame_start
    self._update_fps_stats(frame_time)

def _render_frame_optimized(self):
    """最適化された描画"""
    # 高速クリア
    self.ctx.clear(0.0, 0.0, 0.0, 1.0)
    
    current_time = time.perf_counter()
    
    # シェーダー選択（条件分岐最小化）
    if self.camera_enabled and self.texture_manager:
        program = self.camera_program
        program['u_time'] = current_time
        program['u_camera_enabled'] = True
        program['u_effect_strength'] = 1.0
        
        # テクスチャバインド
        camera_texture = self.texture_manager.get_camera_texture()
        if camera_texture:
            camera_texture.use(0)
            program['u_camera_texture'] = 0
    else:
        program = self.water_program
        program['u_time'] = current_time
        program['u_color'] = (0.3, 0.6, 1.0)
        program['u_show_test_pattern'] = False
    
    # 単一描画コール
    self.quad_vao.render()

def _update_fps_stats(self, frame_time):
    """FPS統計更新"""
    if not hasattr(self, '_fps_samples'):
        self._fps_samples = []
        self._last_fps_log = time.perf_counter()
    
    fps = 1.0 / frame_time if frame_time > 0 else 0
    self._fps_samples.append(fps)
    
    # 5秒ごとに統計表示
    current_time = time.perf_counter()
    if current_time - self._last_fps_log > 5.0:
        if self._fps_samples:
            avg_fps = sum(self._fps_samples) / len(self._fps_samples)
            min_fps = min(self._fps_samples)
            max_fps = max(self._fps_samples)
            
            self.logger.info(f"FPS統計: 平均{avg_fps:.1f} 最小{min_fps:.1f} 最大{max_fps:.1f}")
            
            self._fps_samples = []
            self._last_fps_log = current_time
