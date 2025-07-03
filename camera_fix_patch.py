
# カメラ修正パッチ
# pygame_moderngl_app_complete.py に追加する修正

def _process_camera_frame_fixed(self):
    """カメラフレーム処理（修正版）"""
    if not self.camera_enabled or not self.camera:
        return
    
    try:
        frame = self.camera.get_frame()
        if frame is not None:
            # フレーム前処理
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # BGR → RGB 変換
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # フレームサイズ調整
                target_height, target_width = 1080, 1920
                if frame_rgb.shape[:2] != (target_height, target_width):
                    frame_rgb = cv2.resize(frame_rgb, (target_width, target_height))
                
                # テクスチャアップロード
                if self.texture_manager:
                    self.texture_manager.upload_camera_frame(frame_rgb)
                    self.logger.debug(f"カメラフレーム更新: {frame_rgb.shape}")
            else:
                self.logger.warning(f"未対応フレーム形状: {frame.shape}")
                
    except Exception as e:
        self.logger.error(f"カメラフレーム処理エラー: {e}")

def _create_camera_texture_fixed(self):
    """カメラテクスチャ作成（修正版）"""
    if not self.ctx:
        return None
    
    try:
        # RGB テクスチャ作成
        camera_texture = self.ctx.texture((1920, 1080), 3)
        camera_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        camera_texture.wrap_x = moderngl.CLAMP_TO_EDGE
        camera_texture.wrap_y = moderngl.CLAMP_TO_EDGE
        
        # 初期データ（黒画面）
        import numpy as np
        initial_data = np.zeros((1080, 1920, 3), dtype=np.uint8)
        camera_texture.write(initial_data.tobytes())
        
        return camera_texture
        
    except Exception as e:
        self.logger.error(f"カメラテクスチャ作成エラー: {e}")
        return None
