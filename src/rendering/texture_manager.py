"""
GPU テクスチャマネージャー
カメラフレーム→OpenGL テクスチャ高速変換
"""

import moderngl
import numpy as np
import logging
import time
from typing import Optional, Dict, Tuple, Any

# 最適化案：importはファイルのトップレベルにまとめるのが一般的
# import cv2 

class TextureManager:
    """
    GPU テクスチャ管理・OpenGL統合
    カメラフレーム→GPU転送最適化
    """
    
    def __init__(self, ctx: moderngl.Context, config):
        self.ctx = ctx
        self.config = config
        self.logger = logging.getLogger("TextureManager")
        
        # テクスチャ管理
        self.textures: Dict[str, moderngl.Texture] = {}
        # self.texture_cache は使われていないようなので削除も検討
        self.texture_cache: Dict[str, moderngl.Texture] = {}
        
        # カメラテクスチャ設定
        self.camera_width = config.get('camera.width', 1920)
        self.camera_height = config.get('camera.height', 1080)
        
        # パフォーマンス統計
        self.upload_count = 0
        self.upload_time_total = 0.0
        self.last_upload_time = 0.0
        
        # 最適化案：環境に依存しないCLAMPモードを初期化時に一度だけ決定する
        self.clamp_mode = self._get_clamp_mode()
        
        # カメラテクスチャ事前作成
        self._create_camera_texture()
        
    def _get_clamp_mode(self) -> int:
        """
        利用可能なCLAMP_TO_EDGEの値を取得する。
        環境によって定数が存在しない問題に対応するためのヘルパーメソッド。
        """
        if hasattr(moderngl, 'CLAMP_TO_EDGE'):
            return moderngl.CLAMP_TO_EDGE
        else:
            self.logger.warning(
                "moderngl.CLAMP_TO_EDGE が見つかりません。フォールバック値(33071)を使用します。"
            )
            return 33071

    def _create_camera_texture(self):
        """カメラフレーム用テクスチャ事前作成"""
        try:
            camera_texture = self.ctx.texture(
                (self.camera_width, self.camera_height), 
                3,  # RGB
                data=None,
                dtype='u1'
            )
            
            camera_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            
            # 最適化案：マジックナンバーの代わりに、初期化時に決定した変数を使用
            camera_texture.wrap_x = self.clamp_mode
            camera_texture.wrap_y = self.clamp_mode
            
            self.textures['camera_input'] = camera_texture
            self.logger.info(f"✅ カメラテクスチャ作成: {self.camera_width}x{self.camera_height}")
            
        except Exception as e:
            self.logger.error(f"❌ カメラテクスチャ作成失敗: {e}")
            raise
    
    def upload_camera_frame(self, frame: np.ndarray) -> bool:
        """
        カメラフレーム→GPU テクスチャ高速アップロード
        （中略）
        """
        if frame is None:
            return False

        upload_start = time.time()
        
        try:
            camera_texture = self.textures['camera_input']
            
            # np.flipud() は非連続な配列ビューを返すため、.copy() を呼び出して
            # C-contiguous な（メモリ上で連続した）配列を作成する
            flipped_frame = np.flipud(frame).copy()
            
            # これで連続したデータとなり、writeメソッドに渡せる
            camera_texture.write(flipped_frame)
            
            # 統計更新
            upload_time = time.time() - upload_start
            self.upload_count += 1
            self.upload_time_total += upload_time
            self.last_upload_time = upload_time
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ フレームアップロードエラー: {e}")
            return False
    
    def get_camera_texture(self) -> Optional[moderngl.Texture]:
        """カメラテクスチャ取得"""
        return self.textures.get('camera_input')
    
    def create_render_target(self, name: str, width: int, height: int, 
                             components: int = 4, dtype: str = 'f4') -> moderngl.Texture:
        """
        レンダーターゲットテクスチャ作成
        最適化：dtypeを指定可能に
        """
        try:
            # RGBA、浮動小数点数(f4)がレンダーターゲットでは一般的
            texture = self.ctx.texture((width, height), components, dtype=dtype)
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            
            # 最適化案：一貫性を保つため、ここでも self.clamp_mode を使用
            texture.wrap_x = self.clamp_mode
            texture.wrap_y = self.clamp_mode
            
            self.textures[name] = texture
            self.logger.info(f"✅ レンダーターゲット '{name}' 作成: {width}x{height}")
            return texture
            
        except Exception as e:
            self.logger.error(f"❌ レンダーターゲット '{name}' 作成失敗: {e}")
            raise
    
    def bind_texture(self, name: str, unit: int = 0) -> bool:
        """テクスチャバインド"""
        texture = self.textures.get(name)
        if not texture:
            self.logger.warning(f"テクスチャ '{name}' が見つかりません")
            return False
        
        try:
            texture.use(unit)
            return True
        except Exception as e:
            self.logger.error(f"テクスチャ '{name}' バインドエラー: {e}")
            return False
    
    def get_upload_stats(self) -> Dict[str, Any]:
        """アップロード統計取得"""
        if self.upload_count > 0:
            avg_upload_time = self.upload_time_total / self.upload_count
        else:
            avg_upload_time = 0.0
        
        return {
            'upload_count': self.upload_count,
            'last_upload_time_ms': self.last_upload_time * 1000,
            'avg_upload_time_ms': avg_upload_time * 1000,
            'upload_fps': 1.0 / avg_upload_time if avg_upload_time > 0 else 0
        }
    
    def cleanup_texture(self, name: str):
        """テクスチャ解放"""
        if name in self.textures:
            try:
                self.textures[name].release()
                del self.textures[name]
                self.logger.info(f"✅ テクスチャ '{name}' 解放")
            except Exception as e:
                self.logger.error(f"テクスチャ '{name}' 解放エラー: {e}")
    
    def cleanup_all(self):
        """全テクスチャ解放"""
        # 繰り返し中に辞書を変更しても安全なようにキーのリストを先に作成
        texture_names = list(self.textures.keys())
        for name in texture_names:
            self.cleanup_texture(name)
        
        self.logger.info("✅ 全テクスチャ解放完了")