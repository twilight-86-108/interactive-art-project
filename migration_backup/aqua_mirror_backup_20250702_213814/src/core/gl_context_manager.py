"""
OpenGL/ModernGLコンテキスト管理システム - 修正版
"""

import moderngl
import glfw
import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np

class GLContextManager:
    """
    ModernGL コンテキスト管理クラス
    OpenGL状態管理・リソース管理
    """
    
    def __init__(self, ctx: moderngl.Context, config):
        self.ctx = ctx
        self.config = config
        self.logger = logging.getLogger("GLContextManager")
        
        # GPU情報
        self.gpu_info = self._get_gpu_info()
        
        # リソース管理
        self.textures: Dict[str, moderngl.Texture] = {}
        self.framebuffers: Dict[str, moderngl.Framebuffer] = {}
        self.vertex_arrays: Dict[str, moderngl.VertexArray] = {}
        self.buffers: Dict[str, moderngl.Buffer] = {}
        
        # メモリ使用量追跡
        self.memory_usage = {
            'textures': 0,
            'buffers': 0,
            'total': 0
        }
        
        self._setup_opengl_state()
        
    def _get_gpu_info(self) -> Dict[str, Any]:
        """GPU情報取得 - 安全版"""
        try:
            # 必須情報
            gpu_info = {
                'renderer': self.ctx.info.get('GL_RENDERER', 'Unknown'),
                'version': self.ctx.info.get('GL_VERSION', 'Unknown'),
                'vendor': self.ctx.info.get('GL_VENDOR', 'Unknown'),
                'glsl_version': self.ctx.info.get('GL_SHADING_LANGUAGE_VERSION', 'Unknown'),
                'extensions': []
            }
            
            self.logger.info(f"GPU情報取得完了: {gpu_info['renderer']}")
            return gpu_info
            
        except Exception as e:
            self.logger.error(f"GPU情報取得エラー: {e}")
            return {
                'renderer': 'Unknown',
                'version': 'Unknown',
                'vendor': 'Unknown',
                'glsl_version': 'Unknown',
                'extensions': []
            }
    
    def _setup_opengl_state(self):
        """OpenGL初期状態設定 - 簡素版"""
        try:
            # 基本的なOpenGL状態のみ設定
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            
            # ビューポート設定
            width = self.config.get('app.window.width', 1920)
            height = self.config.get('app.window.height', 1080)
            self.ctx.viewport = (0, 0, width, height)
            
            self.logger.info("OpenGL基本状態設定完了")
            
        except Exception as e:
            self.logger.warning(f"OpenGL状態設定エラー（継続）: {e}")
    
    def create_texture(self, name: str, width: int, height: int, 
                      components: int = 4) -> moderngl.Texture:
        """テクスチャ作成・管理 - 簡素版"""
        if name in self.textures:
            self.logger.warning(f"テクスチャ '{name}' は既に存在します")
            return self.textures[name]
        
        try:
            texture = self.ctx.texture((width, height), components)
            texture.filter = moderngl.LINEAR, moderngl.LINEAR
            texture.wrap_x = moderngl.CLAMP_TO_EDGE
            texture.wrap_y = moderngl.CLAMP_TO_EDGE
            
            self.textures[name] = texture
            self.logger.info(f"テクスチャ '{name}' 作成: {width}x{height}")
            return texture
            
        except Exception as e:
            self.logger.error(f"テクスチャ '{name}' 作成失敗: {e}")
            raise
    
    def create_vertex_array(self, name: str, program: moderngl.Program, 
                           vertices: np.ndarray, indices: Optional[np.ndarray] = None) -> moderngl.VertexArray:
        """頂点配列作成・管理 - 簡素版"""
        if name in self.vertex_arrays:
            self.logger.warning(f"頂点配列 '{name}' は既に存在します")
            return self.vertex_arrays[name]
        
        try:
            # 頂点バッファ作成
            vertex_buffer = self.ctx.buffer(vertices.astype(np.float32).tobytes())
            self.buffers[f"{name}_vertices"] = vertex_buffer
            
            # インデックスバッファ（必要な場合）
            index_buffer = None
            if indices is not None:
                index_buffer = self.ctx.buffer(indices.astype(np.uint32).tobytes())
                self.buffers[f"{name}_indices"] = index_buffer
            
            # 頂点配列作成
            vertex_array = self.ctx.vertex_array(
                program, 
                [(vertex_buffer, '3f 2f', 'position', 'texcoord')], 
                index_buffer
            )
            self.vertex_arrays[name] = vertex_array
            
            self.logger.info(f"頂点配列 '{name}' 作成完了")
            return vertex_array
            
        except Exception as e:
            self.logger.error(f"頂点配列 '{name}' 作成失敗: {e}")
            raise
    
    def get_memory_info(self) -> Dict[str, int]:
        """メモリ使用量情報取得"""
        return self.memory_usage.copy()
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """GPU情報取得"""
        return self.gpu_info.copy()
    
    def cleanup_resource(self, resource_type: str, name: str):
        """リソース解放"""
        try:
            if resource_type == 'texture' and name in self.textures:
                self.textures[name].release()
                del self.textures[name]
            elif resource_type == 'vertex_array' and name in self.vertex_arrays:
                self.vertex_arrays[name].release()
                del self.vertex_arrays[name]
            elif resource_type == 'buffer' and name in self.buffers:
                self.buffers[name].release()
                del self.buffers[name]
            
            self.logger.info(f"{resource_type} '{name}' を解放しました")
            
        except Exception as e:
            self.logger.error(f"リソース解放エラー ({resource_type} '{name}'): {e}")
    
    def cleanup_all(self):
        """全リソース解放"""
        for name in list(self.textures.keys()):
            self.cleanup_resource('texture', name)
        
        for name in list(self.vertex_arrays.keys()):
            self.cleanup_resource('vertex_array', name)
        
        for name in list(self.buffers.keys()):
            self.cleanup_resource('buffer', name)
        
        self.logger.info("全リソース解放完了")
