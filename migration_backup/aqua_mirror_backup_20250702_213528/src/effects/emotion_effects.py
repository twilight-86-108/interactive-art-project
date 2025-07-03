import logging
import numpy as np
import moderngl
import time
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)

class Emotion(Enum):
    HAPPY = 0
    SAD = 1
    ANGRY = 2
    SURPRISED = 3
    NEUTRAL = 4

class EmotionEffectManager:
    def __init__(self, gl_context: moderngl.Context, config_or_texture_manager: Any):
        """
        感情エフェクトマネージャー初期化
        
        Args:
            gl_context: ModernGLコンテキスト
            config_or_texture_manager: 設定オブジェクトまたはテクスチャマネージャー
        """
        self.ctx = gl_context
        
        # 引数の型に基づいて処理を分岐
        if hasattr(config_or_texture_manager, 'get'):
            # 設定オブジェクトとして扱う
            self.config = config_or_texture_manager
            self.texture_manager = None
        else:
            # テクスチャマネージャーとして扱う
            self.texture_manager = config_or_texture_manager
            self.config = None
        
        # 感情状態管理
        self.emotion_intensities = np.array([0.0, 0.0, 0.0, 0.0, 1.0])  # デフォルト: ニュートラル
        self.blend_mode = 0  # 0: smooth, 1: artistic
        self.last_update_time = time.time()
        
        # シェーダープログラム
        self.emotion_shader = None
        
        # テクスチャ
        self.emotion_texture = None
        self.features_texture = None
        
        try:
            self._initialize_textures()
            self._load_emotion_shader()
            logger.info("✅ 感情エフェクトマネージャー初期化完了")
        except Exception as e:
            logger.error(f"❌ 感情エフェクトマネージャー初期化失敗: {e}")
            raise
    
    def _initialize_textures(self):
        """テクスチャ初期化"""
        try:
            # テクスチャマネージャーを使用する場合
            if self.texture_manager:
                # テクスチャマネージャーから取得を試行
                try:
                    self.emotion_texture = self.texture_manager.get_texture('emotion')
                    self.features_texture = self.texture_manager.get_texture('features')
                except (AttributeError, KeyError):
                    # フォールバック: 自分で作成
                    self._create_default_textures()
            else:
                # 自分でテクスチャを作成
                self._create_default_textures()
            
            logger.info("✅ 感情テクスチャ初期化完了")
        except Exception as e:
            logger.error(f"❌ テクスチャ初期化エラー: {e}")
            # フォールバックテクスチャを作成
            self._create_default_textures()
    
    def _create_default_textures(self):
        """デフォルトテクスチャ作成"""
        # 1x1の基本テクスチャを作成
        self.emotion_texture = self.ctx.texture((1, 1), 4)
        self.features_texture = self.ctx.texture((1, 1), 4)
        
        # 初期データを設定
        initial_data = bytes([128, 128, 128, 255])  # RGBA (0.5, 0.5, 0.5, 1.0) を8bit値で
        self.emotion_texture.write(initial_data)
        self.features_texture.write(initial_data)
    
    def _load_emotion_shader(self):
        """感情シェーダー読み込み"""
        try:
            # 頂点シェーダー
            vertex_source = """
            #version 410 core
            
            in vec2 in_position;
            in vec2 in_texcoord;
            
            out vec2 v_texcoord;
            
            void main() {
                gl_Position = vec4(in_position, 0.0, 1.0);
                v_texcoord = in_texcoord;
            }
            """
            
            # フラグメントシェーダー
            fragment_source = """
            #version 410 core

            in vec2 v_texcoord;
            out vec4 fragColor;

            uniform sampler2D u_emotion_texture;
            uniform sampler2D u_features_texture;
            uniform float u_emotion_intensity[5];
            uniform float u_time;
            uniform int u_blend_mode;

            const vec3 EMOTION_COLORS[5] = vec3[](
                vec3(1.0, 0.85, 0.3),   // Happy
                vec3(0.3, 0.5, 0.8),    // Sad
                vec3(0.9, 0.2, 0.3),    // Angry
                vec3(1.0, 0.4, 0.8),    // Surprised
                vec3(0.5, 0.7, 0.6)     // Neutral
            );

            vec3 blend_emotions_smooth() {
                vec3 final_color = vec3(0.0);
                float total_intensity = 0.0;
                
                for(int i = 0; i < 5; i++) {
                    final_color += EMOTION_COLORS[i] * u_emotion_intensity[i];
                    total_intensity += u_emotion_intensity[i];
                }
                
                if(total_intensity > 0.0) {
                    final_color /= total_intensity;
                } else {
                    final_color = EMOTION_COLORS[4];
                }
                
                return final_color;
            }

            vec3 blend_emotions_artistic() {
                vec3 final_color = vec3(0.0);
                
                for(int i = 0; i < 5; i++) {
                    float weight = pow(u_emotion_intensity[i], 1.5);
                    final_color += EMOTION_COLORS[i] * weight;
                }
                
                float time_modulation = sin(u_time * 2.0) * 0.1 + 0.9;
                final_color *= time_modulation;
                
                if (length(final_color) > 0.0) {
                    return normalize(final_color) * length(final_color);
                } else {
                    return EMOTION_COLORS[4];
                }
            }

            void main() {
                vec3 emotion_color;
                
                if(u_blend_mode == 0) {
                    emotion_color = blend_emotions_smooth();
                } else {
                    emotion_color = blend_emotions_artistic();
                }
                
                vec4 features = texture(u_features_texture, v_texcoord);
                float intensity_modifier = max(features.w, 0.1);
                
                emotion_color *= intensity_modifier;
                
                fragColor = vec4(emotion_color, 1.0);
            }
            """
            
            self.emotion_shader = self.ctx.program(
                vertex_shader=vertex_source,
                fragment_shader=fragment_source
            )
            
            logger.info("✅ 感情シェーダー読み込み完了")
        except Exception as e:
            logger.error(f"❌ シェーダー読み込みエラー: {e}")
            # シェーダーが作成できない場合はNoneのままにして、
            # apply_shader_parametersでチェックする
            self.emotion_shader = None
    
    def update_emotion(self, emotion_or_type, intensity=None):
        """感情状態更新（柔軟な引数対応）"""
        try:
            # 引数パターンの解析
            if intensity is None:
                # 1つの引数の場合 - 辞書または感情データオブジェクト
                if isinstance(emotion_or_type, dict):
                    # 辞書形式: {'emotion': Emotion.HAPPY, 'intensity': 0.8}
                    emotion = emotion_or_type.get('emotion', Emotion.NEUTRAL)
                    intensity = emotion_or_type.get('intensity', 0.5)
                elif hasattr(emotion_or_type, 'emotion') and hasattr(emotion_or_type, 'intensity'):
                    # オブジェクト形式
                    emotion = emotion_or_type.emotion
                    intensity = emotion_or_type.intensity
                else:
                    # デフォルト
                    emotion = Emotion.NEUTRAL
                    intensity = 0.5
            else:
                # 2つの引数の場合
                emotion = emotion_or_type
            
            # Emotion型に変換
            if isinstance(emotion, int):
                if 0 <= emotion < 5:
                    emotion = Emotion(emotion)
                else:
                    emotion = Emotion.NEUTRAL
            elif not isinstance(emotion, Emotion):
                emotion = Emotion.NEUTRAL
            
            # 感情強度を配列で管理
            self.emotion_intensities.fill(0.0)
            self.emotion_intensities[emotion.value] = max(0.0, min(1.0, float(intensity)))
        
            logger.debug(f"感情更新: {emotion.name} = {intensity:.3f}")
        except Exception as e:
            logger.debug(f"感情更新エラー: {e}")
            # フォールバック
            self.emotion_intensities.fill(0.0)
            self.emotion_intensities[Emotion.NEUTRAL.value] = 0.5
    
    def update_emotion_blend(self, emotion_dict: Dict[Emotion, float]):
        """複数感情のブレンド更新"""
        try:
            # 全ての感情をリセット
            self.emotion_intensities.fill(0.0)
            
            # 辞書から感情強度を設定
            for emotion, intensity in emotion_dict.items():
                if isinstance(emotion, Emotion):
                    self.emotion_intensities[emotion.value] = max(0.0, min(1.0, intensity))
            
            # 正規化（合計が1.0を超える場合）
            total = np.sum(self.emotion_intensities)
            if total > 1.0:
                self.emotion_intensities /= total
            
            logger.debug(f"感情ブレンド更新: {self.emotion_intensities}")
        except Exception as e:
            logger.error(f"❌ 感情ブレンド更新エラー: {e}")
    
    def apply_shader_parameters(self):
        """シェーダーパラメータ適用"""
        try:
            if self.emotion_shader is None:
                logger.debug("シェーダーが初期化されていません")
                return
            
            # 現在の時間を取得
            current_time = time.time()
            
            # uniform変数を設定（エラー回避版）
            try:
                # 配列形式の感情強度を設定
                for i in range(5):
                    uniform_name = f'u_emotion_intensity[{i}]'
                    if uniform_name in self.emotion_shader:
                        self.emotion_shader[uniform_name].value = float(self.emotion_intensities[i])
                
                # 時間パラメータ
                if 'u_time' in self.emotion_shader:
                    self.emotion_shader['u_time'].value = float(current_time)
                
                # ブレンドモード
                if 'u_blend_mode' in self.emotion_shader:
                    self.emotion_shader['u_blend_mode'].value = int(self.blend_mode)
                
                # テクスチャバインディング
                if 'u_emotion_texture' in self.emotion_shader and self.emotion_texture:
                    self.emotion_texture.use(0)
                    self.emotion_shader['u_emotion_texture'].value = 0
                
                if 'u_features_texture' in self.emotion_shader and self.features_texture:
                    self.features_texture.use(1)
                    self.emotion_shader['u_features_texture'].value = 1
                
            except KeyError as e:
                logger.debug(f"uniform変数が見つかりません: {e}")
            except Exception as e:
                logger.debug(f"uniform設定エラー: {e}")
                
        except Exception as e:
            logger.debug(f"シェーダーパラメータ適用エラー: {e}")
    
    # 旧APIとの互換性（エラー回避）
    def update_emotion_legacy(self, emotion_type: int, emotion_intensity: float):
        """旧APIとの互換性"""
        try:
            if 0 <= emotion_type < 5:
                emotion = Emotion(emotion_type)
                self.update_emotion(emotion, emotion_intensity)
            else:
                logger.warning(f"無効な感情タイプ: {emotion_type}")
                self.update_emotion(Emotion.NEUTRAL, 0.5)
        except Exception as e:
            logger.error(f"旧API感情更新エラー: {e}")
    
    def render(self, render_target=None):
        """感情エフェクト描画"""
        try:
            if self.emotion_shader is None:
                logger.debug("シェーダーが初期化されていません")
                return
            
            # シェーダーパラメータ適用
            self.apply_shader_parameters()
            
            # シェーダープログラム使用
            self.emotion_shader.use()
            
            logger.debug("感情エフェクト描画完了")
            
        except Exception as e:
            logger.debug(f"感情エフェクト描画エラー: {e}")
    
    def set_blend_mode(self, mode: int):
        """ブレンドモード設定"""
        self.blend_mode = max(0, min(1, mode))
        logger.debug(f"ブレンドモード設定: {self.blend_mode}")
    
    def update_animation(self, delta_time=0.016):
        """アニメーション更新"""
        try:
            current_time = time.time()
            self.last_update_time = current_time
            logger.debug(f"アニメーション更新: dt={delta_time:.3f}")
        except Exception as e:
            logger.debug(f"アニメーション更新エラー: {e}")

    def update(self, delta_time=0.016):
        """汎用更新メソッド"""
        try:
            self.update_animation(delta_time)
        except Exception as e:
            logger.debug(f"更新エラー: {e}")

    def process_frame(self, frame_data=None):
        """フレーム処理"""
        try:
            pass
        except Exception as e:
            logger.debug(f"フレーム処理エラー: {e}")

    def get_current_effect(self):
        """現在のエフェクト取得"""
        try:
            return {
                'emotion_intensities': self.emotion_intensities.copy(),
                'blend_mode': self.blend_mode,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.debug(f"エフェクト取得エラー: {e}")
            return None

    def apply_to_shader(self, shader_program=None):
        """シェーダーにエフェクトを適用"""
        try:
            if shader_program is None:
                self.apply_shader_parameters()
            else:
                self._apply_to_external_shader(shader_program)
            logger.debug("シェーダーエフェクト適用完了")
        except Exception as e:
            logger.debug(f"シェーダーエフェクト適用エラー: {e}")

    def _apply_to_external_shader(self, shader_program):
        """外部シェーダーにパラメータ適用"""
        try:
            current_time = time.time()
            for i in range(5):
                try:
                    uniform_name = f'u_emotion_intensity[{i}]'
                    if uniform_name in shader_program:
                        shader_program[uniform_name].value = float(self.emotion_intensities[i])
                except:
                    pass
            try:
                if 'u_time' in shader_program:
                    shader_program['u_time'].value = float(current_time)
            except:
                pass
        except Exception as e:
            logger.debug(f"外部シェーダー適用エラー: {e}")

    def get_shader_uniforms(self):
        """シェーダーuniform値取得"""
        try:
            return {
                'u_emotion_intensity': self.emotion_intensities.tolist(),
                'u_time': time.time(),
                'u_blend_mode': self.blend_mode
            }
        except Exception as e:
            logger.debug(f"uniform値取得エラー: {e}")
            return {}

    def bind_textures(self, emotion_slot=0, features_slot=1):
        """テクスチャバインディング"""
        try:
            if self.emotion_texture:
                self.emotion_texture.use(emotion_slot)
            if self.features_texture:
                self.features_texture.use(features_slot)
        except Exception as e:
            logger.debug(f"テクスチャバインディングエラー: {e}")

    def activate(self):
        """エフェクト有効化"""
        try:
            if self.emotion_shader:
                self.emotion_shader.use()
                self.apply_shader_parameters()
                self.bind_textures()
        except Exception as e:
            logger.debug(f"エフェクト有効化エラー: {e}")

    def cleanup(self):
        """リソース解放"""
        try:
            if self.emotion_texture:
                self.emotion_texture.release()
                self.emotion_texture = None
            
            if self.features_texture:
                self.features_texture.release()
                self.features_texture = None
            
            if self.emotion_shader:
                self.emotion_shader.release()
                self.emotion_shader = None
            
            logger.info("✅ 感情エフェクトマネージャーリソース解放完了")
        except Exception as e:
            logger.error(f"❌ リソース解放エラー: {e}")