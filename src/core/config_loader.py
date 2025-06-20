# src/core/config_loader.py
import json
import os
import logging
from typing import Dict, Any, Optional

class ConfigLoader:
    """設定ファイル読み込みクラス（エラー修正版）"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._config = None
        self.logger = logging.getLogger(__name__)
    
    def load(self) -> Dict[str, Any]:
        """設定ファイルを読み込む"""
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"設定ファイルが見つかりません: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            
            self.logger.info(f"設定ファイルを読み込みました: {self.config_path}")
            return self._config
            
        except FileNotFoundError as e:
            self.logger.error(f"設定ファイル読み込みエラー: {e}")
            # デフォルト設定を返す
            return self._get_default_config()
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析エラー: {e}")
            return self._get_default_config()
        except Exception as e:
            self.logger.error(f"予期しないエラー: {e}")
            return self._get_default_config()
    
    def reload(self) -> Dict[str, Any]:
        """設定ファイルを再読み込み"""
        self.logger.info("設定ファイルを再読み込みします")
        return self.load()
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """ネストしたキーを取得（例: "camera.device_id"）"""
        try:
            if self._config is None:
                self.logger.warning("設定が読み込まれていません。デフォルト値を返します。")
                return default
            
            keys = key_path.split('.')
            value = self._config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    self.logger.debug(f"キー '{key_path}' が見つかりません。デフォルト値を返します。")
                    return default
            
            return value
            
        except Exception as e:
            self.logger.error(f"設定値取得エラー: {e}")
            return default
    
    def set(self, key_path: str, value: Any) -> bool:
        """ネストしたキーに値を設定"""
        try:
            if self._config is None:
                self._config = {}
            
            keys = key_path.split('.')
            current = self._config
            
            # 最後のキー以外は辞書を作成
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                elif not isinstance(current[key], dict):
                    current[key] = {}
                current = current[key]
            
            # 最後のキーに値を設定
            current[keys[-1]] = value
            self.logger.debug(f"設定値を更新しました: {key_path} = {value}")
            return True
            
        except Exception as e:
            self.logger.error(f"設定値設定エラー: {e}")
            return False
    
    def save(self, output_path: Optional[str] = None) -> bool:
        """設定をファイルに保存"""
        try:
            if self._config is None:
                self.logger.warning("保存する設定がありません")
                return False
            
            save_path = output_path or self.config_path
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"設定を保存しました: {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"設定保存エラー: {e}")
            return False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す"""
        return {
            "camera": {
                "device_id": 0,
                "width": 1920,
                "height": 1080,
                "fps": 30
            },
            "display": {
                "width": 1920,
                "height": 1080,
                "fullscreen": False
            },
            "detection": {
                "face_detection_confidence": 0.7,
                "hand_detection_confidence": 0.7,
                "max_num_faces": 1,
                "max_num_hands": 2,
                "min_tracking_confidence": 0.5
            },
            "interaction": {
                "approach_threshold_z": 0.8,
                "interaction_regions": {
                    "water_surface": {
                        "x": 400,
                        "y": 300,
                        "width": 1120,
                        "height": 480
                    }
                }
            },
            "assets": {
                "background_image": "assets/images/underwater_scene.jpg",
                "fish_images": ["assets/images/fish1.png", "assets/images/fish2.png"],
                "sounds": {
                    "splash": "assets/sounds/splash.wav",
                    "bubble": "assets/sounds/bubble.wav"
                }
            },
            "visual_effects": {
                "quality_level": "medium",
                "particles": {
                    "max_count": 500,
                    "gpu_acceleration": True
                },
                "water_effects": {
                    "wave_simulation": True,
                    "caustics": False
                }
            },
            "audio": {
                "enabled": True,
                "master_volume": 0.7,
                "sample_rate": 22050,
                "channels": 2
            },
            "performance": {
                "target_fps": 30,
                "adaptive_quality": True,
                "memory_limit_mb": 16384
            }
        }
    
    def validate_config(self) -> bool:
        """設定の妥当性確認"""
        try:
            if self._config is None:
                return False
            
            # 必須項目の確認
            required_keys = [
                "camera.device_id",
                "display.width",
                "display.height",
                "detection.face_detection_confidence"
            ]
            
            for key in required_keys:
                if self.get(key) is None:
                    self.logger.warning(f"必須設定項目が不足: {key}")
                    return False
            
            # 値の範囲確認
            if not (0 <= self.get("detection.face_detection_confidence", 0) <= 1):
                self.logger.warning("face_detection_confidence は 0-1 の範囲で設定してください")
                return False
            
            if not (0 <= self.get("detection.hand_detection_confidence", 0) <= 1):
                self.logger.warning("hand_detection_confidence は 0-1 の範囲で設定してください")
                return False
            
            self.logger.info("設定の妥当性確認が完了しました")
            return True
            
        except Exception as e:
            self.logger.error(f"設定妥当性確認エラー: {e}")
            return False
    
    def get_config(self) -> Optional[Dict[str, Any]]:
        """現在の設定を取得"""
        return self._config
    
    def merge_config(self, additional_config: Dict[str, Any]) -> bool:
        """追加設定をマージ"""
        try:
            if self._config is None:
                self._config = {}
            
            self._deep_merge(self._config, additional_config)
            self.logger.info("設定をマージしました")
            return True
            
        except Exception as e:
            self.logger.error(f"設定マージエラー: {e}")
            return False
    
    def _deep_merge(self, base_dict: Dict[str, Any], merge_dict: Dict[str, Any]):
        """辞書の深いマージ"""
        for key, value in merge_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value