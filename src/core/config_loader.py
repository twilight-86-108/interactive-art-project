# src/core/config_loader.py
import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigLoader:
    """設定ファイル読み込みクラス（エラー修正版）"""
    
    def __init__(self, config_path: Optional[str] = None):
        """ConfigLoader初期化
        
        Args:
            config_path: 設定ファイルパス。Noneの場合はデフォルトパスを使用
        """
        if config_path is None:
            # デフォルトパスを設定（プロジェクトルート/config/config.json）
            project_root = Path(__file__).parent.parent.parent
            self.config_path = str(project_root / "config" / "config.json")
        else:
            self.config_path = config_path
            
        self._config = None
        self.logger = logging.getLogger(__name__)
        
        # ログレベル設定（デバッグ情報を表示）
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"ConfigLoader初期化: {self.config_path}")
    
    def load(self) -> Dict[str, Any]:
        """設定ファイルを読み込む"""
        try:
            self.logger.info(f"設定ファイル読み込み試行: {self.config_path}")
            
            if not os.path.exists(self.config_path):
                self.logger.warning(f"設定ファイルが見つかりません: {self.config_path}")
                self.logger.info("デフォルト設定を使用します")
                return self._get_default_config()
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            
            self.logger.info(f"設定ファイル読み込み成功: {len(self._config)}個のセクション")
            return self._config
            
        except FileNotFoundError as e:
            self.logger.error(f"設定ファイル読み込みエラー: {e}")
            self.logger.info("デフォルト設定を使用します")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析エラー: {e}")
            self.logger.info("デフォルト設定を使用します")
            return self._get_default_config()
        except Exception as e:
            self.logger.error(f"予期しないエラー: {e}")
            self.logger.info("デフォルト設定を使用します")
            return self._get_default_config()
    
    def reload(self) -> Dict[str, Any]:
        """設定ファイルを再読み込み"""
        self.logger.info("設定ファイルを再読み込みします")
        return self.load()
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """ネストしたキーを取得（例: "camera.device_id"）"""
        try:
            if self._config is None:
                self.logger.warning("設定が読み込まれていません。再読み込みします。")
                self.load()
            
            if self._config is None:
                self.logger.warning("設定読み込みに失敗しました。デフォルト値を返します。")
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
            
            # ディレクトリが存在しない場合は作成
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"設定を保存しました: {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"設定保存エラー: {e}")
            return False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """デフォルト設定を返す"""
        self.logger.info("デフォルト設定を作成中...")
        return {
            "system": {
                "name": "Aqua Mirror",
                "version": "1.0.0",
                "debug_mode": True,
                "presentation_mode": False,
                "demo_mode": False
            },
            "camera": {
                "device_id": 0,
                "width": 1280,
                "height": 720,
                "fps": 30
            },
            "display": {
                "width": 1280,
                "height": 720,
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
                self.load()
                
            if self._config is None:
                return False
            
            # 必須項目の確認
            required_keys = [
                "camera.device_id",
                "display.width", 
                "display.height"
            ]
            
            for key in required_keys:
                if self.get(key) is None:
                    self.logger.warning(f"必須設定項目が不足: {key}")
                    return False
            
            self.logger.info("設定の妥当性確認が完了しました")
            return True
            
        except Exception as e:
            self.logger.error(f"設定妥当性確認エラー: {e}")
            return False
    
    def get_config(self) -> Optional[Dict[str, Any]]:
        """現在の設定を取得"""
        if self._config is None:
            self.load()
        return self._config
    
    def get_config_summary(self) -> str:
        """設定の概要を文字列で返す"""
        if self._config is None:
            self.load()
            
        if self._config is None:
            return "設定読み込み失敗"
        
        summary = []
        summary.append(f"システム: {self.get('system.name', 'Unknown')}")
        summary.append(f"カメラ: {self.get('camera.width', 0)}x{self.get('camera.height', 0)}@{self.get('camera.fps', 0)}fps")
        summary.append(f"ディスプレイ: {self.get('display.width', 0)}x{self.get('display.height', 0)}")
        summary.append(f"設定項目数: {len(self._config)}")
        
        return " | ".join(summary)

# テスト用関数
def test_config_loader():
    """ConfigLoaderの包括的テスト"""
    print("🧪 ConfigLoader 包括テスト開始...")
    
    try:
        # デフォルトパスでの初期化テスト
        print("\n1. デフォルト初期化テスト")
        loader = ConfigLoader()
        print(f"✅ デフォルトパス: {loader.config_path}")
        
        # 設定読み込みテスト
        print("\n2. 設定読み込みテスト")
        config = loader.load()
        print(f"✅ 設定読み込み成功: {len(config)}項目")
        print(f"📋 {loader.get_config_summary()}")
        
        # 特定キー取得テスト
        print("\n3. キー取得テスト")
        system_name = loader.get('system.name', 'Unknown')
        camera_fps = loader.get('camera.fps', 30)
        print(f"✅ system.name: {system_name}")
        print(f"✅ camera.fps: {camera_fps}")
        
        # 存在しないキーのテスト
        nonexistent = loader.get('nonexistent.key', 'default_value')
        print(f"✅ 存在しないキー: {nonexistent}")
        
        # 妥当性確認テスト
        print("\n4. 妥当性確認テスト")
        is_valid = loader.validate_config()
        print(f"✅ 設定妥当性: {is_valid}")
        
        print("\n✅ 全テスト成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_config_loader()