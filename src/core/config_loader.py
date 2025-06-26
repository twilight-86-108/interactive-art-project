"""
設定ファイル読み込み・管理システム
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

class ConfigLoader:
    """
    JSON設定ファイル読み込み・管理クラス
    """
    
    def __init__(self, config_path: str = "config/config.json"):
        self.config_path = Path(config_path)
        self.config_data: Dict[str, Any] = {}
        self.logger = logging.getLogger("ConfigLoader")
        
        self.load_config()
    
    def load_config(self) -> bool:
        """設定ファイル読み込み"""
        try:
            if not self.config_path.exists():
                self.logger.error(f"設定ファイルが見つかりません: {self.config_path}")
                return False
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = json.load(f)
            
            self.logger.info(f"設定ファイル読み込み完了: {self.config_path}")
            return True
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析エラー: {e}")
            return False
        except Exception as e:
            self.logger.error(f"設定ファイル読み込みエラー: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        ドット記法で設定値取得
        例: get('app.window.width', 1920)
        """
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """設定値変更"""
        keys = key.split('.')
        config = self.config_data
        
        try:
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            config[keys[-1]] = value
            return True
        except Exception as e:
            self.logger.error(f"設定値変更エラー: {e}")
            return False
    
    def save_config(self) -> bool:
        """設定ファイル保存"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info("設定ファイル保存完了")
            return True
        except Exception as e:
            self.logger.error(f"設定ファイル保存エラー: {e}")
            return False
