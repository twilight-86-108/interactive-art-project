import json
import os

class ConfigLoader:
    """設定ファイル読み込みクラス"""
    
    def __init__(self, config_path):
        self.config_path = config_path
        self._config = None
    
    def load(self):
        """設定ファイルを読み込む"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = json.load(f)
        
        return self._config
    
    def reload(self):
        """設定ファイルを再読み込み"""
        return self.load()
    
    def get(self, key_path, default=None):
        """ネストしたキーを取得（例: "camera.device_id"）"""
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value