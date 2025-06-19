# src/core/camera_troubleshoot.py
class CameraTroubleshooter:
    """カメラ問題診断・修復"""
    
    @staticmethod
    def diagnose_camera_issues():
        """カメラ問題診断"""
        issues = []
        
        # 1. デバイス存在確認
        import subprocess
        try:
            result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                                   capture_output=True, text=True)
            if 'C922' not in result.stdout:
                issues.append("C922N デバイスが認識されていません")
        except:
            issues.append("v4l2-utils がインストールされていません")
        
        # 2. 権限確認
        import os
        if not os.access('/dev/video0', os.R_OK | os.W_OK):
            issues.append("カメラデバイスへの権限がありません")
        
        # 3. 他プロセスによる使用確認
        try:
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if any('video' in str(cmd).lower() for cmd in proc.info['cmdline'] or []):
                    if proc.info['pid'] != os.getpid():
                        issues.append(f"他のプロセス {proc.info['name']} がカメラを使用中")
        except:
            pass
        
        return issues
    
    @staticmethod
    def fix_camera_issues():
        """カメラ問題自動修復"""
        fixes_applied = []
        
        # 1. 権限修復
        try:
            os.system('sudo usermod -aG video $USER')
            fixes_applied.append("ユーザーをvideoグループに追加")
        except:
            pass
        
        # 2. カメラ設定リセット
        try:
            os.system('v4l2-ctl --device=/dev/video0 --set-ctrl=focus_auto=0')
            os.system('v4l2-ctl --device=/dev/video0 --set-ctrl=exposure_auto=1')
            fixes_applied.append("カメラ設定をリセット")
        except:
            pass
        
        return fixes_applied