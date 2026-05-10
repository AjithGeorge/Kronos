import os
import sys
from typing import Dict, List, Optional, Tuple

# Add root to sys.path to import storage_manager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from storage_manager import AnalysisStorageManager
except ImportError:
    # If it fails, we might be running in a different context
    AnalysisStorageManager = None

class StorageService:
    def __init__(self):
        if AnalysisStorageManager:
            self.manager = AnalysisStorageManager()
        else:
            self.manager = None

    def list_analyses(self, symbol: Optional[str] = None, period: Optional[str] = None):
        if not self.manager:
            return []
        return self.manager.list_analyses(symbol, period)

    def get_analysis(self, key: str):
        if not self.manager:
            return None
        
        predictions, pred_config, backtest_results, backtest_config = self.manager.load_analysis(key)
        metadata = self.manager.get_analysis_metadata(key)
        
        return {
            "predictions": predictions,
            "pred_config": pred_config,
            "backtest_results": backtest_results,
            "backtest_config": backtest_config,
            "metadata": metadata
        }

    def save_analysis(self, symbol: str, period: str, interval: str, pred_config: Dict, 
                      predictions: Dict, backtest_config: Optional[Dict] = None, 
                      backtest_results: Optional[Dict] = None):
        if not self.manager:
            return None
        
        return self.manager.save_analysis(
            symbol=symbol,
            period=period,
            interval=interval,
            pred_config=pred_config,
            predictions=predictions,
            backtest_config=backtest_config,
            backtest_results=backtest_results
        )

    def delete_analysis(self, key: str):
        if not self.manager:
            return False
        return self.manager.delete_analysis(key)

    def get_stats(self):
        if not self.manager:
            return {}
        return self.manager.get_storage_size()
