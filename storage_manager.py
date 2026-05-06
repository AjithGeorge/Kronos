"""
Persistent storage manager for Kronos predictions and backtest analyses.
Stores data to user's home directory for permanent retention across sessions.
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd


class AnalysisStorageManager:
    """Manages persistent storage of predictions and backtest results."""

    def __init__(self):
        """Initialize storage directory in user's home folder."""
        self.storage_root = Path.home() / ".kronos_analyses"
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_root / "analyses_index.json"
        self._load_index()

    def _load_index(self) -> None:
        """Load or create the analyses index."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    self.index = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load index: {e}. Creating new index.")
                self.index = {}
        else:
            self.index = {}

    def _save_index(self) -> None:
        """Save the analyses index."""
        try:
            with open(self.index_file, "w") as f:
                json.dump(self.index, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save index: {e}")

    def _generate_config_hash(self, config_dict: Dict) -> str:
        """Generate a hash of configuration parameters."""
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _generate_analysis_key(
        self,
        symbol: str,
        period: str,
        interval: str,
        pred_config: Dict,
        backtest_config: Optional[Dict] = None,
    ) -> str:
        """Generate unique key for analysis based on all parameters."""
        pred_hash = self._generate_config_hash(pred_config)
        bt_hash = self._generate_config_hash(backtest_config or {})

        # Clean symbol for folder name
        clean_symbol = symbol.replace(".", "_")

        # Format: SYMBOL_PERIOD_INTERVAL_PREDHASH_BTHASH
        key = f"{clean_symbol}_{period}_{interval}_p{pred_hash}"
        if backtest_config:
            key += f"_b{bt_hash}"

        return key

    def save_analysis(
        self,
        symbol: str,
        period: str,
        interval: str,
        pred_config: Dict,
        predictions: Dict,
        backtest_config: Optional[Dict] = None,
        backtest_results: Optional[Dict] = None,
    ) -> str:
        """
        Save predictions and backtest results for future retrieval.

        Args:
            symbol: Stock symbol (e.g., 'RECLTD.NS')
            period: Historical period (e.g., '1y')
            interval: Data interval (e.g., '1d')
            pred_config: Prediction configuration (models, lookback, pred_len)
            predictions: Dictionary of predictions from all models
            backtest_config: Backtest configuration (optional)
            backtest_results: Backtest results (optional)

        Returns:
            Analysis key for later retrieval
        """
        # Generate unique key
        key = self._generate_analysis_key(
            symbol, period, interval, pred_config, backtest_config
        )

        # Create analysis directory
        analysis_dir = self.storage_root / key
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Save prediction data
        pred_data = {
            "all_predictions": predictions,
            "config": pred_config,
        }
        with open(analysis_dir / "predictions.pkl", "wb") as f:
            pickle.dump(pred_data, f)

        # Save backtest data if available
        if backtest_results:
            bt_data = {
                "backtest_results": backtest_results,
                "config": backtest_config,
            }
            with open(analysis_dir / "backtest.pkl", "wb") as f:
                pickle.dump(bt_data, f)

        # Create metadata
        metadata = {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "created_at": datetime.now().isoformat(),
            "pred_config": pred_config,
            "backtest_config": backtest_config,
            "has_predictions": True,
            "has_backtest": bool(backtest_results),
            "num_models": len(predictions),
            "model_names": [
                predictions[m].get("config", {}).get("name", m)
                for m in predictions.keys()
            ],
        }

        with open(analysis_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Update index
        self.index[key] = metadata
        self._save_index()

        return key

    def load_analysis(
        self, key: str
    ) -> Tuple[Dict, Dict, Optional[Dict], Optional[Dict]]:
        """
        Load a previously saved analysis.

        Args:
            key: Analysis key returned from save_analysis

        Returns:
            Tuple of (predictions, pred_config, backtest_results, backtest_config)
            Returns (None, None, None, None) if not found
        """
        analysis_dir = self.storage_root / key

        if not analysis_dir.exists():
            print(f"Analysis {key} not found.")
            return None, None, None, None

        try:
            # Load predictions
            with open(analysis_dir / "predictions.pkl", "rb") as f:
                pred_data = pickle.load(f)
            predictions = pred_data.get("all_predictions", {})
            pred_config = pred_data.get("config", {})

            # Load backtest if available
            backtest_results = None
            backtest_config = None
            backtest_file = analysis_dir / "backtest.pkl"
            if backtest_file.exists():
                with open(backtest_file, "rb") as f:
                    bt_data = pickle.load(f)
                backtest_results = bt_data.get("backtest_results", {})
                backtest_config = bt_data.get("config", {})

            return predictions, pred_config, backtest_results, backtest_config

        except Exception as e:
            print(f"Error loading analysis {key}: {e}")
            return None, None, None, None

    def get_analysis_metadata(self, key: str) -> Optional[Dict]:
        """Get metadata for a stored analysis."""
        analysis_dir = self.storage_root / key
        metadata_file = analysis_dir / "metadata.json"

        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading metadata for {key}: {e}")
        return None

    def list_analyses(
        self, symbol: Optional[str] = None, period: Optional[str] = None
    ) -> List[Dict]:
        """
        List all stored analyses, optionally filtered by symbol or period.

        Args:
            symbol: Filter by symbol (optional)
            period: Filter by period (optional)

        Returns:
            List of analysis metadata dictionaries
        """
        results = []

        for key, metadata in self.index.items():
            # Apply filters
            if symbol and metadata.get("symbol", "").upper() != symbol.upper():
                continue
            if period and metadata.get("period") != period:
                continue

            results.append({**metadata, "key": key})

        # Sort by creation date (newest first)
        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return results

    def delete_analysis(self, key: str) -> bool:
        """Delete a stored analysis."""
        analysis_dir = self.storage_root / key

        if not analysis_dir.exists():
            print(f"Analysis {key} not found.")
            return False

        try:
            # Remove all files in directory
            for file in analysis_dir.iterdir():
                if file.is_file():
                    file.unlink()
            # Remove directory
            analysis_dir.rmdir()
            # Update index
            if key in self.index:
                del self.index[key]
            self._save_index()
            return True
        except Exception as e:
            print(f"Error deleting analysis {key}: {e}")
            return False

    def get_storage_path(self) -> Path:
        """Get the storage directory path."""
        return self.storage_root

    def get_storage_size(self) -> Dict:
        """Get storage size statistics."""
        total_size = 0
        num_analyses = 0
        num_predictions = 0
        num_backtests = 0

        for key in self.index.keys():
            analysis_dir = self.storage_root / key
            if analysis_dir.exists():
                num_analyses += 1
                if (analysis_dir / "predictions.pkl").exists():
                    num_predictions += 1
                if (analysis_dir / "backtest.pkl").exists():
                    num_backtests += 1

                for file in analysis_dir.rglob("*"):
                    if file.is_file():
                        total_size += file.stat().st_size

        return {
            "total_size_mb": total_size / (1024 * 1024),
            "num_analyses": num_analyses,
            "num_predictions": num_predictions,
            "num_backtests": num_backtests,
        }

    def get_duplicate_analyses(
        self, symbol: str, period: str, interval: str
    ) -> List[Dict]:
        """
        Get all analyses for a specific symbol, period, and interval combination.
        Useful for showing which configs have been run for same data.
        """
        results = []

        for key, metadata in self.index.items():
            if (
                metadata.get("symbol", "").upper() == symbol.upper()
                and metadata.get("period") == period
                and metadata.get("interval") == interval
            ):
                results.append({**metadata, "key": key})

        results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return results

    def check_exists(
        self,
        symbol: str,
        period: str,
        interval: str,
        pred_config: Dict,
        backtest_config: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Check if an analysis with exact configuration already exists.

        Returns:
            Analysis key if found, None otherwise
        """
        key = self._generate_analysis_key(
            symbol, period, interval, pred_config, backtest_config
        )

        if key in self.index:
            return key

        return None

    def export_analysis(self, key: str, output_path: Path) -> bool:
        """
        Export an analysis to a ZIP file for backup or sharing.

        Args:
            key: Analysis key
            output_path: Path for output ZIP file

        Returns:
            True if successful, False otherwise
        """
        import shutil

        analysis_dir = self.storage_root / key

        if not analysis_dir.exists():
            print(f"Analysis {key} not found.")
            return False

        try:
            # Create ZIP archive
            shutil.make_archive(
                str(output_path.with_suffix("")), "zip", analysis_dir.parent, key
            )
            return True
        except Exception as e:
            print(f"Error exporting analysis {key}: {e}")
            return False

    def import_analysis(self, import_zip_path: Path) -> Optional[str]:
        """
        Import an analysis from a ZIP file.

        Args:
            import_zip_path: Path to ZIP file

        Returns:
            Analysis key if successful, None otherwise
        """
        import shutil
        import zipfile

        try:
            # Extract to temporary location first
            with zipfile.ZipFile(import_zip_path, "r") as zip_ref:
                # Get the top-level folder name
                names = zip_ref.namelist()
                top_folder = names[0].split("/")[0]

                # Extract
                extract_dir = self.storage_root / top_folder
                if extract_dir.exists():
                    shutil.rmtree(extract_dir)

                zip_ref.extractall(self.storage_root)

            # Load metadata to update index
            metadata_file = extract_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                self.index[top_folder] = metadata
                self._save_index()

            return top_folder

        except Exception as e:
            print(f"Error importing analysis: {e}")
            return None
