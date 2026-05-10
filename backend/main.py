from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional, Any
import pandas as pd
import json

from services.data_service import DataService
from services.prediction_service import PredictionService, KRONOS_MODELS
from services.storage_service import StorageService

app = FastAPI(title="Kronos API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
data_service = DataService()
prediction_service = PredictionService()
storage_service = StorageService()

@app.get("/api/models")
async def get_models():
    return KRONOS_MODELS

@app.get("/api/data")
async def get_data(symbol: str, period: str = "1y", interval: str = "1d"):
    df, error = data_service.load_data(symbol, period, interval)
    if error:
        raise HTTPException(status_code=400, detail=error)
    
    # Return as JSON records for the frontend
    return df.to_dict(orient="records")

from pydantic import BaseModel

class PredictRequest(BaseModel):
    symbol: str
    period: str
    interval: str
    models: List[str]
    pred_len: int = 30
    lookback_limit: int = 256

class BacktestRequest(BaseModel):
    symbol: str
    period: str
    interval: str
    model_key: str
    lookback: int = 256
    pred_len: int = 30

class BacktestAllRequest(BaseModel):
    symbol: str
    period: str
    interval: str
    models: List[str]
    backtest_pred_len: int = 30
    backtest_lookback: int = 256

@app.post("/api/predict")
async def predict(request: PredictRequest):
    df, error = data_service.load_data(request.symbol, request.period, request.interval)
    if error:
        raise HTTPException(status_code=400, detail=error)
    
    results = prediction_service.predict_parallel(
        request.models, df, request.pred_len, request.lookback_limit
    )
    
    # Prepare results for JSON serialization
    serialized_results = {}
    for model_key, res in results.items():
        if "error" in res:
            serialized_results[model_key] = res
        else:
            # Convert DataFrame to JSON serializable format
            res_copy = res.copy()
            res_copy["pred_df"] = res["pred_df"].reset_index().to_dict(orient="records")
            res_copy["y_timestamp"] = [t.isoformat() for t in res["y_timestamp"]]
            serialized_results[model_key] = res_copy
            
    return serialized_results

@app.post("/api/backtest")
async def backtest(request: BacktestRequest):
    df, error = data_service.load_data(request.symbol, request.period, request.interval)
    if error:
        raise HTTPException(status_code=400, detail=error)
    
    try:
        res = prediction_service.run_backtest(
            request.model_key, df, request.lookback, request.pred_len
        )
        # Serialize DataFrames
        res["pred_df"] = res["pred_df"].reset_index().to_dict(orient="records")
        res["actual_df"] = res["actual_df"].reset_index().to_dict(orient="records")
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _serialize_backtest_result(res):
    """Helper to serialize a single backtest result dict."""
    if "error" in res:
        return res
    serialized = res.copy()
    if isinstance(res.get("pred_df"), pd.DataFrame):
        serialized["pred_df"] = res["pred_df"].reset_index().to_dict(orient="records")
    if isinstance(res.get("actual_df"), pd.DataFrame):
        serialized["actual_df"] = res["actual_df"].reset_index().to_dict(orient="records")
    return serialized

@app.post("/api/backtest-all")
async def backtest_all(request: BacktestAllRequest):
    df, error = data_service.load_data(request.symbol, request.period, request.interval)
    if error:
        raise HTTPException(status_code=400, detail=error)
    
    if len(df) < request.backtest_pred_len + 50:
        raise HTTPException(
            status_code=400, 
            detail=f"Not enough data for backtest. Need at least {request.backtest_pred_len + 50} rows, but only have {len(df)}."
        )
    
    try:
        results = prediction_service.run_backtest_all(
            request.models, df, request.backtest_lookback, request.backtest_pred_len
        )
        # Serialize all results
        serialized = {}
        for model_key, res in results.items():
            serialized[model_key] = _serialize_backtest_result(res)
        return serialized
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analyses")
async def list_analyses(symbol: Optional[str] = None, period: Optional[str] = None):
    return storage_service.list_analyses(symbol, period)

@app.get("/api/analyses/{key}")
async def get_analysis(key: str):
    analysis = storage_service.get_analysis(key)
    if not analysis or not analysis["metadata"]:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Serialize dataframes in predictions (handles Streamlit-saved DataFrame objects)
    if analysis["predictions"]:
        for model_key in list(analysis["predictions"].keys()):
            pred = analysis["predictions"][model_key]
            if "pred_df" in pred:
                if isinstance(pred["pred_df"], pd.DataFrame):
                    pdf = pred["pred_df"].copy()
                    if pdf.index.name != "datetime":
                        pdf.index.name = "datetime"
                    pred["pred_df"] = pdf.reset_index().to_dict(orient="records")
                # Ensure datetime strings are present (for dict/list format too)

    # Serialize backtest dataframes
    if analysis["backtest_results"]:
        for model_key in list(analysis["backtest_results"].keys()):
            bt = analysis["backtest_results"][model_key]
            if "error" in bt:
                continue
            # Handle pred_df
            if "pred_df" in bt and isinstance(bt["pred_df"], pd.DataFrame):
                pdf = bt["pred_df"].copy()
                if pdf.index.name != "datetime":
                    pdf.index.name = "datetime"
                bt["pred_df"] = pdf.reset_index().to_dict(orient="records")
            # Handle actual_df (API-saved) or test_df (Streamlit-saved)
            if "test_df" in bt and "actual_df" not in bt:
                bt["actual_df"] = bt.pop("test_df")
            if "actual_df" in bt and isinstance(bt["actual_df"], pd.DataFrame):
                adf = bt["actual_df"].copy()
                if adf.index.name != "datetime":
                    adf.index.name = "datetime"
                bt["actual_df"] = adf.reset_index().to_dict(orient="records")
                
    return analysis

@app.post("/api/analyses")
async def save_analysis(payload: Dict[str, Any] = Body(...)):
    try:
        key = storage_service.save_analysis(
            symbol=payload["symbol"],
            period=payload["period"],
            interval=payload["interval"],
            pred_config=payload["pred_config"],
            predictions=payload["predictions"],
            backtest_config=payload.get("backtest_config"),
            backtest_results=payload.get("backtest_results")
        )
        return {"key": key, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/analyses/{key}")
async def delete_analysis(key: str):
    success = storage_service.delete_analysis(key)
    if not success:
        raise HTTPException(status_code=404, detail="Analysis not found or could not be deleted")
    return {"status": "success"}

@app.get("/api/stats")
async def get_stats():
    return storage_service.get_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
