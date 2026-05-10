import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  TrendingUp, Database, RefreshCw, Cpu, Save, Table as TableIcon
} from 'lucide-react';
import Sidebar from './components/Sidebar';
import MainChart from './components/MainChart';
import Metrics from './components/Metrics';
import AnalysisList from './components/AnalysisList';
import PredictionTable from './components/PredictionTable';
import BacktestSection from './components/BacktestSection';
import ModelComparison from './components/ModelComparison';

const API_BASE = 'http://localhost:8000/api';

function App() {
  const [activeSymbol, setActiveSymbol] = useState('RECLTD.NS');
  const [period, setPeriod] = useState('1y');
  const [interval, setInterval] = useState('1d');
  const [lookback, setLookback] = useState(256);
  const [predLen, setPredLen] = useState(30);

  const [activeTabs, setActiveTabs] = useState([]); 
  const [data, setData] = useState([]);
  const [predictions, setPredictions] = useState({});
  const [models, setModels] = useState({});
  const [selectedModels, setSelectedModels] = useState(['kronos-base']);
  const [loading, setLoading] = useState(false);
  const [activeModelTab, setActiveModelTab] = useState(null);

  // Backtest state - separate config matching Streamlit
  const [backtestResultsAll, setBacktestResultsAll] = useState(null);
  const [backtestPredLen, setBtPredLen] = useState(30);
  const [backtestLookback, setBtLookback] = useState(256);

  // Storage
  const [savedAnalyses, setSavedAnalyses] = useState([]);
  const [view, setView] = useState('dashboard');

  // Summary metrics (per-model)
  const [summaryMetrics, setSummaryMetrics] = useState({});

  // Raw data visibility
  const [showRawData, setShowRawData] = useState(false);

  useEffect(() => { fetchModels(); fetchAnalyses(); }, []);

  useEffect(() => {
    if (selectedModels.length > 0 && !activeModelTab) {
      setActiveModelTab(selectedModels[0]);
    }
  }, [selectedModels]);

  const fetchModels = async () => {
    try { const res = await axios.get(`${API_BASE}/models`); setModels(res.data); }
    catch (err) { console.error('Error fetching models:', err); }
  };

  const fetchAnalyses = async () => {
    try { const res = await axios.get(`${API_BASE}/analyses`); setSavedAnalyses(res.data); }
    catch (err) { console.error('Error fetching analyses:', err); }
  };

  const loadSymbolData = async (symbolOverride) => {
    const sym = symbolOverride || activeSymbol;
    if (!sym) return;
    setLoading(true);
    try {
      const res = await axios.get(`${API_BASE}/data`, { params: { symbol: sym, period, interval } });
      setData(res.data);
      setPredictions({});
      setBacktestResultsAll(null);
      setSummaryMetrics({});
      if (!activeTabs.includes(sym)) setActiveTabs(prev => [...prev, sym]);
    } catch (err) {
      console.error('Error loading data:', err);
      alert('Failed to load data for this symbol/config.');
    } finally { setLoading(false); }
  };

  const calculateSummaryMetrics = (preds) => {
    if (!data || data.length === 0 || !preds) return {};
    const results = {};
    const lastHist = data[data.length - 1];
    Object.keys(preds).forEach(modelKey => {
      const pred = preds[modelKey];
      if (pred.error) return;
      const predDf = pred.pred_df;
      const lastPred = predDf[predDf.length - 1];
      const futurePrice = lastPred.close;
      const lastPrice = lastHist.close;
      const changePct = ((futurePrice - lastPrice) / lastPrice) * 100;
      const minPred = Math.min(...predDf.map(d => d.low || d.close));
      const maxPred = Math.max(...predDf.map(d => d.high || d.close));
      const avgPredVol = predDf.reduce((sum, d) => sum + (d.volume || 0), 0) / predDf.length;
      const lastVol = lastHist.volume;
      results[modelKey] = {
        lastPrice, futurePrice, changePct, minPred, maxPred,
        predRange: maxPred - minPred, avgPredVol,
        volChangePct: ((avgPredVol - lastVol) / lastVol) * 100,
        trend: changePct > 0 ? 'Bullish' : changePct < 0 ? 'Bearish' : 'Neutral',
        highDelta: ((maxPred - lastPrice) / lastPrice) * 100,
        lowDelta: ((minPred - lastPrice) / lastPrice) * 100,
        period: predDf.length
      };
    });
    setSummaryMetrics(results);
  };

  const closeTab = (symbolToClose) => {
    setActiveTabs(prev => prev.filter(s => s !== symbolToClose));
    if (activeSymbol === symbolToClose) {
      setData([]); setPredictions({}); setBacktestResultsAll(null); setSummaryMetrics({});
    }
  };

  const runPredictions = async () => {
    if (data.length === 0) return;
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/predict`, {
        symbol: activeSymbol, period, interval,
        models: selectedModels, pred_len: predLen, lookback_limit: lookback
      });
      setPredictions(res.data);
      calculateSummaryMetrics(res.data);
      const firstValid = Object.keys(res.data).find(k => !res.data[k].error);
      if (firstValid) setActiveModelTab(firstValid);
    } catch (err) { console.error('Error running predictions:', err); }
    finally { setLoading(false); }
  };

  const runBacktestAll = async () => {
    if (data.length === 0 || selectedModels.length === 0) return;
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/backtest-all`, {
        symbol: activeSymbol, period, interval,
        models: selectedModels,
        backtest_pred_len: backtestPredLen,
        backtest_lookback: backtestLookback
      });
      setBacktestResultsAll(res.data);
    } catch (err) {
      console.error('Error running backtest:', err);
      alert(err.response?.data?.detail || 'Backtest failed');
    } finally { setLoading(false); }
  };

  const saveAnalysis = async () => {
    const hasPreds = Object.keys(predictions).length > 0;
    const hasBt = backtestResultsAll && Object.keys(backtestResultsAll).length > 0;
    if (!hasPreds) return;
    try {
      await axios.post(`${API_BASE}/analyses`, {
        symbol: activeSymbol, period, interval,
        pred_config: { models: selectedModels, pred_len: predLen, lookback_limit: lookback },
        predictions,
        backtest_results: hasBt ? backtestResultsAll : null,
        backtest_config: hasBt ? { lookback: backtestLookback, pred_len: backtestPredLen } : null
      });
      fetchAnalyses();
      alert('Analysis saved successfully!');
    } catch (err) { console.error('Error saving analysis:', err); }
  };

  const loadAnalysis = async (key) => {
    setLoading(true);
    try {
      const res = await axios.get(`${API_BASE}/analyses/${key}`);
      const { metadata, predictions: preds, backtest_results: bt } = res.data;
      setActiveSymbol(metadata.symbol);
      setPredictions(preds || {});
      calculateSummaryMetrics(preds);
      if (bt && Object.keys(bt).length > 0) setBacktestResultsAll(bt);
      else setBacktestResultsAll(null);
      await loadSymbolData(metadata.symbol);
      setView('dashboard');
    } catch (err) { console.error('Error loading analysis:', err); }
    finally { setLoading(false); }
  };

  const handleDeleteAnalysis = () => { fetchAnalyses(); };

  const hasPredictions = predictions && Object.keys(predictions).length > 0;
  const validModelKeys = hasPredictions ? Object.keys(predictions).filter(k => !predictions[k].error) : [];
  const rawDataRows = data.slice(-5);

  return (
    <div className="flex h-screen bg-[#0a0a0a] text-white overflow-hidden font-sans">
      <Sidebar activeSymbol={activeSymbol} onSymbolChange={setActiveSymbol} setView={setView}
        currentView={view} activeTabs={activeTabs} onCloseTab={closeTab} />

      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="h-16 glass flex items-center justify-between px-6 z-10 border-b border-white/5">
          <div className="flex items-center gap-3">
            <TrendingUp className="text-accent" size={24} />
            <h1 className="text-xl font-bold tracking-tight">Kronos <span className="text-accent">Stock Predictor</span></h1>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center bg-white/5 rounded-lg px-3 py-1.5 border border-white/10">
              <Cpu size={16} className="text-accent mr-2" />
              <span className="text-xs font-bold uppercase tracking-wider text-white/60">Device: CUDA</span>
            </div>
            <button onClick={saveAnalysis} disabled={!hasPredictions}
              className="flex items-center gap-2 bg-accent hover:bg-accent/90 disabled:opacity-50 text-white px-4 py-1.5 rounded-lg transition-all shadow-lg shadow-accent/20">
              <Save size={16} /><span className="text-sm font-bold">Save Analysis</span>
            </button>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto p-6 scrollbar-hide">
          {view === 'dashboard' ? (
            <div className="max-w-[1400px] mx-auto space-y-6 pb-20">
              {/* Configuration Bar */}
              <div className="glass rounded-2xl p-6 space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6">
                  <div className="space-y-2">
                    <label className="text-xs font-semibold text-white/40 uppercase tracking-wider">Stock Symbol</label>
                    <input type="text" value={activeSymbol} onChange={e => setActiveSymbol(e.target.value.toUpperCase())}
                      className="w-full bg-white/5 border border-white/10 rounded-xl py-2 px-4 text-sm focus:outline-none focus:border-accent/50 transition-all font-bold"
                      placeholder="e.g. RELIANCE.NS" />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-semibold text-white/40 uppercase tracking-wider">Historical Period</label>
                    <select value={period} onChange={e => setPeriod(e.target.value)}
                      className="w-full bg-[#0a0a0a] border border-white/10 rounded-xl py-2 px-4 text-sm focus:outline-none focus:border-accent/50 transition-all cursor-pointer">
                      {['1mo','3mo','6mo','1y','2y','5y'].map(p => <option key={p} value={p}>{p}</option>)}
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-semibold text-white/40 uppercase tracking-wider">Interval</label>
                    <select value={interval} onChange={e => setInterval(e.target.value)}
                      className="w-full bg-[#0a0a0a] border border-white/10 rounded-xl py-2 px-4 text-sm focus:outline-none focus:border-accent/50 transition-all cursor-pointer">
                      {['1d','1h','30m'].map(i => <option key={i} value={i}>{i}</option>)}
                    </select>
                  </div>
                  <div className="space-y-2 lg:col-span-1">
                    <label className="text-xs font-semibold text-white/40 uppercase tracking-wider">Prediction Length ({predLen}d)</label>
                    <input type="range" min="5" max="60" value={predLen} onChange={e => setPredLen(parseInt(e.target.value))} className="w-full accent-accent cursor-pointer" />
                  </div>
                  <div className="space-y-2 lg:col-span-1">
                    <label className="text-xs font-semibold text-white/40 uppercase tracking-wider">Lookback ({lookback})</label>
                    <input type="range" min="100" max="512" value={lookback} onChange={e => setLookback(parseInt(e.target.value))} className="w-full accent-accent cursor-pointer" />
                  </div>
                </div>

                <div className="flex flex-col md:flex-row items-center justify-between gap-6 pt-4 border-t border-white/5">
                  <div className="flex flex-wrap gap-4 items-center">
                    <div className="space-y-2">
                      <label className="text-[10px] font-bold text-white/20 uppercase tracking-widest block">Model Comparison</label>
                      <div className="flex gap-2">
                        {Object.keys(models).map(key => (
                          <button key={key}
                            onClick={() => setSelectedModels(prev => prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key])}
                            className={`px-3 py-1.5 rounded-lg text-xs font-bold border transition-all ${
                              selectedModels.includes(key) ? 'bg-accent border-accent text-white' : 'bg-white/5 border-white/10 text-white/60 hover:bg-white/10'}`}>
                            {models[key].name}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <button onClick={() => loadSymbolData()} disabled={loading || !activeSymbol}
                      className="h-11 bg-white/5 hover:bg-white/10 text-white font-bold px-6 rounded-xl border border-white/10 transition-all flex items-center gap-2">
                      <Database size={18} /> Load Data
                    </button>
                    <button onClick={runPredictions} disabled={loading || selectedModels.length === 0 || data.length === 0}
                      className="h-11 bg-accent text-white font-bold px-8 rounded-xl hover:bg-accent/90 disabled:opacity-50 transition-all flex items-center gap-2 shadow-lg shadow-accent/20">
                      {loading ? <RefreshCw className="animate-spin" size={18} /> : <TrendingUp size={18} />}
                      Run Prediction
                    </button>
                  </div>
                </div>
              </div>

              {/* Raw Data Table */}
              {data.length > 0 && (
                <div className="glass rounded-2xl overflow-hidden border border-white/5">
                  <button onClick={() => setShowRawData(!showRawData)}
                    className="w-full px-6 py-4 flex items-center justify-between bg-white/5 hover:bg-white/[0.07] transition-colors">
                    <div className="flex items-center gap-2">
                      <TableIcon size={16} className="text-accent" />
                      <span className="font-bold text-sm uppercase tracking-wider text-white/80">Raw Data</span>
                      <span className="text-[10px] text-white/30 ml-2">{data.length} rows loaded</span>
                    </div>
                    <span className="text-xs text-white/30">{showRawData ? '▲ Hide' : '▼ Show last 5 rows'}</span>
                  </button>
                  {showRawData && (
                    <div className="overflow-x-auto">
                      <table className="w-full text-left border-collapse">
                        <thead><tr className="bg-white/5 text-[10px] font-bold text-white/40 uppercase tracking-widest border-b border-white/5">
                          <th className="px-6 py-3">Date</th><th className="px-6 py-3">Open</th><th className="px-6 py-3">High</th>
                          <th className="px-6 py-3">Low</th><th className="px-6 py-3">Close</th><th className="px-6 py-3">Volume</th>
                        </tr></thead>
                        <tbody className="text-sm">{rawDataRows.map((r, i) => (
                          <tr key={i} className="border-b border-white/5 hover:bg-white/[0.02]">
                            <td className="px-6 py-2 text-white/60">{new Date(r.datetime).toLocaleDateString()}</td>
                            <td className="px-6 py-2 font-mono text-white/80">{r.open?.toFixed(2)}</td>
                            <td className="px-6 py-2 font-mono text-emerald-400">{r.high?.toFixed(2)}</td>
                            <td className="px-6 py-2 font-mono text-red-400">{r.low?.toFixed(2)}</td>
                            <td className="px-6 py-2 font-mono font-bold text-white">{r.close?.toFixed(2)}</td>
                            <td className="px-6 py-2 font-mono text-white/40">{r.volume?.toLocaleString()}</td>
                          </tr>))}</tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}

              {/* Prediction Results */}
              {hasPredictions ? (
                <>
                  {/* Model Tabs */}
                  <div className="flex items-center gap-4 bg-white/5 p-4 rounded-2xl border border-white/5">
                    <div className="flex gap-2 flex-1 overflow-x-auto">
                      {validModelKeys.map(modelKey => (
                        <button key={modelKey} onClick={() => setActiveModelTab(modelKey)}
                          className={`px-6 py-2 rounded-xl text-sm font-bold transition-all whitespace-nowrap ${
                            activeModelTab === modelKey ? 'bg-accent text-white shadow-lg' : 'text-white/40 hover:text-white hover:bg-white/5'}`}>
                          {models[modelKey]?.name || modelKey}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Per-Model Content */}
                  {activeModelTab && predictions[activeModelTab] && !predictions[activeModelTab].error && (
                    <>
                      {/* Model info */}
                      <div className="glass rounded-xl p-4 border border-white/5 flex items-center gap-4">
                        <div className="flex-1">
                          <span className="text-sm font-bold text-white">{predictions[activeModelTab].config?.name || activeModelTab}</span>
                          <span className="text-xs text-white/30 ml-3">
                            Context: {predictions[activeModelTab].config?.context_length} | 
                            Params: {predictions[activeModelTab].config?.params} | 
                            Lookback: {predictions[activeModelTab].lookback_used} pts
                          </span>
                        </div>
                        <span className="text-xs text-white/20">{predictions[activeModelTab].config?.description}</span>
                      </div>

                      {/* Metrics */}
                      {summaryMetrics[activeModelTab] && <Metrics metrics={summaryMetrics[activeModelTab]} />}

                      {/* Chart (single model view) */}
                      <MainChart data={data} predictions={predictions} activeModelKey={activeModelTab} mode="single" />

                      {/* Prediction Table */}
                      <PredictionTable data={predictions[activeModelTab].pred_df} modelName={models[activeModelTab]?.name} />
                    </>
                  )}

                  {/* Model Comparison (when >1 model) */}
                  <ModelComparison predictions={predictions} data={data} models={models} backtestResultsAll={backtestResultsAll} />

                  {/* Backtest Section */}
                  <BacktestSection
                    backtestResultsAll={backtestResultsAll}
                    models={models}
                    selectedModels={selectedModels}
                    onRunBacktest={runBacktestAll}
                    loading={loading}
                    backtestPredLen={backtestPredLen}
                    setBtPredLen={setBtPredLen}
                    backtestLookback={backtestLookback}
                    setBtLookback={setBtLookback}
                  />
                </>
              ) : (
                <MainChart data={data} />
              )}
            </div>
          ) : (
            <AnalysisList analyses={savedAnalyses} onLoad={loadAnalysis} onDelete={handleDeleteAnalysis} onRefresh={fetchAnalyses} />
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
