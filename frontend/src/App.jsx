import React, { useState, useEffect, useCallback } from 'react';
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

let sessionCounter = 0;
const makeSessionId = () => `sess_${Date.now()}_${++sessionCounter}`;

const defaultSession = (overrides = {}) => ({
  id: makeSessionId(),
  symbol: 'RECLTD.NS',
  period: '1y',
  interval: '1d',
  predLen: 45,
  lookback: 400,
  startOffset: 0,
  backtestPredLen: 45,
  backtestLookback: 400,
  selectedModels: ['kronos-base','kronos-small','kronos-mini'],
  data: [],
  predictions: {},
  backtestResultsAll: null,
  summaryMetrics: {},
  activeModelTab: null,
  showRawData: false,
  ...overrides,
});

function App() {
  // Session-based state
  const [sessions, setSessions] = useState(() => {
    const init = defaultSession();
    return { [init.id]: init };
  });
  const [sessionOrder, setSessionOrder] = useState(() => Object.keys(sessions));
  const [activeSessionId, setActiveSessionId] = useState(() => Object.keys(sessions)[0]);

  // Global state (not per-session)
  const [models, setModels] = useState({});
  const [loading, setLoading] = useState(false);
  const [savedAnalyses, setSavedAnalyses] = useState([]);
  const [view, setView] = useState('dashboard');

  // Convenience: active session
  const s = sessions[activeSessionId] || defaultSession();

  const updateSession = useCallback((updates, sessionId) => {
    const id = sessionId || activeSessionId;
    setSessions(prev => ({ ...prev, [id]: { ...prev[id], ...updates } }));
  }, [activeSessionId]);

  useEffect(() => { fetchModels(); fetchAnalyses(); }, []);

  const fetchModels = async () => {
    try { const res = await axios.get(`${API_BASE}/models`); setModels(res.data); }
    catch (err) { console.error('Error fetching models:', err); }
  };

  const fetchAnalyses = async () => {
    try { const res = await axios.get(`${API_BASE}/analyses`); setSavedAnalyses(res.data); }
    catch (err) { console.error('Error fetching analyses:', err); }
  };

  // --- Session management ---
  const addNewSession = (overrides = {}) => {
    const sess = defaultSession(overrides);
    setSessions(prev => ({ ...prev, [sess.id]: sess }));
    setSessionOrder(prev => [...prev, sess.id]);
    setActiveSessionId(sess.id);
    setView('dashboard');
    return sess.id;
  };

  const switchSession = (sessionId) => {
    if (sessions[sessionId]) {
      setActiveSessionId(sessionId);
      setView('dashboard');
    }
  };

  const closeSession = (sessionId) => {
    setSessionOrder(prev => prev.filter(id => id !== sessionId));
    setSessions(prev => {
      const next = { ...prev };
      delete next[sessionId];
      return next;
    });
    if (activeSessionId === sessionId) {
      const remaining = sessionOrder.filter(id => id !== sessionId);
      if (remaining.length > 0) {
        setActiveSessionId(remaining[remaining.length - 1]);
      } else {
        const newSess = defaultSession();
        setSessions(prev => ({ ...prev, [newSess.id]: newSess }));
        setSessionOrder([newSess.id]);
        setActiveSessionId(newSess.id);
      }
    }
  };

  // --- Metrics calculator ---
  const calcMetrics = (preds, histData) => {
    if (!histData || histData.length === 0 || !preds) return {};
    const results = {};
    const lastHist = histData[histData.length - 1];
    Object.keys(preds).forEach(mk => {
      const pred = preds[mk];
      if (pred.error || !pred.pred_df || pred.pred_df.length === 0) return;
      const df = pred.pred_df;
      const fp = df[df.length - 1].close;
      const lp = lastHist.close;
      const cp = ((fp - lp) / lp) * 100;
      const mn = Math.min(...df.map(d => d.low || d.close));
      const mx = Math.max(...df.map(d => d.high || d.close));
      const av = df.reduce((s, d) => s + (d.volume || 0), 0) / df.length;
      const lv = lastHist.volume;
      results[mk] = {
        lastPrice: lp, futurePrice: fp, changePct: cp, minPred: mn, maxPred: mx,
        predRange: mx - mn, avgPredVol: av,
        volChangePct: lv ? ((av - lv) / lv) * 100 : 0,
        trend: cp > 0 ? 'Bullish' : cp < 0 ? 'Bearish' : 'Neutral',
        highDelta: ((mx - lp) / lp) * 100,
        lowDelta: ((mn - lp) / lp) * 100,
        period: df.length
      };
    });
    return results;
  };

  // --- Data / Prediction / Backtest ---
  const loadSymbolData = async () => {
    if (!s.symbol) return;
    setLoading(true);
    try {
      const res = await axios.get(`${API_BASE}/data`, { params: { symbol: s.symbol, period: s.period, interval: s.interval } });
      updateSession({ data: res.data, predictions: {}, backtestResultsAll: null, summaryMetrics: {} });
    } catch (err) {
      console.error('Error loading data:', err);
      alert('Failed to load data for this symbol/config.');
    } finally { setLoading(false); }
  };

  const runPredictions = async () => {
    if (s.data.length === 0) return;
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/predict`, {
        symbol: s.symbol, period: s.period, interval: s.interval,
        models: s.selectedModels, pred_len: s.predLen, lookback_limit: s.lookback,
        start_offset: s.startOffset
      });
      const metrics = calcMetrics(res.data, s.data);
      const firstValid = Object.keys(res.data).find(k => !res.data[k].error);
      updateSession({ predictions: res.data, summaryMetrics: metrics, activeModelTab: firstValid || s.activeModelTab });
    } catch (err) { console.error('Error running predictions:', err); }
    finally { setLoading(false); }
  };

  const runBacktestAll = async () => {
    if (s.data.length === 0 || s.selectedModels.length === 0) return;
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/backtest-all`, {
        symbol: s.symbol, period: s.period, interval: s.interval,
        models: s.selectedModels,
        backtest_pred_len: s.backtestPredLen,
        backtest_lookback: s.backtestLookback
      });
      updateSession({ backtestResultsAll: res.data });
    } catch (err) {
      console.error('Error running backtest:', err);
      alert(err.response?.data?.detail || 'Backtest failed');
    } finally { setLoading(false); }
  };

  const saveAnalysis = async () => {
    const hasPreds = Object.keys(s.predictions).length > 0;
    const hasBt = s.backtestResultsAll && Object.keys(s.backtestResultsAll).length > 0;
    if (!hasPreds) return;
    try {
      await axios.post(`${API_BASE}/analyses`, {
        symbol: s.symbol, period: s.period, interval: s.interval,
        pred_config: { symbol: s.symbol, period: s.period, interval: s.interval, models: s.selectedModels, pred_len: s.predLen, lookback_limit: s.lookback, start_offset: s.startOffset },
        predictions: s.predictions,
        backtest_results: hasBt ? s.backtestResultsAll : null,
        backtest_config: hasBt ? { symbol: s.symbol, period: s.period, interval: s.interval, models: s.selectedModels, lookback: s.backtestLookback, pred_len: s.backtestPredLen } : null
      });
      fetchAnalyses();
      alert('Analysis saved successfully!');
    } catch (err) {
      console.error('Error saving:', err);
      alert('Failed to save: ' + (err.response?.data?.detail || err.message));
    }
  };

  const loadAnalysis = async (key) => {
    setLoading(true);
    try {
      const res = await axios.get(`${API_BASE}/analyses/${key}`);
      const { metadata, predictions: preds, pred_config, backtest_results: bt, backtest_config: btConfig } = res.data;
      const sym = metadata.symbol || 'UNKNOWN';
      const per = metadata.period || '1y';
      const intv = metadata.interval || '1d';

      let rawData = [];
      try {
        const dataRes = await axios.get(`${API_BASE}/data`, { params: { symbol: sym, period: per, interval: intv } });
        rawData = dataRes.data;
      } catch (e) { console.warn('Could not reload raw data:', e); }

      const loadedPreds = preds || {};
      const metrics = rawData.length > 0 ? calcMetrics(loadedPreds, rawData) : {};
      const firstValid = Object.keys(loadedPreds).find(k => !loadedPreds[k].error);

      // Create a new session for the loaded analysis
      const newId = addNewSession({
        symbol: sym, period: per, interval: intv,
        predLen: pred_config?.pred_len || 45,
        lookback: pred_config?.lookback_limit || 400,
        startOffset: pred_config?.start_offset || 0,
        backtestPredLen: btConfig?.pred_len || 45,
        backtestLookback: btConfig?.lookback || 400,
        selectedModels: pred_config?.models || ['kronos-base'],
        data: rawData,
        predictions: loadedPreds,
        backtestResultsAll: (bt && Object.keys(bt).length > 0) ? bt : null,
        summaryMetrics: metrics,
        activeModelTab: firstValid || null,
      });
      setView('dashboard');
    } catch (err) {
      console.error('Error loading analysis:', err);
      alert('Failed to load: ' + (err.response?.data?.detail || err.message));
    } finally { setLoading(false); }
  };

  const handleDeleteAnalysis = () => { fetchAnalyses(); };

  const hasPredictions = s.predictions && Object.keys(s.predictions).length > 0;
  const validModelKeys = hasPredictions ? Object.keys(s.predictions).filter(k => !s.predictions[k].error) : [];

  // Build sidebar session list
  const sidebarSessions = sessionOrder.map(id => {
    const sess = sessions[id];
    if (!sess) return null;
    const hasPred = sess.predictions && Object.keys(sess.predictions).length > 0;
    const hasBt = sess.backtestResultsAll && Object.keys(sess.backtestResultsAll).length > 0;
    return { id, symbol: sess.symbol, period: sess.period, interval: sess.interval, hasPredictions: hasPred, hasBacktest: hasBt };
  }).filter(Boolean);

  return (
    <div className="flex h-screen bg-[#0a0a0a] text-white overflow-hidden font-sans">
      <Sidebar
        sessions={sidebarSessions}
        activeSessionId={activeSessionId}
        onSwitchSession={switchSession}
        onCloseSession={closeSession}
        onAddSession={() => addNewSession()}
        setView={setView}
        currentView={view}
      />

      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="h-16 glass flex items-center justify-between px-6 z-10 border-b border-white/5">
          <div className="flex items-center gap-3">
            <TrendingUp className="text-accent" size={24} />
            <h1 className="text-xl font-bold tracking-tight">Kronos <span className="text-accent">Stock Predictor</span></h1>
          </div>
          <div className="flex items-center gap-4">
            <button onClick={saveAnalysis} disabled={!hasPredictions}
              className="flex items-center gap-2 bg-white/5 hover:bg-accent/20 border border-white/10 hover:border-accent/50 text-white hover:text-accent disabled:opacity-50 px-4 py-2 rounded-xl backdrop-blur-md transition-all shadow-lg shadow-black/20">
              <Save size={16} /><span className="text-sm font-bold">Save Analysis</span>
            </button>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto p-6 scrollbar-hide">
          {view === 'dashboard' ? (
            <div className="max-w-[1400px] mx-auto space-y-6 pb-20">
              {/* Configuration Bar */}
              <div className="glass rounded-2xl p-6 space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-6">
                  <div className="space-y-2">
                    <label className="text-xs font-semibold text-white/40 uppercase tracking-wider">Stock Symbol</label>
                    <input type="text" value={s.symbol} onChange={e => updateSession({ symbol: e.target.value.toUpperCase() })}
                      className="w-full bg-white/5 border border-white/10 rounded-xl py-2 px-4 text-sm focus:outline-none focus:border-accent/50 transition-all font-bold"
                      placeholder="e.g. RELIANCE.NS" />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-semibold text-white/40 uppercase tracking-wider">Historical Period</label>
                    <select value={s.period} onChange={e => updateSession({ period: e.target.value })}
                      className="w-full bg-[#0a0a0a] border border-white/10 rounded-xl py-2 px-4 text-sm focus:outline-none focus:border-accent/50 transition-all cursor-pointer">
                      {['1mo','3mo','6mo','1y','2y','5y'].map(p => <option key={p} value={p}>{p}</option>)}
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-semibold text-white/40 uppercase tracking-wider">Interval</label>
                    <select value={s.interval} onChange={e => updateSession({ interval: e.target.value })}
                      className="w-full bg-[#0a0a0a] border border-white/10 rounded-xl py-2 px-4 text-sm focus:outline-none focus:border-accent/50 transition-all cursor-pointer">
                      {['1d','1h','30m'].map(i => <option key={i} value={i}>{i}</option>)}
                    </select>
                  </div>
                  <div className="space-y-2 lg:col-span-1">
                    <label className="text-xs font-semibold text-white/40 uppercase tracking-wider">Prediction Length ({s.predLen}d)</label>
                    <input type="range" min="5" max="60" value={s.predLen} onChange={e => updateSession({ predLen: parseInt(e.target.value) })} className="w-full accent-accent cursor-pointer" />
                  </div>
                  <div className="space-y-2 lg:col-span-1">
                    <label className="text-xs font-semibold text-white/40 uppercase tracking-wider">Lookback ({s.lookback})</label>
                    <input type="range" min="100" max="512" value={s.lookback} onChange={e => updateSession({ lookback: parseInt(e.target.value) })} className="w-full accent-accent cursor-pointer" />
                  </div>
                  <div className="space-y-2 lg:col-span-1">
                    <label className="text-xs font-semibold text-white/40 uppercase tracking-wider">Start Offset ({s.startOffset}d)</label>
                    <input type="range" min="0" max="100" value={s.startOffset} onChange={e => updateSession({ startOffset: parseInt(e.target.value) })} className="w-full accent-accent cursor-pointer" />
                  </div>
                </div>

                <div className="flex flex-col md:flex-row items-center justify-between gap-6 pt-4 border-t border-white/5">
                  <div className="flex flex-wrap gap-4 items-center">
                    <div className="space-y-2">
                      <label className="text-[10px] font-bold text-white/20 uppercase tracking-widest block">Model Comparison</label>
                      <div className="flex gap-2">
                        {Object.keys(models).map(key => (
                          <button key={key}
                            onClick={() => {
                              const cur = s.selectedModels;
                              updateSession({ selectedModels: cur.includes(key) ? cur.filter(k => k !== key) : [...cur, key] });
                            }}
                            className={`px-3 py-1.5 rounded-lg text-xs font-bold border transition-all ${
                              s.selectedModels.includes(key) ? 'bg-accent border-accent text-white' : 'bg-white/5 border-white/10 text-white/60 hover:bg-white/10'}`}>
                            {models[key].name}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <button onClick={loadSymbolData} disabled={loading || !s.symbol}
                      className="h-11 bg-white/5 hover:bg-white/10 text-white font-bold px-6 rounded-xl border border-white/10 backdrop-blur-md transition-all flex items-center gap-2 shadow-lg shadow-black/20">
                      <Database size={18} /> Load Data
                    </button>
                    <button onClick={runPredictions} disabled={loading || s.selectedModels.length === 0 || s.data.length === 0}
                      className="h-11 bg-gradient-to-r from-accent to-purple-500 hover:from-accent/90 hover:to-purple-500/90 text-white font-bold px-8 rounded-xl disabled:opacity-50 transition-all flex items-center gap-2 shadow-lg shadow-accent/20 border border-white/10 backdrop-blur-md">
                      {loading ? <RefreshCw className="animate-spin" size={18} /> : <TrendingUp size={18} />}
                      Run Prediction
                    </button>
                  </div>
                </div>
              </div>

              {/* Raw Data Table */}
              {s.data.length > 0 && (
                <div className="glass rounded-2xl overflow-hidden border border-white/5">
                  <button onClick={() => updateSession({ showRawData: !s.showRawData })}
                    className="w-full px-6 py-4 flex items-center justify-between bg-white/5 hover:bg-white/[0.07] transition-colors">
                    <div className="flex items-center gap-2">
                      <TableIcon size={16} className="text-accent" />
                      <span className="font-bold text-sm uppercase tracking-wider text-white/80">Raw Data</span>
                      <span className="text-[10px] text-white/30 ml-2">{s.data.length} rows loaded</span>
                    </div>
                    <span className="text-xs text-white/30">{s.showRawData ? '▲ Hide' : '▼ Show last 5 rows'}</span>
                  </button>
                  {s.showRawData && (
                    <div className="overflow-x-auto">
                      <table className="w-full text-left border-collapse">
                        <thead><tr className="bg-white/5 text-[10px] font-bold text-white/40 uppercase tracking-widest border-b border-white/5">
                          <th className="px-6 py-3">Date</th><th className="px-6 py-3">Open</th><th className="px-6 py-3">High</th>
                          <th className="px-6 py-3">Low</th><th className="px-6 py-3">Close</th><th className="px-6 py-3">Volume</th>
                        </tr></thead>
                        <tbody className="text-sm">{s.data.slice(-5).map((r, i) => (
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
                  <div className="flex items-center gap-4 bg-white/5 p-4 rounded-2xl border border-white/5">
                    <div className="flex gap-2 flex-1 overflow-x-auto">
                      {validModelKeys.map(modelKey => (
                        <button key={modelKey} onClick={() => updateSession({ activeModelTab: modelKey })}
                          className={`px-6 py-2 rounded-xl text-sm font-bold transition-all whitespace-nowrap ${
                            s.activeModelTab === modelKey ? 'bg-accent text-white shadow-lg' : 'text-white/40 hover:text-white hover:bg-white/5'}`}>
                          {models[modelKey]?.name || modelKey}
                        </button>
                      ))}
                    </div>
                  </div>

                  {s.activeModelTab && s.predictions[s.activeModelTab] && !s.predictions[s.activeModelTab].error && (
                    <>
                      <div className="glass rounded-xl p-4 border border-white/5 flex items-center gap-4">
                        <div className="flex-1">
                          <span className="text-sm font-bold text-white">{s.predictions[s.activeModelTab].config?.name || s.activeModelTab}</span>
                          <span className="text-xs text-white/30 ml-3">
                            Context: {s.predictions[s.activeModelTab].config?.context_length} | 
                            Params: {s.predictions[s.activeModelTab].config?.params} | 
                            Lookback: {s.predictions[s.activeModelTab].lookback_used} pts
                          </span>
                        </div>
                        <span className="text-xs text-white/20">{s.predictions[s.activeModelTab].config?.description}</span>
                      </div>
                      {s.summaryMetrics[s.activeModelTab] && <Metrics metrics={s.summaryMetrics[s.activeModelTab]} />}
                      <MainChart data={s.data} predictions={s.predictions} activeModelKey={s.activeModelTab} mode="single" />
                      <PredictionTable data={s.predictions[s.activeModelTab].pred_df} modelName={models[s.activeModelTab]?.name} />
                    </>
                  )}

                  <ModelComparison predictions={s.predictions} data={s.data} models={models} backtestResultsAll={s.backtestResultsAll} />

                  <BacktestSection
                    backtestResultsAll={s.backtestResultsAll}
                    models={models}
                    selectedModels={s.selectedModels}
                    onRunBacktest={runBacktestAll}
                    loading={loading}
                    backtestPredLen={s.backtestPredLen}
                    setBtPredLen={v => updateSession({ backtestPredLen: v })}
                    backtestLookback={s.backtestLookback}
                    setBtLookback={v => updateSession({ backtestLookback: v })}
                  />
                </>
              ) : (
                <MainChart data={s.data} />
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
