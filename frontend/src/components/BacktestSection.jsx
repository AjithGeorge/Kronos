import React, { useState } from 'react';
import createPlotlyComponent from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';
import { FlaskConical, Target, Activity, BarChart, ChevronDown, ChevronUp, Gauge } from 'lucide-react';

const Plot = (createPlotlyComponent.default || createPlotlyComponent)(Plotly);

const Percent = ({ size, className }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <line x1="19" y1="5" x2="5" y2="19" /><circle cx="6.5" cy="6.5" r="2.5" /><circle cx="17.5" cy="17.5" r="2.5" />
  </svg>
);

const BtMetricCard = ({ title, value, subtitle, icon: Icon, color = 'blue' }) => {
  const colors = { blue: 'border-blue-500/20', green: 'border-emerald-500/20', red: 'border-red-500/20', orange: 'border-orange-500/20', purple: 'border-purple-500/20' };
  return (
    <div className={`glass rounded-2xl p-5 border-l-4 ${colors[color]} relative overflow-hidden group hover:scale-[1.02] transition-all`}>
      <div className="flex justify-between items-start mb-2">
        <span className="text-[10px] font-bold text-white/40 uppercase tracking-widest">{title}</span>
        <Icon size={16} className="text-white/20" />
      </div>
      <span className="text-2xl font-bold text-white tracking-tight">{value}</span>
      {subtitle && <p className="text-[11px] text-white/30 mt-1 font-medium">{subtitle}</p>}
    </div>
  );
};

const plotConfig = { responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToRemove: ['select2d', 'lasso2d'] };
const darkLayout = {
  template: 'plotly_dark', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
  margin: { l: 50, r: 20, t: 40, b: 50 }, hovermode: 'x unified', legend: { orientation: 'h', y: -0.15 },
  yaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.05)' },
};

const BacktestSection = ({ backtestResultsAll, models, selectedModels, onRunBacktest, loading, backtestPredLen, setBtPredLen, backtestLookback, setBtLookback }) => {
  const [activeTab, setActiveTab] = useState(null);
  const [expandedVol, setExpandedVol] = useState({});
  const hasResults = backtestResultsAll && Object.keys(backtestResultsAll).length > 0;

  if (hasResults && !activeTab) {
    const first = Object.keys(backtestResultsAll).find(k => !backtestResultsAll[k].error);
    if (first) setActiveTab(first);
  }

  const consolidated = !hasResults ? [] : Object.entries(backtestResultsAll).filter(([, r]) => !r.error).map(([k, r]) => {
    const cm = r.metrics?.close || {};
    return { key: k, model: r.config?.name || k, ctx: r.config?.context_length || '-', params: r.config?.params || '-',
      mae: cm.MAE || 0, rmse: cm.RMSE || 0, mape: cm['MAPE (%)'] || cm.MAPE || 0, count: cm.Count || 0, pl: r.pred_len || backtestPredLen };
  });

  const vals = (f) => consolidated.map(c => c[f]);
  const best = (f) => Math.min(...vals(f));
  const worst = (f) => Math.max(...vals(f));
  const cc = (v, f) => consolidated.length <= 1 ? '' : v === best(f) ? 'bg-emerald-500/20 text-emerald-300 font-bold' : v === worst(f) ? 'bg-red-500/20 text-red-300 font-bold' : '';

  const res = activeTab && hasResults ? backtestResultsAll[activeTab] : null;
  const predDf = res && !res.error ? (res.pred_df || []) : [];
  const actualDf = res && !res.error ? (res.actual_df || []) : [];
  const closeMet = res && !res.error ? (res.metrics?.close || {}) : {};
  const volMet = res && !res.error ? (res.metrics?.volume || {}) : {};

  return (
    <div className="space-y-6">
      {/* Config */}
      <div className="glass rounded-2xl p-6 border border-white/5">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-orange-500/20 rounded-xl flex items-center justify-center"><FlaskConical size={20} className="text-orange-400" /></div>
          <div><h2 className="text-lg font-bold">Backtest Accuracy Check</h2><p className="text-xs text-white/40">Test prediction accuracy using historical data</p></div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="space-y-2">
            <label className="text-xs font-semibold text-white/40 uppercase tracking-wider">Backtest Period ({backtestPredLen}d)</label>
            <input type="range" min="5" max="60" value={backtestPredLen} onChange={e => setBtPredLen(parseInt(e.target.value))} className="w-full accent-orange-400 cursor-pointer" />
          </div>
          <div className="space-y-2">
            <label className="text-xs font-semibold text-white/40 uppercase tracking-wider">Backtest Lookback ({backtestLookback})</label>
            <input type="range" min="100" max="512" value={backtestLookback} onChange={e => setBtLookback(parseInt(e.target.value))} className="w-full accent-orange-400 cursor-pointer" />
          </div>
          <div className="flex items-end">
            <button onClick={onRunBacktest} disabled={loading || selectedModels.length === 0}
              className="w-full h-11 bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-500/90 hover:to-red-500/90 text-white font-bold px-6 rounded-xl disabled:opacity-50 transition-all flex items-center justify-center gap-2 shadow-lg shadow-orange-500/20 border border-white/10 backdrop-blur-md">
              <FlaskConical size={18} /> Run Backtest for All Models
            </button>
          </div>
        </div>
      </div>

      {/* Consolidated Table */}
      {hasResults && consolidated.length > 0 && (
        <div className="glass rounded-2xl overflow-hidden border border-white/5">
          <div className="px-6 py-4 border-b border-white/5 bg-white/5 flex items-center gap-2">
            <Gauge size={18} className="text-orange-400" />
            <h3 className="font-bold text-sm uppercase tracking-wider text-white/80">Backtest Metrics — All Models</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead><tr className="bg-white/5 text-[10px] font-bold text-white/40 uppercase tracking-widest border-b border-white/5">
                <th className="px-6 py-4">Model</th><th className="px-6 py-4">Context</th><th className="px-6 py-4">Params</th>
                <th className="px-6 py-4">MAE</th><th className="px-6 py-4">RMSE</th><th className="px-6 py-4">MAPE (%)</th>
                <th className="px-6 py-4">Samples</th><th className="px-6 py-4">Period</th>
              </tr></thead>
              <tbody className="text-sm">{consolidated.map(r => (
                <tr key={r.key} className="border-b border-white/5 hover:bg-white/[0.02]">
                  <td className="px-6 py-3 font-bold text-white">{r.model}</td>
                  <td className="px-6 py-3 text-white/60 font-mono">{r.ctx}</td>
                  <td className="px-6 py-3 text-white/60 font-mono">{r.params}</td>
                  <td className={`px-6 py-3 font-mono ${cc(r.mae,'mae')}`}>{r.mae.toFixed(4)}</td>
                  <td className={`px-6 py-3 font-mono ${cc(r.rmse,'rmse')}`}>{r.rmse.toFixed(4)}</td>
                  <td className={`px-6 py-3 font-mono ${cc(r.mape,'mape')}`}>{r.mape.toFixed(2)}%</td>
                  <td className="px-6 py-3 text-white/60 font-mono">{r.count}</td>
                  <td className="px-6 py-3 text-white/40">{r.pl} days</td>
                </tr>))}</tbody>
            </table>
          </div>
          <div className="px-6 py-3 border-t border-white/5 flex items-center gap-4 text-[10px] text-white/30">
            <span className="flex items-center gap-1"><span className="w-3 h-3 bg-emerald-500/30 rounded-sm inline-block"></span> Best</span>
            <span className="flex items-center gap-1"><span className="w-3 h-3 bg-red-500/30 rounded-sm inline-block"></span> Worst</span>
            <span>Lower values = better accuracy</span>
          </div>
        </div>
      )}

      {/* Per-model tabs */}
      {hasResults && (<div className="space-y-4">
        <div className="flex gap-2 bg-white/5 p-2 rounded-2xl border border-white/5">
          {Object.keys(backtestResultsAll).filter(k => !backtestResultsAll[k].error).map(mk => (
            <button key={mk} onClick={() => setActiveTab(mk)}
              className={`px-6 py-2 rounded-xl text-sm font-bold transition-all ${activeTab === mk ? 'bg-orange-500 text-white shadow-lg' : 'text-white/40 hover:text-white hover:bg-white/5'}`}>
              {models[mk]?.name || mk}
            </button>
          ))}
        </div>

        {res && !res.error && (<div className="space-y-6">
          {/* Metric cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <BtMetricCard title="MAE (Close)" value={(closeMet.MAE||0).toFixed(4)} subtitle="Lower is better" icon={Target} color="blue" />
            <BtMetricCard title="RMSE (Close)" value={(closeMet.RMSE||0).toFixed(4)} subtitle="Lower is better" icon={Activity} color="purple" />
            <BtMetricCard title="MAPE (Close)" value={`${(closeMet['MAPE (%)']||closeMet.MAPE||0).toFixed(2)}%`} subtitle="Percentage error" icon={Percent} color="orange" />
            <BtMetricCard title="Samples" value={closeMet.Count||0} subtitle="Data points" icon={BarChart} color="green" />
          </div>

          {volMet.MAE !== undefined && (<div>
            <button onClick={() => setExpandedVol(p => ({...p, [activeTab]: !p[activeTab]}))}
              className="flex items-center gap-2 text-xs font-bold text-white/40 uppercase tracking-wider mb-3 hover:text-white/60 transition-colors">
              {expandedVol[activeTab] ? <ChevronUp size={14}/> : <ChevronDown size={14}/>} VOLUME Accuracy Metrics
            </button>
            {expandedVol[activeTab] && (<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <BtMetricCard title="MAE (Vol)" value={(volMet.MAE||0).toFixed(4)} subtitle="Lower is better" icon={Target} color="blue"/>
              <BtMetricCard title="RMSE (Vol)" value={(volMet.RMSE||0).toFixed(4)} subtitle="Lower is better" icon={Activity} color="purple"/>
              <BtMetricCard title="MAPE (Vol)" value={`${(volMet['MAPE (%)']||volMet.MAPE||0).toFixed(2)}%`} subtitle="Pct error" icon={Percent} color="orange"/>
              <BtMetricCard title="Samples" value={volMet.Count||0} subtitle="Data points" icon={BarChart} color="green"/>
            </div>)}
          </div>)}

          {/* Actual vs Predicted chart */}
          <div className="glass rounded-2xl p-6 border border-white/5">
            <h3 className="text-sm font-bold uppercase tracking-wider text-white/60 mb-4">Actual vs Predicted — {res.config?.name}</h3>
            <Plot data={[
              { x: actualDf.map(d=>d.datetime), y: actualDf.map(d=>d.close), type:'scatter', mode:'lines+markers', name:'Actual Close', line:{color:'#3b82f6',width:3}, marker:{size:6} },
              { x: predDf.map(d=>d.datetime), y: predDf.map(d=>d.close), type:'scatter', mode:'lines+markers', name:'Predicted Close', line:{color:'#ef4444',width:2,dash:'dash'}, marker:{size:5} }
            ]} layout={{...darkLayout, title:'Backtest: Actual vs Predicted Close Price', height:500, yaxis:{...darkLayout.yaxis,title:'Price'}}} config={plotConfig} className="w-full" useResizeHandler />
          </div>

          {/* Volume comparison */}
          {actualDf.length > 0 && actualDf[0].volume !== undefined && (
            <div className="glass rounded-2xl p-6 border border-white/5">
              <h3 className="text-sm font-bold uppercase tracking-wider text-white/60 mb-4">Volume Comparison — {res.config?.name}</h3>
              <Plot data={[
                { x: actualDf.map(d=>d.datetime), y: actualDf.map(d=>d.volume), type:'bar', name:'Actual Volume', marker:{color:'#3b82f6',opacity:0.7} },
                { x: predDf.map(d=>d.datetime), y: predDf.map(d=>d.volume), type:'bar', name:'Predicted Volume', marker:{color:'#ef4444',opacity:0.7} }
              ]} layout={{...darkLayout, title:`Volume Comparison — ${res.config?.name}`, height:400, barmode:'group', yaxis:{...darkLayout.yaxis,title:'Volume'}}} config={plotConfig} className="w-full" useResizeHandler />
            </div>
          )}

          {/* Detailed table */}
          <div className="glass rounded-2xl overflow-hidden border border-white/5">
            <div className="px-6 py-4 border-b border-white/5 bg-white/5"><h3 className="font-bold text-sm uppercase tracking-wider text-white/80">Detailed Comparison</h3></div>
            <div className="overflow-x-auto max-h-[400px] scrollbar-hide">
              <table className="w-full text-left border-collapse">
                <thead><tr className="bg-white/5 text-[10px] font-bold text-white/40 uppercase tracking-widest border-b border-white/5">
                  <th className="px-6 py-4">Date</th><th className="px-6 py-4">Actual Close</th><th className="px-6 py-4">Predicted Close</th><th className="px-6 py-4">Error</th><th className="px-6 py-4">Error %</th>
                </tr></thead>
                <tbody className="text-sm">{actualDf.map((a,i) => {
                  const p = predDf[i]; if(!p) return null;
                  const err = a.close - p.close;
                  const ep = err / a.close * 100;
                  const ec = Math.abs(ep)<2 ? 'text-emerald-400' : Math.abs(ep)<5 ? 'text-yellow-400' : 'text-red-400';
                  return (<tr key={i} className="border-b border-white/5 hover:bg-white/[0.02]">
                    <td className="px-6 py-3 text-white/60">{new Date(a.datetime).toLocaleDateString(undefined,{year:'numeric',month:'short',day:'numeric'})}</td>
                    <td className="px-6 py-3 text-white/80 font-mono">{a.close.toFixed(4)}</td>
                    <td className="px-6 py-3 text-white/80 font-mono">{p.close.toFixed(4)}</td>
                    <td className={`px-6 py-3 font-mono ${ec}`}>{err.toFixed(4)}</td>
                    <td className={`px-6 py-3 font-mono font-bold ${ec}`}>{ep.toFixed(2)}%</td>
                  </tr>);
                })}</tbody>
              </table>
            </div>
          </div>

          <div className="glass rounded-xl p-4 border border-white/5 flex items-center gap-2">
            <FlaskConical size={16} className="text-orange-400" />
            <span className="text-xs text-white/40">Backtest with {res.pred_len || backtestPredLen} days prediction using {res.config?.name}.</span>
          </div>
        </div>)}
      </div>)}
    </div>
  );
};

export default BacktestSection;
