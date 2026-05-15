import React from 'react';
import createPlotlyComponent from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';
import { GitCompareArrows } from 'lucide-react';

const formatXLabel = (isoStr) => {
  if (!isoStr) return '';
  const parts = isoStr.split(/[T ]/);
  const d = parts[0].split('-');
  if (d.length !== 3) return isoStr;
  const m = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][parseInt(d[1], 10) - 1];
  const cleanDate = `${parseInt(d[2], 10)} ${m} ${d[0]}`;
  const t = parts[1] || '';
  if (!t || t.startsWith('00:00:00')) return cleanDate;
  const tp = t.split(':');
  if (tp.length >= 2) return `${cleanDate} ${tp[0]}:${tp[1]}`;
  return cleanDate;
};

const Plot = (createPlotlyComponent.default || createPlotlyComponent)(Plotly);

const plotConfig = { responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToRemove: ['select2d', 'lasso2d'] };
const darkLayout = {
  template: 'plotly_dark', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
  margin: { l: 50, r: 20, t: 40, b: 50 }, hovermode: 'x unified',
  legend: { yanchor: 'top', y: 0.99, xanchor: 'left', x: 0.01 },
  yaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.05)' },
  xaxis: { type: 'category', nticks: 10, showgrid: false },
};

const ModelComparison = ({ predictions, data, models, backtestResultsAll }) => {
  if (!predictions || Object.keys(predictions).length <= 1) return null;

  const histData = data.slice(-150);
  const validPreds = Object.entries(predictions).filter(([, p]) => !p.error);

  // Comparison table data
  const tableRows = validPreds.map(([key, pred]) => {
    const df = pred.pred_df;
    const cfg = pred.config || models[key] || {};
    return {
      key, model: cfg.name || key, ctx: cfg.context_length || '-', params: cfg.params || '-',
      finalPrice: df[df.length - 1]?.close?.toFixed(4) || '-',
      minPred: Math.min(...df.map(d => d.low || d.close)).toFixed(4),
      maxPred: Math.max(...df.map(d => d.high || d.close)).toFixed(4),
      avgVol: (df.reduce((s, d) => s + (d.volume || 0), 0) / df.length).toLocaleString(undefined, { maximumFractionDigits: 0 }),
    };
  });

  // Line comparison chart traces
  const lineTraces = [];
  const lineColors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b'];

  // Historical reference
  lineTraces.push({
    x: histData.map(d => formatXLabel(d.datetime)), y: histData.map(d => d.close),
    type: 'scatter', mode: 'lines', name: 'Historical',
    line: { color: 'rgba(255,255,255,0.3)', dash: 'dash', width: 2 },
  });

  validPreds.forEach(([key, pred], idx) => {
    const df = pred.pred_df;
    const cfg = pred.config || models[key] || {};
    lineTraces.push({
      x: df.map(d => formatXLabel(d.datetime)), y: df.map(d => d.close),
      type: 'scatter', mode: 'lines+markers', name: `${cfg.name || key} (Forward)`,
      line: { width: 2, color: lineColors[idx % lineColors.length] }, marker: { size: 6 },
    });
  });

  // Add backtest traces if available
  if (backtestResultsAll && Object.keys(backtestResultsAll).length > 0) {
    Object.entries(backtestResultsAll).filter(([, r]) => !r.error).forEach(([key, res]) => {
      const cfg = res.config || models[key] || {};
      lineTraces.push({
        x: (res.pred_df || []).map(d => formatXLabel(d.datetime)), y: (res.pred_df || []).map(d => d.close),
        type: 'scatter', mode: 'lines+markers', name: `${cfg.name || key} (Backtest Pred)`,
        line: { width: 2, dash: 'dot' }, marker: { size: 5 }, opacity: 0.7,
      });
    });
    // Actual from first result
    const firstRes = Object.values(backtestResultsAll).find(r => !r.error);
    if (firstRes && firstRes.actual_df) {
      lineTraces.push({
        x: firstRes.actual_df.map(d => formatXLabel(d.datetime)), y: firstRes.actual_df.map(d => d.close),
        type: 'scatter', mode: 'lines+markers', name: 'Actual (Backtest Period)',
        line: { color: '#FF6B6B', width: 3 }, marker: { size: 8, symbol: 'diamond' },
      });
    }
  }

  // Candlestick comparison traces
  const candleTraces = [];
  candleTraces.push({
    x: histData.map(d => formatXLabel(d.datetime)), open: histData.map(d => d.open), high: histData.map(d => d.high),
    low: histData.map(d => d.low), close: histData.map(d => d.close),
    type: 'candlestick', name: 'Historical',
    increasing: { line: { color: '#10b981' } }, decreasing: { line: { color: '#ef4444' } }, opacity: 0.8,
  });

  const candleColors = [['blue', 'orange'], ['cyan', 'magenta'], ['yellow', 'purple'], ['lime', 'deeppink']];
  validPreds.forEach(([key, pred], idx) => {
    const df = pred.pred_df;
    const cfg = pred.config || models[key] || {};
    const [inc, dec] = candleColors[idx % candleColors.length];
    candleTraces.push({
      x: df.map(d => formatXLabel(d.datetime)), open: df.map(d => d.open || d.close), high: df.map(d => d.high || d.close),
      low: df.map(d => d.low || d.close), close: df.map(d => d.close),
      type: 'candlestick', name: `${cfg.name || key} Prediction`,
      increasing: { line: { color: inc } }, decreasing: { line: { color: dec } }, opacity: 0.6,
    });
  });

  if (backtestResultsAll && Object.keys(backtestResultsAll).length > 0) {
    Object.entries(backtestResultsAll).filter(([, r]) => !r.error).forEach(([key, res]) => {
      const cfg = res.config || models[key] || {};
      const df = res.pred_df || [];
      candleTraces.push({
        x: df.map(d => formatXLabel(d.datetime)), open: df.map(d => d.open || d.close), high: df.map(d => d.high || d.close),
        low: df.map(d => d.low || d.close), close: df.map(d => d.close),
        type: 'candlestick', name: `${cfg.name || key} (Backtest Pred)`,
        increasing: { line: { color: 'royalblue' } }, decreasing: { line: { color: 'darkorange' } }, opacity: 0.5,
      });
    });
    const firstRes = Object.values(backtestResultsAll).find(r => !r.error);
    if (firstRes && firstRes.actual_df) {
      candleTraces.push({
        x: firstRes.actual_df.map(d => formatXLabel(d.datetime)), y: firstRes.actual_df.map(d => d.close),
        type: 'scatter', mode: 'lines+markers', name: 'Actual (Backtest Period)',
        line: { color: '#FF6B6B', width: 3 }, marker: { size: 6, symbol: 'diamond' }, opacity: 0.8,
      });
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 bg-purple-500/20 rounded-xl flex items-center justify-center">
          <GitCompareArrows size={20} className="text-purple-400" />
        </div>
        <div>
          <h2 className="text-lg font-bold">Model Comparison</h2>
          <p className="text-xs text-white/40">Compare predictions across {validPreds.length} models</p>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="glass rounded-2xl overflow-hidden border border-white/5">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead><tr className="bg-white/5 text-[10px] font-bold text-white/40 uppercase tracking-widest border-b border-white/5">
              <th className="px-6 py-4">Model</th><th className="px-6 py-4">Context</th><th className="px-6 py-4">Params</th>
              <th className="px-6 py-4">Final Price</th><th className="px-6 py-4">Min Predicted</th>
              <th className="px-6 py-4">Max Predicted</th><th className="px-6 py-4">Avg Volume</th>
            </tr></thead>
            <tbody className="text-sm">{tableRows.map(r => (
              <tr key={r.key} className="border-b border-white/5 hover:bg-white/[0.02]">
                <td className="px-6 py-3 font-bold text-white">{r.model}</td>
                <td className="px-6 py-3 text-white/60 font-mono">{r.ctx}</td>
                <td className="px-6 py-3 text-white/60 font-mono">{r.params}</td>
                <td className="px-6 py-3 text-accent font-mono font-bold">{r.finalPrice}</td>
                <td className="px-6 py-3 text-red-400 font-mono">{r.minPred}</td>
                <td className="px-6 py-3 text-emerald-400 font-mono">{r.maxPred}</td>
                <td className="px-6 py-3 text-white/40 font-mono">{r.avgVol}</td>
              </tr>))}</tbody>
          </table>
        </div>
      </div>

      {/* Line Comparison Chart */}
      <div className="glass rounded-2xl p-6 border border-white/5">
        <h3 className="text-sm font-bold uppercase tracking-wider text-white/60 mb-4">Price Predictions Comparison</h3>
        <Plot data={lineTraces} layout={{...darkLayout, title: 'Price Predictions Comparison Across Models', height: 600}} config={plotConfig} className="w-full" useResizeHandler />
      </div>

      {/* Candlestick Comparison Chart */}
      <div className="glass rounded-2xl p-6 border border-white/5">
        <h3 className="text-sm font-bold uppercase tracking-wider text-white/60 mb-4">Price Predictions Comparison (Candlestick)</h3>
        <Plot data={candleTraces} layout={{...darkLayout, title: 'Price Predictions Comparison (Candlestick)', height: 700, xaxis: { type: 'category', nticks: 10, showgrid: false, rangeslider: { visible: true } }}} config={plotConfig} className="w-full" useResizeHandler />
      </div>
    </div>
  );
};

export default ModelComparison;
