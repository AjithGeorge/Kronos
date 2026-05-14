import React from 'react';
import createPlotlyComponent from 'react-plotly.js/factory';
import Plotly from 'plotly.js-dist-min';
import { Activity } from 'lucide-react';

const Plot = (createPlotlyComponent.default || createPlotlyComponent)(Plotly);

const plotConfig = {
  responsive: true, displayModeBar: true, displaylogo: false,
  modeBarButtonsToRemove: ['select2d', 'lasso2d']
};

/**
 * MainChart supports two modes:
 * - mode='all' (default): shows historical + all model predictions stacked (original behavior)
 * - mode='single': shows historical + a single model's prediction (per-model tab view)
 * 
 * Props: data, predictions, activeModelKey, mode
 */
const MainChart = ({ data, predictions, activeModelKey, mode = 'all' }) => {
  if (!data || data.length === 0) return (
    <div className="glass rounded-2xl h-[600px] flex items-center justify-center border border-white/5">
      <div className="text-center">
        <Activity size={48} className="text-white/10 mx-auto mb-4" />
        <p className="text-white/30 font-medium">No data loaded. Please configure and load a symbol.</p>
      </div>
    </div>
  );

  const histData = data.slice(-150);

  const layout = {
    template: 'plotly_dark',
    grid: { rows: 2, columns: 1, roworder: 'top to bottom', pattern: 'independent' },
    height: 750,
    margin: { l: 50, r: 20, t: 40, b: 50 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    showlegend: true,
    legend: { orientation: 'h', y: -0.1 },
    xaxis: { rangeslider: { visible: true }, domain: [0, 1], anchor: 'y2' },
    yaxis: { title: 'Price', domain: [0.35, 1], gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.05)' },
    yaxis2: { title: 'Volume', domain: [0, 0.25], gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.05)' },
    hovermode: 'x unified'
  };

  const traces = [];

  // Historical candlestick
  traces.push({
    x: histData.map(d => d.datetime), open: histData.map(d => d.open), high: histData.map(d => d.high),
    low: histData.map(d => d.low), close: histData.map(d => d.close),
    type: 'candlestick', name: 'Historical',
    increasing: { line: { color: '#10b981' } }, decreasing: { line: { color: '#ef4444' } }, yaxis: 'y'
  });

  // Historical volume
  traces.push({
    x: histData.map(d => d.datetime), y: histData.map(d => d.volume),
    type: 'bar', name: 'Hist Volume', marker: { color: 'rgba(255,255,255,0.1)' }, yaxis: 'y2'
  });

  if (predictions) {
    const modelsToShow = mode === 'single' && activeModelKey
      ? [[activeModelKey, predictions[activeModelKey]]].filter(([, v]) => v && !v.error)
      : Object.entries(predictions).filter(([, p]) => !p.error);

    const colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b'];

    modelsToShow.forEach(([modelKey, pred], idx) => {
      if (!pred || !pred.pred_df) return;
      const predDf = pred.pred_df;
      const color = colors[idx % colors.length];

      // Prediction candlestick
      traces.push({
        x: predDf.map(d => d.datetime),
        open: predDf.map(d => d.open || d.close), high: predDf.map(d => d.high || d.close),
        low: predDf.map(d => d.low || d.close), close: predDf.map(d => d.close),
        type: 'candlestick', name: `${modelKey} Pred`,
        increasing: { line: { color } }, decreasing: { line: { color: '#f97316' } },
        opacity: 0.8, yaxis: 'y'
      });

      // Prediction volume
      traces.push({
        x: predDf.map(d => d.datetime), y: predDf.map(d => d.volume),
        type: 'bar', name: `${modelKey} Vol`,
        marker: { color, opacity: 0.4 }, yaxis: 'y2'
      });
    });

    const firstPredDf = modelsToShow.length > 0 ? modelsToShow[0][1]?.pred_df : null;
    const separatorDate = (firstPredDf && firstPredDf.length > 0) 
      ? firstPredDf[0].datetime 
      : histData[histData.length - 1].datetime;

    // Separator line
    layout.shapes = [{
      type: 'line',
      x0: separatorDate,
      x1: separatorDate,
      y0: 0, y1: 1, yref: 'paper',
      line: { color: 'white', width: 2, dash: 'dash' }
    }];
  }

  return (
    <div className="glass rounded-2xl p-6 border border-white/5">
      <Plot data={traces} layout={layout} config={plotConfig} className="w-full h-full" useResizeHandler={true} />
    </div>
  );
};

export default MainChart;
