import React, { useState } from 'react';
import axios from 'axios';
import { 
  FileText, Calendar, Database, Filter, Trash2, ExternalLink, RefreshCw
} from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

const AnalysisList = ({ analyses, onLoad, onDelete, onRefresh }) => {
  const [filterSymbol, setFilterSymbol] = useState('');
  const [filterPeriod, setFilterPeriod] = useState('All');
  const [deleting, setDeleting] = useState(null);

  const periods = ['All', ...new Set((analyses || []).map(a => a.period).filter(Boolean))];

  const filtered = (analyses || []).filter(a => {
    if (filterSymbol && !a.symbol?.toUpperCase().includes(filterSymbol.toUpperCase())) return false;
    if (filterPeriod !== 'All' && a.period !== filterPeriod) return false;
    return true;
  });

  const handleDelete = async (key) => {
    if (!window.confirm('Delete this analysis?')) return;
    setDeleting(key);
    try {
      await axios.delete(`${API_BASE}/analyses/${key}`);
      if (onDelete) onDelete(key);
    } catch (err) {
      console.error('Delete failed:', err);
    } finally {
      setDeleting(null);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-3xl font-bold mb-2">Stored Analyses</h2>
          <p className="text-white/40 text-sm">Browse and load your previous research results</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="relative">
            <input type="text" placeholder="Filter by symbol..." value={filterSymbol}
              onChange={e => setFilterSymbol(e.target.value)}
              className="bg-white/5 border border-white/10 rounded-xl py-2 pl-10 pr-4 text-sm focus:outline-none focus:border-accent/50 transition-all" />
            <Filter className="absolute left-3 top-2.5 text-white/30" size={16} />
          </div>
          <select value={filterPeriod} onChange={e => setFilterPeriod(e.target.value)}
            className="bg-[#0a0a0a] border border-white/10 rounded-xl py-2 px-4 text-sm focus:outline-none focus:border-accent/50 cursor-pointer">
            {periods.map(p => <option key={p} value={p}>{p}</option>)}
          </select>
          {onRefresh && (
            <button onClick={onRefresh} className="p-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl transition-all" title="Refresh">
              <RefreshCw size={16} className="text-white/40" />
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {filtered.length > 0 ? filtered.map(analysis => (
          <div key={analysis.key} className="glass rounded-2xl p-6 border border-white/5 hover:border-accent/30 transition-all group relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-accent/5 blur-3xl -mr-16 -mt-16 group-hover:bg-accent/10 transition-all"></div>
            <div className="flex justify-between items-start mb-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-white/5 rounded-xl flex items-center justify-center border border-white/10 text-accent group-hover:bg-accent group-hover:text-white transition-all">
                  <FileText size={24} />
                </div>
                <div>
                  <h3 className="text-xl font-bold">{analysis.symbol}</h3>
                  <div className="flex items-center gap-2 text-white/40 text-[11px] mt-1">
                    <Calendar size={12} />
                    {new Date(analysis.created_at).toLocaleDateString()}
                  </div>
                </div>
              </div>
              <div className="flex flex-col items-end gap-1">
                <span className="text-[10px] bg-white/5 px-2 py-0.5 rounded border border-white/10 font-bold uppercase text-white/40">{analysis.interval}</span>
                <span className="text-[10px] text-white/20">{analysis.period}</span>
              </div>
            </div>
            <div className="space-y-3 mb-6">
              <div className="flex items-center justify-between text-xs">
                <span className="text-white/40">Models Used</span>
                <div className="flex gap-1">
                  {analysis.model_names?.map(name => (
                    <span key={name} className="bg-accent/10 text-accent px-2 py-0.5 rounded-[4px] font-medium text-[10px]">{name}</span>
                  ))}
                </div>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-white/40">Data Points</span>
                <span className="font-medium">{analysis.has_backtest ? 'Predictions + Backtest' : 'Predictions Only'}</span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button onClick={() => onLoad(analysis.key)}
                className="flex-1 bg-gradient-to-r from-accent to-purple-500 hover:from-accent/90 hover:to-purple-500/90 text-white border border-white/10 rounded-xl py-3 text-sm font-bold backdrop-blur-md transition-all flex items-center justify-center gap-2 shadow-lg shadow-accent/20">
                <ExternalLink size={16} /> Load Analysis
              </button>
              <button onClick={() => handleDelete(analysis.key)} disabled={deleting === analysis.key}
                className="w-12 h-12 bg-white/5 hover:bg-red-500/20 hover:text-red-500 border border-white/10 rounded-xl flex items-center justify-center transition-all disabled:opacity-50">
                {deleting === analysis.key ? <RefreshCw size={18} className="animate-spin" /> : <Trash2 size={18} />}
              </button>
            </div>
          </div>
        )) : (
          <div className="col-span-2 py-24 glass rounded-3xl border border-dashed border-white/10 flex flex-col items-center justify-center gap-4">
            <Database size={48} className="text-white/10" />
            <div className="text-center">
              <p className="text-white/60 font-bold">No saved analyses found</p>
              <p className="text-white/30 text-sm">Run and save a prediction to see it here</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisList;
