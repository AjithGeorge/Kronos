import React, { useState } from 'react';
import { 
  Search, History, LayoutDashboard, TrendingUp, Clock, Plus, X,
  FlaskConical, BarChart2
} from 'lucide-react';

const Sidebar = ({ sessions, activeSessionId, onSwitchSession, onCloseSession, onAddSession, setView, currentView }) => {
  const [search, setSearch] = useState('');

  const handleSearch = (e) => {
    e.preventDefault();
    if (search.trim()) {
      onAddSession?.({ symbol: search.toUpperCase() });
      setSearch('');
    }
  };

  return (
    <div className="w-64 h-screen glass border-r border-white/5 flex flex-col z-20">
      <div className="p-6">
        <div className="flex items-center gap-2 mb-8">
          <div className="w-8 h-8 bg-accent rounded-lg flex items-center justify-center">
            <TrendingUp size={18} className="text-white" />
          </div>
          <span className="font-bold text-lg tracking-tight">KRONOS <span className="text-white/30">v2.0</span></span>
        </div>

        <nav className="space-y-1">
          <button onClick={() => setView('dashboard')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${
              currentView === 'dashboard' ? 'bg-accent text-white shadow-lg shadow-accent/20' : 'text-white/50 hover:bg-white/5 hover:text-white'}`}>
            <LayoutDashboard size={18} /> Dashboard
          </button>
          <button onClick={() => setView('history')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${
              currentView === 'history' ? 'bg-accent text-white shadow-lg shadow-accent/20' : 'text-white/50 hover:bg-white/5 hover:text-white'}`}>
            <History size={18} /> Saved Analyses
          </button>
        </nav>

        <div className="mt-10">
          <div className="flex items-center justify-between mb-4 px-2">
            <span className="text-[10px] font-bold text-white/20 uppercase tracking-widest">Active Research</span>
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-white/20 font-bold bg-white/5 px-1.5 py-0.5 rounded">{sessions?.length || 0}</span>
              <button onClick={() => onAddSession?.()} title="New research session"
                className="w-6 h-6 flex items-center justify-center bg-accent/20 hover:bg-accent/40 text-accent rounded-lg transition-all">
                <Plus size={14} />
              </button>
            </div>
          </div>
          <div className="space-y-1 max-h-[350px] overflow-y-auto scrollbar-hide">
            {sessions && sessions.length > 0 ? (
              sessions.map(sess => (
                <div key={sess.id}
                  className={`group w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs transition-all cursor-pointer ${
                    activeSessionId === sess.id ? 'bg-accent/10 border border-accent/20' : 'hover:bg-white/5 border border-transparent'}`}
                  onClick={() => onSwitchSession(sess.id)}>
                  <div className="flex flex-col min-w-0 flex-1 mr-2">
                    <span className={`font-bold truncate ${activeSessionId === sess.id ? 'text-accent' : 'text-white/60'}`}>
                      {sess.symbol || 'New Session'}
                    </span>
                    <div className="flex items-center gap-1.5 mt-0.5">
                      <span className="text-[9px] text-white/20 bg-white/5 px-1 rounded">{sess.period}</span>
                      <span className="text-[9px] text-white/20 bg-white/5 px-1 rounded">{sess.interval}</span>
                      {sess.hasPredictions && (
                        <span className="text-[9px] text-emerald-400/60" title="Has predictions">
                          <BarChart2 size={9} />
                        </span>
                      )}
                      {sess.hasBacktest && (
                        <span className="text-[9px] text-orange-400/60" title="Has backtest">
                          <FlaskConical size={9} />
                        </span>
                      )}
                    </div>
                  </div>
                  <button onClick={(e) => { e.stopPropagation(); onCloseSession(sess.id); }}
                    className="opacity-0 group-hover:opacity-100 p-1 hover:bg-white/10 rounded transition-all flex-shrink-0">
                    <X size={12} className="text-white/40" />
                  </button>
                </div>
              ))
            ) : (
              <div className="px-4 py-8 text-center border border-dashed border-white/5 rounded-xl">
                <p className="text-[10px] text-white/20 italic">No active research</p>
                <button onClick={() => onAddSession?.()} className="mt-2 text-[10px] text-accent hover:underline">
                  + Start new
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="mt-auto p-6 border-t border-white/5">
        <div className="bg-gradient-to-br from-accent/20 to-transparent p-4 rounded-2xl border border-accent/10">
          <div className="flex items-center gap-2 mb-2">
            <Clock size={14} className="text-accent" />
            <span className="text-[11px] font-bold text-white/60">Session Info</span>
          </div>
          <p className="text-[10px] text-white/30 leading-relaxed">
            Each tab retains its analysis. Use + to add sessions. Load saved analyses from History.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
