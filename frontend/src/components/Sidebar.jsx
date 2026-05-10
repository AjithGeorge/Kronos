import React, { useState } from 'react';
import { 
  Search, 
  History, 
  LayoutDashboard, 
  TrendingUp, 
  Star, 
  Clock,
  Plus,
  X
} from 'lucide-react';

const Sidebar = ({ activeSymbol, onSymbolChange, setView, currentView, activeTabs, onCloseTab }) => {
  const [search, setSearch] = useState('');

  const handleSearch = (e) => {
    e.preventDefault();
    if (search.trim()) {
      onSymbolChange(search.toUpperCase());
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

        <form onSubmit={handleSearch} className="relative mb-8">
          <input
            type="text"
            placeholder="Search symbol..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-white/5 border border-white/10 rounded-xl py-2.5 pl-10 pr-4 text-sm focus:outline-none focus:border-accent/50 transition-all"
          />
          <Search className="absolute left-3 top-2.5 text-white/30" size={16} />
        </form>

        <nav className="space-y-1">
          <button
            onClick={() => setView('dashboard')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${
              currentView === 'dashboard' ? 'bg-accent text-white shadow-lg shadow-accent/20' : 'text-white/50 hover:bg-white/5 hover:text-white'
            }`}
          >
            <LayoutDashboard size={18} />
            Dashboard
          </button>
          <button
            onClick={() => setView('history')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${
              currentView === 'history' ? 'bg-accent text-white shadow-lg shadow-accent/20' : 'text-white/50 hover:bg-white/5 hover:text-white'
            }`}
          >
            <History size={18} />
            History
          </button>
        </nav>

        <div className="mt-10">
          <div className="flex items-center justify-between mb-4 px-2">
            <span className="text-[10px] font-bold text-white/20 uppercase tracking-widest">Active Research</span>
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-white/20 font-bold bg-white/5 px-1.5 py-0.5 rounded">{activeTabs.length}</span>
            </div>
          </div>
          <div className="space-y-1 max-h-[300px] overflow-y-auto scrollbar-hide">
            {activeTabs.length > 0 ? (
              activeTabs.map(symbol => (
                <div 
                  key={symbol}
                  className={`group w-full flex items-center justify-between px-4 py-2.5 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                    activeSymbol === symbol ? 'text-accent bg-accent/10' : 'text-white/40 hover:text-white hover:bg-white/5'
                  }`}
                  onClick={() => onSymbolChange(symbol)}
                >
                  <span className="truncate mr-2">{symbol}</span>
                  <button 
                    onClick={(e) => {
                      e.stopPropagation();
                      onCloseTab(symbol);
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1 hover:bg-white/10 rounded transition-all"
                  >
                    <X size={12} />
                  </button>
                </div>
              ))
            ) : (
              <div className="px-4 py-8 text-center border border-dashed border-white/5 rounded-xl">
                <p className="text-[10px] text-white/20 italic">No symbols loaded</p>
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
            Symbols added here persist during your current research session.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
