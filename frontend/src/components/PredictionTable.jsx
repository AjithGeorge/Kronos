import React from 'react';
import { Table, Eye } from 'lucide-react';

const PredictionTable = ({ data, modelName }) => {
  if (!data || data.length === 0) return null;

  return (
    <div className="glass rounded-2xl overflow-hidden border border-white/5 mt-6">
      <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between bg-white/5">
        <div className="flex items-center gap-2">
          <Table size={18} className="text-accent" />
          <h3 className="font-bold text-sm uppercase tracking-wider text-white/80">
            {modelName} - Prediction Details
          </h3>
        </div>
        <div className="flex items-center gap-2 px-3 py-1 bg-accent/10 rounded-full border border-accent/20">
          <span className="text-[10px] font-bold text-accent">{data.length} Forecast Days</span>
        </div>
      </div>
      
      <div className="overflow-x-auto max-h-[400px] scrollbar-hide">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-white/5 text-[10px] font-bold text-white/40 uppercase tracking-widest border-b border-white/5">
              <th className="px-6 py-4">Date</th>
              <th className="px-6 py-4">Open</th>
              <th className="px-6 py-4">High</th>
              <th className="px-6 py-4">Low</th>
              <th className="px-6 py-4">Close</th>
              <th className="px-6 py-4">Volume</th>
            </tr>
          </thead>
          <tbody className="text-sm">
            {data.map((row, idx) => (
              <tr key={idx} className="border-b border-white/5 hover:bg-white/[0.02] transition-colors">
                <td className="px-6 py-3 font-medium text-white/60">
                  {new Date(row.datetime).toLocaleDateString(undefined, { 
                    year: 'numeric', 
                    month: 'short', 
                    day: 'numeric' 
                  })}
                </td>
                <td className="px-6 py-3 text-white/80 font-mono">{row.open.toFixed(4)}</td>
                <td className="px-6 py-3 text-emerald-400 font-mono">{row.high.toFixed(4)}</td>
                <td className="px-6 py-3 text-red-400 font-mono">{row.low.toFixed(4)}</td>
                <td className="px-6 py-3 font-bold text-white font-mono">{row.close.toFixed(4)}</td>
                <td className="px-6 py-3 text-white/40 font-mono">{row.volume.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default PredictionTable;
