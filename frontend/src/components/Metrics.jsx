import React from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  DollarSign, 
  BarChart, 
  Calendar, 
  ArrowUpRight, 
  ArrowDownRight,
  Activity,
  Maximize2,
  Target
} from 'lucide-react';

const MetricCard = ({ title, value, delta, deltaLabel, icon: Icon, color = 'blue' }) => {
  const isPositive = delta > 0;
  const isNegative = delta < 0;
  
  const colors = {
    blue: 'from-blue-500/20 to-transparent border-blue-500/20 text-blue-400',
    green: 'from-emerald-500/20 to-transparent border-emerald-500/20 text-emerald-400',
    red: 'from-red-500/20 to-transparent border-red-500/20 text-red-400',
    orange: 'from-orange-500/20 to-transparent border-orange-500/20 text-orange-400',
    purple: 'from-purple-500/20 to-transparent border-purple-500/20 text-purple-400',
  };

  return (
    <div className={`glass rounded-2xl p-5 border-l-4 ${colors[color]} relative overflow-hidden group hover:scale-[1.02] transition-all`}>
      <div className="flex justify-between items-start mb-2">
        <span className="text-[10px] font-bold text-white/40 uppercase tracking-widest">{title}</span>
        <Icon size={16} className="text-white/20 group-hover:text-white/40 transition-colors" />
      </div>
      <div className="flex flex-col">
        <span className="text-2xl font-bold text-white tracking-tight">{value}</span>
        {delta !== undefined && (
          <div className={`flex items-center gap-1 mt-1 text-[11px] font-bold ${isPositive ? 'text-emerald-400' : isNegative ? 'text-red-400' : 'text-white/30'}`}>
            {isPositive ? <ArrowUpRight size={12} /> : isNegative ? <ArrowDownRight size={12} /> : <Minus size={12} />}
            {deltaLabel || `${isPositive ? '+' : ''}${delta.toFixed(2)}%`}
          </div>
        )}
      </div>
    </div>
  );
};

const Metrics = ({ metrics }) => {
  // Check if we are showing backtest metrics or prediction metrics
  // If it has 'MAE', it's backtest
  const isBacktest = metrics && (metrics.MAE !== undefined || metrics.close?.MAE !== undefined);

  if (!metrics) return null;

  if (isBacktest) {
    const m = metrics.close || metrics;
    const mapeVal = m['MAPE (%)'] ?? m.MAPE ?? 0;
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mt-6">
        <MetricCard 
          title="MAE" 
          value={(m.MAE || 0).toFixed(4)} 
          icon={Target} 
          color="blue"
          delta={0}
          deltaLabel="Lower is better"
        />
        <MetricCard 
          title="RMSE" 
          value={(m.RMSE || 0).toFixed(4)} 
          icon={Activity} 
          color="purple"
          delta={0}
          deltaLabel="Lower is better"
        />
        <MetricCard 
          title="MAPE" 
          value={`${mapeVal.toFixed(2)}%`} 
          icon={Percent} 
          color="orange"
          delta={0}
          deltaLabel="Percentage error"
        />
        <MetricCard 
          title="Samples" 
          value={m.Count || 0} 
          icon={BarChart} 
          color="green"
          delta={0}
          deltaLabel="Data points"
        />
      </div>
    );
  }

  // Otherwise show prediction metrics
  const {
    lastPrice,
    futurePrice,
    changePct,
    minPred,
    maxPred,
    predRange,
    avgPredVol,
    volChangePct,
    trend,
    highDelta,
    lowDelta,
    period
  } = metrics;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mt-6">
      <MetricCard 
        title="Current Price" 
        value={lastPrice.toFixed(2)} 
        icon={DollarSign} 
        color="blue"
      />
      <MetricCard 
        title="Predicted Price" 
        value={futurePrice.toFixed(2)} 
        delta={changePct}
        icon={TrendingUp}
        color={changePct > 0 ? 'green' : 'red'}
      />
      <MetricCard 
        title="Predicted High" 
        value={maxPred.toFixed(2)} 
        delta={highDelta}
        icon={ArrowUpRight}
        color="green"
      />
      <MetricCard 
        title="Predicted Low" 
        value={minPred.toFixed(2)} 
        delta={lowDelta}
        icon={ArrowDownRight}
        color="red"
      />
      <MetricCard 
        title="Price Range" 
        value={predRange.toFixed(2)} 
        icon={Maximize2}
        color="purple"
      />
      <MetricCard 
        title="Trend Indicator" 
        value={trend} 
        delta={changePct}
        deltaLabel={`${changePct > 0 ? 'Bullish' : changePct < 0 ? 'Bearish' : 'Neutral'}`}
        icon={Activity}
        color={changePct > 0 ? 'green' : changePct < 0 ? 'red' : 'blue'}
      />
      <MetricCard 
        title="Avg Predicted Volume" 
        value={avgPredVol.toLocaleString(undefined, { maximumFractionDigits: 0 })} 
        delta={volChangePct}
        icon={BarChart}
        color="orange"
      />
      <MetricCard 
        title="Forecast Horizon" 
        value={`${period} Days`} 
        icon={Calendar}
        color="blue"
      />
    </div>
  );
};

// Mock Percent icon if missing from lucide
const Percent = ({ size, className }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    <line x1="19" y1="5" x2="5" y2="19"></line>
    <circle cx="6.5" cy="6.5" r="2.5"></circle>
    <circle cx="17.5" cy="17.5" r="2.5"></circle>
  </svg>
);

export default Metrics;
