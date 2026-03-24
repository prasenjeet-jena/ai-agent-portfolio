import { 
  Layout, Smartphone, MessageSquare, DollarSign, 
  Monitor, Globe, Smile, Frown, Megaphone, Command, 
  ChevronRight, Compass, Shield, Zap, Activity
} from 'lucide-react';

const FilterSidebar = ({ filters, setFilters }) => {
  const sources = [
    { id: 'all', label: 'All Sources', icon: Layout },
    { id: 'App Store', label: 'App Store', icon: Smartphone },
    { id: 'NPS Surveys', label: 'NPS Surveys', icon: MessageSquare },
    { id: 'Sales Calls', label: 'Sales Notes', icon: DollarSign },
  ];

  const segments = [
    { id: 'all', label: 'All Segments' },
    { id: 'enterprise', label: 'Enterprise' },
    { id: 'pro', label: 'Pro' },
    { id: 'free', label: 'Free' },
  ];

  const platforms = [
    { id: 'all', label: 'All Platforms', icon: Command },
    { id: 'iOS', label: 'iOS', icon: Smartphone },
    { id: 'Android', label: 'Android', icon: Zap },
    { id: 'Web', label: 'Web', icon: Globe },
    { id: 'Desktop', label: 'Desktop', icon: Monitor },
  ];

  const sentiments = [
    { id: 'all', label: 'All Pulse', icon: Activity },
    { id: 'positive', label: 'Positive', icon: Smile, color: 'text-emerald-500' },
    { id: 'negative', label: 'Negative', icon: Frown, color: 'text-red-500' },
    { id: 'neutral', label: 'Neutral', icon: Megaphone, color: 'text-slate-400' },
  ];

  const regions = [
    { id: 'all', label: 'Global' },
    { id: 'US', label: 'United States' },
    { id: 'UK', label: 'United Kingdom' },
    { id: 'IN', label: 'India' },
    { id: 'AU', label: 'Australia' },
  ];

  const browsers = ['Chrome', 'Safari', 'Firefox'];

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const SectionHeader = ({ title, icon: Icon }) => (
    <div className="flex items-center gap-2 mb-4 mt-8 first:mt-0">
      <Icon className="w-3 h-3 text-slate-400" />
      <h2 className="text-[10px] font-black text-slate-400 uppercase tracking-widest">{title}</h2>
    </div>
  );

  return (
    <div className="w-72 bg-white border-r border-slate-200 min-h-screen flex flex-col p-6 sticky top-0 overflow-y-auto max-h-screen scrollbar-hide">
      
      <div className="mb-2 flex items-center gap-2 px-2">
        <div className="bg-blue-600 p-1.5 rounded-lg">
           <Compass className="w-4 h-4 text-white" />
        </div>
        <h1 className="font-bold text-slate-900 tracking-tight">Lens Explorer</h1>
      </div>

      <div className="mt-8">
        <SectionHeader title="Source Channels" icon={Shield} />
        <div className="space-y-1">
          {sources.map((source) => {
            const Icon = source.icon;
            const isActive = filters.source === source.id;
            return (
              <button
                key={source.id}
                onClick={() => handleFilterChange('source', source.id)}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl text-xs font-bold transition-all duration-200 ${
                  isActive 
                    ? 'bg-blue-600 text-white shadow-md shadow-blue-100' 
                    : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-white' : 'text-slate-400'}`} />
                {source.label}
              </button>
            );
          })}
        </div>
      </div>

      <SectionHeader title="User Identity" icon={Command} />
      <div className="flex flex-wrap gap-2">
        {segments.map((segment) => (
          <button
            key={segment.id}
            onClick={() => handleFilterChange('segment', segment.id)}
            className={`px-3 py-1.5 rounded-lg text-[10px] font-bold transition-all border ${
              filters.segment === segment.id 
                ? 'bg-slate-900 text-white border-slate-900 shadow-sm' 
                : 'bg-white text-slate-600 border-slate-200 hover:border-slate-300'
            }`}
          >
            {segment.label}
          </button>
        ))}
      </div>

      <SectionHeader title="Environment" icon={Monitor} />
      <div className="space-y-1">
        {platforms.map((platform) => {
          const Icon = platform.icon;
          const isActive = filters.platform === platform.id;
          return (
            <div key={platform.id}>
              <button
                onClick={() => handleFilterChange('platform', platform.id)}
                className={`w-full flex items-center justify-between px-3 py-2 rounded-xl text-xs font-bold transition-all ${
                  isActive ? 'bg-slate-100 text-slate-900' : 'text-slate-500 hover:text-slate-900 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-center gap-3">
                  <Icon className="w-3.5 h-3.5" />
                  {platform.label}
                </div>
                {isActive && <ChevronRight className="w-3 h-3 text-blue-600" />}
              </button>
            </div>
          );
        })}
      </div>

      <SectionHeader title="Signal Pulse" icon={Zap} />
      <div className="grid grid-cols-2 gap-2">
        {sentiments.map((s) => {
          const Icon = s.icon;
          const isActive = filters.sentiment === s.id;
          return (
            <button
              key={s.id}
              onClick={() => handleFilterChange('sentiment', s.id)}
              className={`flex items-center gap-2 px-3 py-2 rounded-xl text-[10px] font-black transition-all border ${
                isActive 
                  ? 'bg-white border-blue-600 text-blue-600 shadow-sm' 
                  : 'bg-slate-50 border-transparent text-slate-500 hover:bg-slate-100'
              }`}
            >
              <Icon className={`w-3 h-3 ${isActive ? 'text-blue-600' : s.color}`} />
              {s.label}
            </button>
          );
        })}
      </div>

      <SectionHeader title="Regional Context" icon={Globe} />
      <div className="grid grid-cols-2 gap-2">
        {regions.map((region) => (
          <button
            key={region.id}
            onClick={() => handleFilterChange('geography', region.id)}
            className={`px-2 py-2 rounded-lg text-[9px] font-black uppercase tracking-tighter transition-all border ${
              filters.geography === region.id 
                ? 'bg-blue-50 border-blue-200 text-blue-700' 
                : 'bg-white border-slate-100 text-slate-400 hover:border-slate-200 hover:text-slate-600'
            }`}
          >
            {region.id}
          </button>
        ))}
      </div>

      <div className="mt-auto pt-10">
        <div className="bg-slate-900 rounded-2xl p-4 relative overflow-hidden">
          <div className="absolute -right-4 -bottom-4 bg-blue-500/10 w-20 h-20 rounded-full blur-xl"></div>
          <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Status</p>
          <div className="flex items-center gap-2">
            <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse"></div>
            <p className="text-[10px] font-black text-white uppercase tracking-tight">Active Pulse Engine</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FilterSidebar;
