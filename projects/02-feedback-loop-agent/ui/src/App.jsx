import { useState, useEffect } from 'react'
import { Sparkles, BarChart3, Clock, Activity, ThumbsUp, Search } from 'lucide-react'
import RiskRadar from './components/RiskRadar'
import TrendGrid from './components/TrendGrid'
import EvidenceModal from './components/EvidenceModal'
import FilterSidebar from './components/FilterSidebar'
import './App.css'

function App() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  // -- Modal State --
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [selectedCluster, setSelectedCluster] = useState(null)
  const [modalItems, setModalItems] = useState([])

  // -- Segmented Intelligence: State --
  const [filters, setFilters] = useState({ 
    source: 'all', 
    segment: 'all',
    platform: 'all',
    sentiment: 'all',
    geography: 'all'
  });

  useEffect(() => {
    fetch('/latest_intelligence.json')
      .then(res => res.json())
      .then(json => {
        setData(json)
        setLoading(false)
      })
      .catch(err => {
        console.error('Failed to load intelligence data:', err)
        setLoading(false)
      })
  }, [])

  // -- Drill-Down Logic --
  const handleViewEvidence = (cluster) => {
    if (!data?.raw_feedback_lookup) return;

    // Filter raw items based on the IDs stored in the cluster
    const items = data.raw_feedback_lookup.filter(item => 
      cluster.feedback_ids?.includes(item.feedback_id)
    );

    setSelectedCluster(cluster.theme_name);
    setModalItems(items);
    setIsModalOpen(true);
  };

  const handleViewPositiveEvidence = () => {
    if (!data?.raw_feedback_lookup) return;

    // DRILL-DOWN RULE: Only show strictly Positive/Neutral items
    const items = data.raw_feedback_lookup.filter(item => 
      ['positive', 'neutral'].includes(item.enrichment?.true_sentiment)
    );

    setSelectedCluster('Positive & Neutral Signals');
    setModalItems(items);
    setIsModalOpen(true);
  };

  const handleViewUnorganizedIssues = () => {
    if (!data?.raw_feedback_lookup) return;

    // Find all IDs that ARE in clusters
    const clusteredIds = new Set(
      data.clusters?.flatMap(c => c.feedback_ids || []) || []
    );

    // Filter for everything ELSE that is Negative/Mixed (The 'Hidden' Issues)
    const items = data.raw_feedback_lookup.filter(item => 
      !clusteredIds.has(item.feedback_id) && 
      ['negative', 'mixed'].includes(item.enrichment?.true_sentiment)
    );

    setSelectedCluster('Unorganized Strategic Issues');
    setModalItems(items);
    setIsModalOpen(true);
  };

  // Format the timestamp into a human-readable string
  const formatDate = (isoString) => {
    if (!isoString) return 'N/A'
    const date = new Date(isoString)
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  if (loading) {
    return (
      <div className="bg-slate-50 min-h-screen flex items-center justify-center">
        <p className="text-slate-500 text-lg animate-pulse">Loading intelligence data...</p>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="bg-slate-50 min-h-screen flex items-center justify-center">
        <p className="text-red-500 text-lg font-bold">Failed to load intelligence data.</p>
      </div>
    )
  }

  // -- Dynamic Filter Engine --
  // Multi-Dimensional Source of Truth.
  const filteredItems = data?.raw_feedback_lookup?.filter(item => {
    const sourceMatch = filters.source === 'all' || item.source === filters.source;
    const segmentMatch = filters.segment === 'all' || item.user_context?.user_segment === filters.segment;
    const platformMatch = filters.platform === 'all' || item.product_context?.platform === filters.platform;
    const sentimentMatch = filters.sentiment === 'all' || item.enrichment?.true_sentiment === filters.sentiment;
    const geoMatch = filters.geography === 'all' || item.user_context?.geography === filters.geography;
    
    return sourceMatch && segmentMatch && platformMatch && sentimentMatch && geoMatch;
  }) || [];

  // Verification log for background math
  console.log('Filtered Count:', filteredItems.length);

  // -- Metrics Calculation --
  // Pointing all existing Metric Cards to use filteredItems.length instead of static metadata
  const totalProcessed = filteredItems.length;
  
  // 1. Strictly Positive/Neutral (from within the filtered set)
  const positiveNeutralItems = filteredItems.filter(item => 
    ['positive', 'neutral'].includes(item.enrichment?.true_sentiment)
  );
  
  // 2. Unorganized Issues (Negative/Mixed items NOT in a cluster, from within the filtered set)
  const clusteredIds = new Set(data?.clusters?.flatMap(c => c.feedback_ids || []) || []);
  const unorganizedIssues = filteredItems.filter(item => 
    !clusteredIds.has(item.feedback_id) && 
    ['negative', 'mixed'].includes(item.enrichment?.true_sentiment)
  );

  return (
    <div className="flex bg-slate-50 min-h-screen font-sans">
      
      {/* ── Left Sidebar Filter Control ── */}
      <FilterSidebar filters={filters} setFilters={setFilters} />

      {/* ── Main Dashboard Content ── */}
      <div className="flex-1 p-10 pb-20">
        <div className="max-w-6xl mx-auto">
          
          <div className="flex items-center justify-between mb-8">
            <h1 className="text-3xl font-bold text-slate-900 tracking-tight">
              Command Center
            </h1>
            <div className="flex items-center gap-4 bg-white px-4 py-2 rounded-lg border border-slate-200">
               <Clock className="w-4 h-4 text-slate-400" />
               <p className="text-xs font-bold text-slate-500 uppercase tracking-widest">
                  Updated: {formatDate(data.metadata?.generated_at)}
               </p>
            </div>
          </div>

          {/* ── Intelligence Hero Card ── */}
          <div className="bg-white rounded-xl shadow-sm border-l-4 border-blue-600 p-8 mb-8 relative">
            <div className="absolute top-8 right-8">
              <Sparkles className="w-6 h-6 text-blue-500" />
            </div>
            <p className="text-xs font-semibold text-blue-600 uppercase tracking-wide mb-2">
              AI Recommendation
            </p>
            <h2 className="text-lg font-semibold text-slate-800 mb-3">
              Strategic Priority
            </h2>
            <p className="text-slate-600 leading-relaxed pr-20 text-sm">
              {data.pm_recommendation}
            </p>
          </div>

          {/* ── Metric Row ── */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            {/* Card 1: Quantity */}
            <div className="bg-white rounded-xl shadow-sm p-6 border border-slate-100 flex flex-col justify-center">
              <div className="flex items-center gap-2 mb-2">
                <BarChart3 className="w-4 h-4 text-blue-600" />
                <p className="text-[10px] font-bold text-slate-500 uppercase tracking-tight">Active Sample Volume</p>
              </div>
              <p className="text-3xl font-bold text-slate-900 leading-none">
                {totalProcessed}
              </p>
            </div>

            {/* Card 2: Positive/Neutral Balance */}
            <div 
              onClick={handleViewPositiveEvidence}
              className="group bg-white rounded-xl shadow-sm p-6 border border-emerald-100 cursor-pointer hover:bg-emerald-50 transition-all flex flex-col justify-center"
            >
              <div className="flex items-center gap-2 mb-2">
                <ThumbsUp className="w-4 h-4 text-emerald-600" />
                <p className="text-[10px] font-bold text-emerald-700 uppercase tracking-tight">Good Signals</p>
              </div>
              <div className="flex items-center justify-between">
                <p className="text-3xl font-bold text-emerald-700 leading-none">
                  {positiveNeutralItems.length}
                </p>
                <span className="text-[8px] font-black text-emerald-600 uppercase tracking-widest opacity-0 group-hover:opacity-100 transition-opacity underline decoration-emerald-200 decoration-2 underline-offset-4">Explore Evidence →</span>
              </div>
            </div>

            {/* Card 3: Unorganized Issues */}
            <div 
              onClick={handleViewUnorganizedIssues}
              className={`group bg-white rounded-xl shadow-sm p-6 border cursor-pointer transition-all flex flex-col justify-center ${
                unorganizedIssues.length > 0 ? 'border-red-100 hover:bg-red-50' : 'border-slate-100 opacity-50 grayscale'
              }`}
            >
              <div className="flex items-center gap-2 mb-2">
                <Activity className="w-4 h-4 text-red-600" />
                <p className="text-[10px] font-bold text-red-700 uppercase tracking-tight">Misc Issues</p>
              </div>
              <div className="flex items-center justify-between">
                <p className="text-3xl font-bold text-red-700 leading-none">
                  {unorganizedIssues.length}
                </p>
                {unorganizedIssues.length > 0 && (
                  <span className="text-[8px] font-black text-red-600 uppercase tracking-widest opacity-0 group-hover:opacity-100 transition-opacity underline decoration-red-200 decoration-2 underline-offset-4">Trace Items →</span>
                )}
              </div>
            </div>

            {/* Card 4: Signal Indexing */}
            <div className="bg-slate-900 rounded-xl shadow-sm p-6 border border-slate-800 flex flex-col justify-center">
              <div className="flex items-center gap-2 mb-2">
                <Search className="w-4 h-4 text-blue-400" />
                <p className="text-[10px] font-bold text-slate-400 uppercase tracking-tight">Intelligence Coverage</p>
              </div>
              <div className="flex items-end justify-between">
                <p className="text-3xl font-bold text-white leading-none">100<span className="text-sm font-black text-blue-400">%</span></p>
                <p className="text-[8px] font-black text-emerald-400 uppercase tracking-widest">Indexed</p>
              </div>
            </div>
          </div>

          {/* ── Risk Radar ── */}
          <RiskRadar risks={data.emerging_risks} />

          {/* ── Trend Grid ── */}
          <div className="mt-12">
            <h2 className="text-2xl font-bold text-slate-900 mb-6 tracking-tight flex items-center gap-3">
              📦 Active Feedback Clusters
              <span className="text-xs font-bold text-slate-400 bg-slate-100 px-2 py-1 rounded">AI Detected</span>
            </h2>
            <TrendGrid 
              filteredItems={filteredItems} 
              staticClusters={data.clusters}
              onViewEvidence={handleViewEvidence}
            />
          </div>

          {/* ── Raw Signal Explorer Placeholder ── */}
          <div className="mt-20 pt-10 border-t border-slate-200">
             <div className="flex items-center gap-6">
                <div className="bg-slate-900 p-4 rounded-full">
                  <Search className="w-6 h-6 text-blue-400" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-slate-900 tracking-tight mb-1">🔍 Raw Signal Explorer</h2>
                  <p className="text-slate-500 text-sm leading-relaxed max-w-2xl">
                    Every piece of feedback from App Store, NPS, and Sales is indexed. Our AI Analyst automatically clusters roughly {Math.round((clusteredIds.size / data.metadata?.total_items_processed) * 100)}% of signals. Use the Sidebar to slice data by source or segment.
                  </p>
                </div>
             </div>
          </div>

        </div>
      </div>

      {/* ── Drill-Down Modal ── */}
      <EvidenceModal 
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        clusterName={selectedCluster}
        items={modalItems}
      />
    </div>
  )
}

export default App
