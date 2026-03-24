import { useState, useEffect } from 'react'
import { Sparkles, BarChart3, Clock, Activity, ThumbsUp, Search } from 'lucide-react'
import RiskRadar from './components/RiskRadar'
import TrendGrid from './components/TrendGrid'
import EvidenceModal from './components/EvidenceModal'
import './App.css'

function App() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  // -- Modal State --
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [selectedCluster, setSelectedCluster] = useState(null)
  const [modalItems, setModalItems] = useState([])

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

  // Statistics calculation for the Balanced View
  const totalProcessed = data.metadata?.total_items_processed ?? 0
  
  // 1. Strictly Positive/Neutral (Unclustered or clustered)
  const positiveNeutralItems = data.raw_feedback_lookup?.filter(item => 
    ['positive', 'neutral'].includes(item.enrichment?.true_sentiment)
  ) || [];
  
  // 2. Unorganized Issues (Negative/Mixed items NOT in a cluster)
  const clusteredIds = new Set(data.clusters?.flatMap(c => c.feedback_ids || []) || []);
  const unorganizedIssues = data.raw_feedback_lookup?.filter(item => 
    !clusteredIds.has(item.feedback_id) && 
    ['negative', 'mixed'].includes(item.enrichment?.true_sentiment)
  ) || [];

  return (
    <div className="bg-slate-50 min-h-screen font-sans">
      <div className="max-w-6xl mx-auto p-8 pb-20">

        {/* ── Page Title ── */}
        <h1 className="text-3xl font-bold text-slate-900 mb-8">
          Product Feedback Intelligence Agent System
        </h1>

        {/* ── Intelligence Hero Card ── */}
        <div className="bg-white rounded-xl shadow-sm border-l-4 border-blue-600 p-6 mb-8 relative">
          <div className="absolute top-6 right-6">
            <Sparkles className="w-6 h-6 text-blue-500" />
          </div>
          <p className="text-xs font-semibold text-blue-600 uppercase tracking-wide mb-2">
            AI Recommendation
          </p>
          <h2 className="text-lg font-semibold text-slate-800 mb-3">
            Strategic Priority
          </h2>
          <p className="text-slate-600 leading-relaxed pr-10">
            {data.pm_recommendation}
          </p>
        </div>

        {/* ── Metric Row ── */}
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-8">

          {/* Card 1: Quantity */}
          <div className="bg-white rounded-xl shadow-sm p-5 border border-slate-100 flex flex-col justify-center">
            <div className="flex items-center gap-2 mb-2">
              <BarChart3 className="w-4 h-4 text-blue-600" />
              <p className="text-[10px] font-bold text-slate-500 uppercase tracking-tight">Total Volume</p>
            </div>
            <p className="text-2xl font-bold text-slate-900 leading-none">
              {totalProcessed}
            </p>
          </div>

          {/* Card 2: Freshness */}
          <div className="bg-white rounded-xl shadow-sm p-5 border border-slate-100 flex flex-col justify-center">
            <div className="flex items-center gap-2 mb-2">
              <Clock className="w-4 h-4 text-amber-600" />
              <p className="text-[10px] font-bold text-slate-500 uppercase tracking-tight">Last Sync</p>
            </div>
            <p className="text-sm font-bold text-slate-800 truncate">
              {formatDate(data.metadata?.generated_at)}
            </p>
          </div>

          {/* Card 3: Positive/Neutral Balance */}
          {/* Strictly showing Good signals prevents negative bias */}
          <div 
            onClick={handleViewPositiveEvidence}
            className="group bg-white rounded-xl shadow-sm p-5 border border-green-100 cursor-pointer hover:bg-green-50 transition-all flex flex-col justify-center"
          >
            <div className="flex items-center gap-2 mb-2">
              <ThumbsUp className="w-4 h-4 text-green-600" />
              <p className="text-[10px] font-bold text-green-700 uppercase tracking-tight">Good Signals</p>
            </div>
            <div className="flex items-center justify-between">
              <p className="text-2xl font-bold text-green-700 leading-none">
                {positiveNeutralItems.length}
              </p>
              <span className="text-[8px] font-black text-green-600 uppercase tracking-widest opacity-0 group-hover:opacity-100 transition-opacity">Explore</span>
            </div>
          </div>

          {/* Card 4: Unorganized Issues */}
          {/* Surfacing negative items that escaped the main clusters */}
          <div 
            onClick={handleViewUnorganizedIssues}
            className={`group bg-white rounded-xl shadow-sm p-5 border cursor-pointer transition-all flex flex-col justify-center ${
              unorganizedIssues.length > 0 ? 'border-red-100 hover:bg-red-50' : 'border-slate-100 opacity-50 grayscale'
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-4 h-4 text-red-600" />
              <p className="text-[10px] font-bold text-red-700 uppercase tracking-tight">Misc Issues</p>
            </div>
            <div className="flex items-center justify-between">
              <p className="text-2xl font-bold text-red-700 leading-none">
                {unorganizedIssues.length}
              </p>
              {unorganizedIssues.length > 0 && (
                <span className="text-[8px] font-black text-red-600 uppercase tracking-widest opacity-0 group-hover:opacity-100 transition-opacity">View All</span>
              )}
            </div>
          </div>

          {/* Card 5: Status */}
          <div className="bg-white rounded-xl shadow-sm p-5 border border-slate-100 flex flex-col justify-center">
            <div className="flex items-center gap-2 mb-2">
              <div className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </div>
              <p className="text-[10px] font-bold text-slate-500 uppercase tracking-tight">System</p>
            </div>
            <p className="text-sm font-bold text-emerald-600 tracking-tight uppercase">Live & Monitoring</p>
          </div>

        </div>

        {/* ── Risk Radar ── */}
        <RiskRadar risks={data.emerging_risks} />

        {/* ── Trend Grid ── */}
        <TrendGrid 
          clusters={data.clusters} 
          onViewEvidence={handleViewEvidence}
        />

        {/* ── Raw Signal Explorer Placeholder ── */}
        <div className="mt-16 pt-8 border-t border-slate-200">
          <div className="flex items-center gap-3 mb-4">
            <Search className="w-6 h-6 text-slate-400" />
            <h2 className="text-2xl font-bold text-slate-900 tracking-tight">🔍 Raw Signal Explorer</h2>
          </div>
          <p className="text-slate-500 font-medium text-sm leading-relaxed">
            100% of feedback items from App Store, NPS, and Sales are indexed. Our AI Analyst clusters roughly {Math.round((clusteredIds.size / totalProcessed) * 100)}% of signals into themes. The remaining items are available via the 'Good Signals' and 'Misc Issues' drill-downs.
          </p>
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
