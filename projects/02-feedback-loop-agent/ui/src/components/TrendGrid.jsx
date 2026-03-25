import { ChevronRight, Inbox, FileText } from 'lucide-react';

const TrendGrid = ({ filteredItems = [], staticClusters = [], onViewEvidence, onDraftPRD }) => {
  // -- Intersection-Based Dynamic Clustering --
  // We use the 'staticClusters' (the AI-generated Moat) as our architectural base.
  // Then we filter their contents based on the user's current 'filteredItems'.
  
  const filteredIds = new Set(filteredItems.map(i => i.feedback_id));

  const dynamicClusters = staticClusters.map(cluster => {
    // Find IDs in this cluster that are ALSO in the current filtered set
    const activeIds = (cluster.feedback_ids || []).filter(id => filteredIds.has(id));
    
    if (activeIds.length === 0) return null;

    return {
      ...cluster,
      count: activeIds.length,
      feedback_ids: activeIds,
      // Map the actual item objects back to these IDs for the modal
      filtered_raw_items: filteredItems.filter(item => activeIds.includes(item.feedback_id))
    };
  })
  .filter(Boolean)
  .sort((a, b) => {
    const priorityMap = { 'High': 3, 'Medium': 2, 'Low': 1 };
    return priorityMap[b.priority_level] - priorityMap[a.priority_level];
  });

  const getPriorityStyles = (priority) => {
    switch (priority) {
      case 'High':
        return 'bg-red-50 text-red-700 border-red-100';
      case 'Medium':
        return 'bg-amber-50 text-amber-700 border-amber-100';
      case 'Low':
        return 'bg-emerald-50 text-emerald-700 border-emerald-100';
      default:
        return 'bg-slate-50 text-slate-700 border-slate-100';
    }
  };

  if (dynamicClusters.length === 0) {
    return (
      <div className="bg-white rounded-2xl border-2 border-dashed border-slate-200 p-20 flex flex-col items-center justify-center text-center">
        <div className="bg-slate-50 p-6 rounded-full mb-6">
          <Inbox className="w-12 h-12 text-slate-300" />
        </div>
        <h3 className="text-xl font-bold text-slate-900 mb-2">No intelligence found</h3>
        <p className="text-slate-500 max-w-sm">
          Try broadening your filters. There are currently no feedback clusters matching this specific source or segment.
        </p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {dynamicClusters.map((cluster, index) => (
        <div 
          key={index} 
          className="group bg-white rounded-xl border border-slate-200 shadow-sm hover:shadow-lg transition-all duration-200 flex flex-col"
        >
          {/* Card Header */}
          <div className="p-5 border-b border-slate-100 flex items-start justify-between gap-4 border-t-4 border-t-slate-100 group-hover:border-t-blue-500 transition-all">
            <h3 className="font-bold text-slate-900 leading-tight">
              {cluster.theme_name}
            </h3>
            <div className={`px-2.5 py-0.5 rounded-full text-xs font-bold border ${getPriorityStyles(cluster.priority_level)}`}>
              {cluster.priority_level}
            </div>
          </div>

          {/* Card Body */}
          <div className="p-5 flex-grow">
            <div className="inline-block px-2 py-0.5 bg-slate-100 text-slate-600 rounded-md text-[10px] font-bold uppercase tracking-wider mb-3">
              {cluster.count} feedback items
            </div>
            <p className="text-sm text-slate-600 leading-relaxed line-clamp-3">
              {cluster.summary_of_issues}
            </p>
          </div>

          {/* Card Footer */}
          <div className="p-4 bg-slate-50/50 border-t border-slate-100 mt-auto flex gap-2 justify-end">
            <button 
              onClick={() => onDraftPRD(cluster)}
              className="px-4 py-2 bg-slate-900 border border-slate-900 rounded-lg text-xs font-bold text-white hover:bg-slate-800 flex items-center justify-center gap-1.5 transition-all shadow-sm active:scale-[0.98]"
            >
              <FileText className="w-3.5 h-3.5" />
              Draft PRD
            </button>
            <button 
              onClick={() => onViewEvidence({
                ...cluster,
                // Ensure App.jsx handleViewEvidence finds these items in its lookup
                feedback_ids: cluster.feedback_ids 
              })}
              className="px-4 py-2 bg-white border border-slate-200 rounded-lg text-xs font-bold text-slate-700 hover:bg-slate-50 flex items-center justify-center gap-1 transition-all group-hover:border-blue-400 group-hover:text-blue-600 shadow-sm active:scale-[0.98]"
            >
              View Evidence
              <ChevronRight className="w-3 h-3" />
            </button>
          </div>
        </div>
      ))}
    </div>
  );
};

export default TrendGrid;
