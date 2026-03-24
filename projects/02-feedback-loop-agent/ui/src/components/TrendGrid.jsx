import { ChevronRight } from 'lucide-react';

const TrendGrid = ({ clusters = [], onViewEvidence }) => {
  if (!clusters || clusters.length === 0) return null;

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

  return (
    <section className="mt-12">
      <h2 className="text-2xl font-bold text-slate-900 mb-8 flex items-center gap-3">
        <span>📦</span> Active Feedback Clusters
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {clusters.map((cluster, index) => (
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
            <div className="p-4 bg-slate-50/50 border-t border-slate-100 mt-auto">
              <button 
                onClick={() => onViewEvidence(cluster)}
                className="w-full py-2.5 px-4 bg-white border border-slate-200 rounded-lg text-sm font-semibold text-slate-700 hover:bg-slate-50 flex items-center justify-center gap-1 transition-all group-hover:border-blue-400 group-hover:text-blue-600 group-hover:shadow-sm active:scale-[0.98]"
              >
                View Evidence
                <ChevronRight className="w-4 h-4 transition-transform group-hover:translate-x-0.5" />
              </button>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};

export default TrendGrid;
