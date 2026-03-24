import { X } from 'lucide-react';

const EvidenceModal = ({ isOpen, onClose, clusterName, items = [] }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6">
      {/* Overlay */}
      <div 
        className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm" 
        onClick={onClose}
      ></div>

      {/* Modal Content */}
      <div className="relative bg-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[85vh] flex flex-col overflow-hidden animate-in fade-in zoom-in duration-200">
        
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
          <div>
            <h3 className="text-xl font-bold text-slate-900 leading-tight">
              Evidence: {clusterName}
            </h3>
            <p className="text-sm text-slate-500 mt-1">
              Showing {items.length} relevant feedback signals
            </p>
          </div>
          <button 
            onClick={onClose}
            className="p-2 hover:bg-slate-200 rounded-full transition-colors text-slate-400 hover:text-slate-600"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Table Content */}
        <div className="flex-grow overflow-y-auto p-6 pt-2">
          <table className="w-full text-left border-collapse min-w-[800px]">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="py-4 text-xs font-bold text-slate-400 uppercase tracking-wider w-24">Source</th>
                <th className="py-4 text-xs font-bold text-slate-400 uppercase tracking-wider w-40">Metadata</th>
                <th className="py-4 text-xs font-bold text-slate-400 uppercase tracking-wider w-32">Signal Type</th>
                <th className="py-4 text-xs font-bold text-slate-400 uppercase tracking-wider">Feedback Content</th>
                <th className="py-4 text-xs font-bold text-slate-400 uppercase tracking-wider w-24 text-right">Sentiment</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {items.map((item, idx) => (
                <tr key={idx} className="hover:bg-slate-50/50 transition-colors">
                  {/* Source */}
                  <td className="py-4">
                    <span className={`inline-flex items-center px-2 py-0.5 rounded text-[9px] font-black uppercase tracking-widest border ${
                      item.source === 'App Store' ? 'bg-blue-50 text-blue-700 border-blue-100' :
                      item.source === 'NPS Surveys' || item.source === 'NPS' ? 'bg-purple-50 text-purple-700 border-purple-100' :
                      'bg-amber-50 text-amber-700 border-amber-100'
                    }`}>
                      {item.source?.replace(' Surveys', '')}
                    </span>
                  </td>

                  {/* Metadata (User + Device) */}
                  <td className="py-4 pr-4">
                    <div className="flex flex-col gap-0.5">
                      <span className="text-xs font-bold text-slate-900 capitalize">
                        {item.user_context?.subscription_tier || 'Free'} {item.user_context?.user_type || 'User'}
                      </span>
                      <span className="text-[10px] text-slate-500 truncate max-w-[150px]">
                        {item.product_context?.platform} • {item.product_context?.device_model || 'Unknown Device'}
                      </span>
                    </div>
                  </td>

                  {/* Signal Type (Intent) */}
                  <td className="py-4 pr-4">
                    <span className="inline-flex items-center px-2 py-0.5 rounded bg-slate-100 text-slate-600 text-[10px] font-bold border border-slate-200 capitalize">
                      {item.enrichment?.intent?.replace('_', ' ') || 'General'}
                    </span>
                  </td>

                  {/* Feedback Content */}
                  <td className="py-4 pr-6">
                    <p className="text-sm text-slate-700 leading-relaxed line-clamp-2 italic group-hover:line-clamp-none transition-all">
                      "{item.raw_content?.text || 'No text content available'}"
                    </p>
                    <p className="text-[9px] text-slate-400 mt-1 uppercase font-bold tracking-tight">
                      Ref: {item.feedback_id || 'N/A'}
                    </p>
                  </td>

                  {/* Sentiment */}
                  <td className="py-4 text-right">
                    <span className={`inline-flex px-2 py-0.5 rounded-full text-[10px] font-bold border ${
                      item.enrichment?.true_sentiment === 'positive' ? 'bg-green-50 text-green-700 border-green-200' :
                      item.enrichment?.true_sentiment === 'negative' ? 'bg-red-50 text-red-700 border-red-200' :
                      item.enrichment?.true_sentiment === 'mixed' ? 'bg-amber-50 text-amber-700 border-amber-200' :
                      'bg-slate-50 text-slate-600 border-slate-200'
                    }`}>
                      {item.enrichment?.true_sentiment || 'neutral'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {items.length === 0 && (
            <div className="py-20 text-center">
              <p className="text-slate-400 text-sm italic">No raw items found for this cluster selection.</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-slate-50 border-t border-slate-100 flex justify-end">
          <button 
            onClick={onClose}
            className="px-6 py-2 bg-slate-900 text-white rounded-lg font-semibold text-sm hover:bg-slate-800 transition-colors"
          >
            Close Drill-Down
          </button>
        </div>
      </div>
    </div>
  );
};

export default EvidenceModal;
