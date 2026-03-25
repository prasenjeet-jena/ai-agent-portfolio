import React, { useState, useEffect } from 'react';
import { X, FileText, Download, Copy, Check, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const ActionPanel = ({ isOpen, onClose, cluster }) => {
  const [copied, setCopied] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [prdContent, setPrdContent] = useState('');

  useEffect(() => {
    if (!isOpen || !cluster) return;

    const generatePRD = async () => {
      setIsLoading(true);
      setError(null);
      setPrdContent('');

      try {
        const response = await fetch('http://localhost:8000/generate-prd', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            theme: cluster.theme_name || cluster.cluster_name || 'Unnamed Feature',
            summary: cluster.summary_of_issues || 'No summary available',
            raw_feedback_items: cluster.filtered_raw_items || []
          })
        });

        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }

        const result = await response.json();
        setPrdContent(result);
      } catch (err) {
        console.error('Failed to generate PRD:', err);
        setError(err.message || 'An error occurred while communicating with the AI server.');
      } finally {
        setIsLoading(false);
      }
    };

    // Prevent re-fetching if we already have content for this exact cluster
    // NOTE: In a real app we might cache this by cluster ID, but for now we fetch fresh
    generatePRD();

  }, [isOpen, cluster]);

  const handleCopy = () => {
    navigator.clipboard.writeText(prdContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!isOpen || !cluster) return null;

  return (
    <>
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm z-50 transition-opacity"
        onClick={onClose}
      />

      {/* Slide-out Panel */}
      <div className="fixed inset-y-0 right-0 w-full max-w-2xl bg-slate-50 shadow-2xl z-50 transform transition-transform duration-300 ease-in-out border-l border-slate-200 flex flex-col">
        {/* Header */}
        <div className="p-6 border-bottom border-slate-200 bg-white flex items-center justify-between shadow-sm z-10">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-50 rounded-lg">
              <FileText className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-slate-900">Action: Draft PRD</h2>
              <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">AI Generated Strategy</p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="p-2 hover:bg-slate-100 rounded-full transition-colors"
          >
            <X className="w-6 h-6 text-slate-400" />
          </button>
        </div>

        {/* Content Area (Scrollable) */}
        <div className="flex-1 overflow-y-auto p-8 bg-slate-50">
          
          {isLoading && (
            <div className="flex flex-col items-center justify-center h-full space-y-4 text-slate-500">
              <Loader2 className="w-12 h-12 animate-spin text-blue-500" />
              <p className="text-sm font-bold uppercase tracking-widest text-blue-600 animate-pulse">
                AI Processing...
              </p>
              <p className="text-sm text-center max-w-xs leading-relaxed">
                Analyzing raw feedback and crafting a tailored Product Requirements Document.
              </p>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-8 text-center">
              <div className="text-red-500 font-bold mb-2 text-lg">Failed to Generate PRD</div>
              <div className="text-red-400 text-sm mb-4 leading-relaxed">{error}</div>
              <p className="text-slate-500 text-xs">
                Ensure the backend server is running via <code className="bg-white px-2 py-1 rounded border">npm run dev</code>
              </p>
            </div>
          )}

          {!isLoading && !error && prdContent && (
            <div className="bg-white shadow-xl shadow-slate-200/50 rounded-sm p-12 min-h-full border border-slate-100 font-serif">
              <div className="max-w-none text-slate-800">
                <ReactMarkdown
                  components={{
                    h1: ({node, ...props}) => <h1 className="text-4xl font-black text-slate-900 mb-8 pb-4 border-b-2 border-slate-900 tracking-tight leading-tight" {...props} />,
                    h2: ({node, ...props}) => <h2 className="text-2xl font-bold text-blue-700 mt-12 mb-6 flex items-center gap-3 before:content-[''] before:block before:w-1.5 before:h-8 before:bg-blue-600 before:rounded-full" {...props} />,
                    h3: ({node, ...props}) => <h3 className="text-lg font-black text-slate-800 mt-8 mb-4 uppercase tracking-wider" {...props} />,
                    p: ({node, ...props}) => <p className="text-lg text-slate-700 leading-relaxed mb-6" {...props} />,
                    ul: ({node, ...props}) => <ul className="space-y-3 mb-8" {...props} />,
                    ol: ({node, ...props}) => <ol className="list-decimal list-inside space-y-3 mb-8 text-lg text-slate-700 marker:font-bold marker:text-blue-600" {...props} />,
                    li: ({node, ...props}) => {
                      // Only add custom bullet point if it's unordered list (ul wrapper)
                      // checking if parent is ol can be tricky in simple mapping, so we just use standard margin
                      return (
                        <li className="flex items-start gap-3 text-lg text-slate-700 leading-relaxed">
                          <span className="mt-2.5 w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0" />
                          <span {...props} />
                        </li>
                      );
                    },
                    strong: ({node, ...props}) => <strong className="font-black text-slate-900 bg-slate-100 px-1 rounded block sm:inline mt-2 sm:mt-0" {...props} />,
                    blockquote: ({node, ...props}) => <blockquote className="border-l-4 border-emerald-500 bg-emerald-50 p-6 rounded-r-xl text-emerald-900 italic my-8 shadow-sm" {...props} />,
                    code: ({node, inline, ...props}) => inline ? (
                      <code className="bg-slate-100 text-pink-600 rounded-md px-1.5 py-0.5 text-sm font-mono border border-slate-200" {...props} />
                    ) : (
                      <div className="bg-slate-900 rounded-xl overflow-hidden my-8 shadow-lg">
                        <div className="bg-slate-800 px-4 py-2 text-xs text-slate-400 font-mono border-b border-slate-700 flex items-center justify-between">
                           <span>Code Snippet</span>
                           <div className="flex gap-1.5">
                             <div className="w-2.5 h-2.5 rounded-full bg-red-400"></div>
                             <div className="w-2.5 h-2.5 rounded-full bg-amber-400"></div>
                             <div className="w-2.5 h-2.5 rounded-full bg-emerald-400"></div>
                           </div>
                        </div>
                        <pre className="p-6 overflow-x-auto">
                          <code className="text-blue-300 font-mono text-sm leading-relaxed" {...props} />
                        </pre>
                      </div>
                    ),
                  }}
                >
                  {prdContent}
                </ReactMarkdown>
              </div>

              {/* Signature Area */}
              <div className="mt-16 pt-10 border-t border-slate-100 font-sans italic text-slate-400 text-sm flex justify-between">
                <span>Prepared by Lens Explorer Intelligence</span>
                <span>Proprietary & Confidential</span>
              </div>
            </div>
          )}
        </div>

        {/* Action Bar */}
        <div className="p-6 bg-white border-t border-slate-200 flex gap-4">
          <button 
            disabled
            className="flex-1 flex items-center justify-center gap-2 bg-slate-900 text-white font-bold py-3 rounded-xl opacity-50 cursor-not-allowed"
          >
            <Download className="w-5 h-5" />
            Export to Jira
          </button>
          <button 
            onClick={handleCopy}
            disabled={isLoading || error || !prdContent}
            className="flex-1 flex items-center justify-center gap-2 bg-white border-2 border-slate-200 text-slate-700 font-bold py-3 rounded-xl hover:bg-slate-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {copied ? (
              <>
                <Check className="w-5 h-5 text-emerald-500" />
                Copied!
              </>
            ) : (
              <>
                <Copy className="w-5 h-5" />
                Copy PRD
              </>
            )}
          </button>
        </div>
      </div>
    </>
  );
};

export default ActionPanel;
