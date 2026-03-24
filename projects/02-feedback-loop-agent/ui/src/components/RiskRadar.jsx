import { AlertTriangle } from 'lucide-react';

const RiskRadar = ({ risks = [] }) => {
  if (!risks || risks.length === 0) return null;

  return (
    <section className="mt-10">
      <h2 className="text-2xl font-bold mb-6 bg-gradient-to-r from-red-600 to-orange-500 bg-clip-text text-transparent">
        🔥 Strategic Risk Radar
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {risks.map((risk, index) => (
          <div 
            key={index} 
            className="flex items-start gap-4 p-5 bg-red-50/50 border border-dashed border-red-200 rounded-xl"
          >
            <div className="mt-1">
              <AlertTriangle className="w-5 h-5 text-red-600" />
            </div>
            <p className="text-red-900 font-medium leading-relaxed">
              {risk}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
};

export default RiskRadar;
