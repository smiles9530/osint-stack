import React from 'react';
import { Zap, Play, Pause, Settings } from 'lucide-react';

const Workflows = () => {
  const workflows = [
    {
      id: 1,
      name: 'Enhanced OSINT Processing (Integrated)',
      status: 'active',
      description: 'RSS feed monitoring and content processing',
      lastRun: '2024-09-15T10:30:00Z'
    },
    {
      id: 2,
      name: 'OSINT Monitoring & Analytics (Integrated)',
      status: 'active',
      description: 'System health monitoring and analytics',
      lastRun: '2024-09-15T10:25:00Z'
    },
    {
      id: 3,
      name: 'OSINT Frontend Integration Webhook',
      status: 'active',
      description: 'Frontend-backend data integration',
      lastRun: '2024-09-15T10:20:00Z'
    },
    {
      id: 4,
      name: 'Real-time WebSocket Data Publisher',
      status: 'active',
      description: 'Real-time data broadcasting',
      lastRun: '2024-09-15T10:15:00Z'
    }
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">N8N Workflows</h1>
        <p className="text-gray-600">Manage and monitor automated workflows</p>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Active Workflows</h3>
        </div>
        
        <div className="divide-y divide-gray-200">
          {workflows.map((workflow) => (
            <div key={workflow.id} className="p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Zap className="w-8 h-8 text-blue-600 mr-4" />
                  <div>
                    <h4 className="text-lg font-medium text-gray-900">{workflow.name}</h4>
                    <p className="text-gray-600">{workflow.description}</p>
                    <p className="text-sm text-gray-500">
                      Last run: {new Date(workflow.lastRun).toLocaleString()}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                    workflow.status === 'active' ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-700'
                  }`}>
                    {workflow.status}
                  </span>
                  <button className="p-2 text-gray-400 hover:text-gray-600">
                    {workflow.status === 'active' ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  </button>
                  <button className="p-2 text-gray-400 hover:text-gray-600">
                    <Settings className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Workflows;
