import React, { useState, useEffect } from 'react';
import { 
  Activity, AlertTriangle, TrendingUp, Globe, 
  Clock, Shield, Zap, Brain, RefreshCw
} from 'lucide-react';

const IntelligenceDashboard = () => {
  const [intelligenceData, setIntelligenceData] = useState({
    alerts: [],
    categories: {
      security: 0,
      politics: 0,
      economics: 0,
      technology: 0
    },
    recentActivity: [],
    systemStatus: 'active'
  });
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  // Fetch intelligence data from API
  useEffect(() => {
    fetchIntelligenceData();
    const interval = setInterval(fetchIntelligenceData, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchIntelligenceData = async () => {
    try {
      const response = await fetch('/api/analytics/dashboard');
      const data = await response.json();
      setIntelligenceData(data);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Failed to fetch intelligence data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getCategoryIcon = (category) => {
    const icons = {
      security: Shield,
      politics: Globe,
      economics: TrendingUp,
      technology: Brain
    };
    return icons[category] || Activity;
  };

  const getPriorityColor = (priority) => {
    const colors = {
      high: 'text-red-600 bg-red-50 border-red-200',
      medium: 'text-yellow-600 bg-yellow-50 border-yellow-200',
      low: 'text-green-600 bg-green-50 border-green-200'
    };
    return colors[priority] || 'text-gray-600 bg-gray-50 border-gray-200';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin text-blue-600" />
        <span className="ml-2 text-lg">Loading Intelligence Dashboard...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">
          🌍 Global Intelligence Monitor
        </h2>
        <div className="flex items-center space-x-4">
          <div className="flex items-center text-sm text-gray-500">
            <Clock className="h-4 w-4 mr-1" />
            Last updated: {lastUpdate.toLocaleTimeString()}
          </div>
          <button
            onClick={fetchIntelligenceData}
            className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            <RefreshCw className="h-4 w-4 mr-1" />
            Refresh
          </button>
        </div>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {Object.entries(intelligenceData.categories).map(([category, count]) => {
          const IconComponent = getCategoryIcon(category);
          return (
            <div key={category} className="bg-white overflow-hidden shadow rounded-lg">
              <div className="p-5">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <IconComponent className="h-6 w-6 text-gray-400" />
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <dl>
                      <dt className="text-sm font-medium text-gray-500 truncate capitalize">
                        {category} Intelligence
                      </dt>
                      <dd className="text-lg font-medium text-gray-900">{count} articles</dd>
                    </dl>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Critical Alerts */}
      {intelligenceData.alerts && intelligenceData.alerts.length > 0 && (
        <div className="bg-white shadow rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <div className="flex items-center mb-4">
              <AlertTriangle className="h-5 w-5 text-red-500 mr-2" />
              <h3 className="text-lg leading-6 font-medium text-gray-900">
                Critical Intelligence Alerts
              </h3>
            </div>
            <div className="space-y-3">
              {intelligenceData.alerts.slice(0, 5).map((alert, index) => (
                <div
                  key={index}
                  className={`p-4 rounded-lg border ${getPriorityColor(alert.priority)}`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="text-sm font-medium">{alert.title}</h4>
                      <p className="text-xs mt-1 opacity-75">
                        Category: {alert.category} • Priority: {alert.priority}
                      </p>
                    </div>
                    <div className="text-xs opacity-75">
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                  {alert.source_url && (
                    <a
                      href={alert.source_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-blue-600 hover:text-blue-800 mt-2 inline-block"
                    >
                      View Source →
                    </a>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Recent Activity */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <div className="flex items-center mb-4">
            <Activity className="h-5 w-5 text-blue-500 mr-2" />
            <h3 className="text-lg leading-6 font-medium text-gray-900">
              Recent Intelligence Activity
            </h3>
          </div>
          <div className="flow-root">
            <ul className="-mb-8">
              {intelligenceData.recentActivity.slice(0, 10).map((activity, index) => (
                <li key={index}>
                  <div className="relative pb-8">
                    {index !== intelligenceData.recentActivity.length - 1 && (
                      <span
                        className="absolute top-4 left-4 -ml-px h-full w-0.5 bg-gray-200"
                        aria-hidden="true"
                      />
                    )}
                    <div className="relative flex space-x-3">
                      <div>
                        <span className="h-8 w-8 rounded-full bg-blue-500 flex items-center justify-center ring-8 ring-white">
                          <Zap className="h-4 w-4 text-white" />
                        </span>
                      </div>
                      <div className="min-w-0 flex-1 pt-1.5 flex justify-between space-x-4">
                        <div>
                          <p className="text-sm text-gray-500">
                            {activity.description || 'Intelligence processed'}
                          </p>
                          <p className="text-xs text-gray-400">
                            Type: {activity.type || 'article_processed'}
                          </p>
                        </div>
                        <div className="text-right text-sm whitespace-nowrap text-gray-500">
                          {new Date(activity.timestamp).toLocaleString()}
                        </div>
                      </div>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* System Status */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <div className={`h-3 w-3 rounded-full mr-3 ${
                intelligenceData.systemStatus === 'active' ? 'bg-green-400' :
                intelligenceData.systemStatus === 'warning' ? 'bg-yellow-400' : 'bg-red-400'
              }`} />
              <div>
                <h3 className="text-lg leading-6 font-medium text-gray-900">
                  Intelligence System Status
                </h3>
                <p className="text-sm text-gray-500">
                  N8N Workflow: {intelligenceData.systemStatus === 'active' ? 'Active' : 'Inactive'}
                </p>
              </div>
            </div>
            <div className="text-sm text-gray-500">
              <p>Monitoring: Reuters Global Feeds</p>
              <p>Frequency: Every 10 minutes</p>
              <p>Categories: Security, Politics, Economics, Technology</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default IntelligenceDashboard;
