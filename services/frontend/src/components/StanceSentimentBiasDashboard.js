import React, { useState, useEffect } from 'react';
import { Card } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Tabs } from './ui/tabs';
import { Alert } from './ui/alert';
import { Progress } from './ui/progress';

const StanceSentimentBiasDashboard = () => {
  const [dashboardData, setDashboardData] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    fetchDashboardData();
    fetchAlerts();
  }, []);

  const fetchDashboardData = async () => {
    try {
      const response = await fetch('/api/analysis/dashboard?days=7', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch dashboard data');
      }
      
      const data = await response.json();
      setDashboardData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchAlerts = async () => {
    try {
      const response = await fetch('/api/analysis/alerts?hours=24', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch alerts');
      }
      
      const data = await response.json();
      setAlerts(data.alerts || []);
    } catch (err) {
      console.error('Failed to fetch alerts:', err);
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high': return 'bg-red-500';
      case 'medium': return 'bg-yellow-500';
      case 'low': return 'bg-green-500';
      default: return 'bg-gray-500';
    }
  };

  const getAlertTypeIcon = (type) => {
    switch (type) {
      case 'sentiment_shift': return '📊';
      case 'stance_change': return '🔄';
      case 'toxicity_spike': return '⚠️';
      case 'extreme_bias': return '⚖️';
      default: return '🔔';
    }
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg">Loading analysis dashboard...</div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert className="bg-red-50 border-red-200 text-red-800">
        <div className="font-semibold">Error loading dashboard</div>
        <div>{error}</div>
      </Alert>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Analysis Dashboard</h2>
        <div className="flex space-x-2">
          <Button onClick={fetchDashboardData} variant="outline">
            Refresh
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <div className="flex space-x-1 border-b border-gray-200">
          <button
            className={`px-4 py-2 text-sm font-medium ${
              activeTab === 'overview' 
                ? 'text-blue-600 border-b-2 border-blue-600' 
                : 'text-gray-500 hover:text-gray-700'
            }`}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </button>
          <button
            className={`px-4 py-2 text-sm font-medium ${
              activeTab === 'alerts' 
                ? 'text-blue-600 border-b-2 border-blue-600' 
                : 'text-gray-500 hover:text-gray-700'
            }`}
            onClick={() => setActiveTab('alerts')}
          >
            Alerts ({alerts.length})
          </button>
          <button
            className={`px-4 py-2 text-sm font-medium ${
              activeTab === 'sources' 
                ? 'text-blue-600 border-b-2 border-blue-600' 
                : 'text-gray-500 hover:text-gray-700'
            }`}
            onClick={() => setActiveTab('sources')}
          >
            Sources
          </button>
          <button
            className={`px-4 py-2 text-sm font-medium ${
              activeTab === 'topics' 
                ? 'text-blue-600 border-b-2 border-blue-600' 
                : 'text-gray-500 hover:text-gray-700'
            }`}
            onClick={() => setActiveTab('topics')}
          >
            Topics
          </button>
        </div>

        {activeTab === 'overview' && (
          <div className="space-y-6 mt-6">
            {/* Statistics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <Card className="p-6">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                      <span className="text-blue-600 text-sm font-semibold">📄</span>
                    </div>
                  </div>
                  <div className="ml-4">
                    <div className="text-sm font-medium text-gray-500">Articles Analyzed</div>
                    <div className="text-2xl font-bold text-gray-900">
                      {dashboardData?.statistics?.total_articles_analyzed || 0}
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-6">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                      <span className="text-green-600 text-sm font-semibold">🎯</span>
                    </div>
                  </div>
                  <div className="ml-4">
                    <div className="text-sm font-medium text-gray-500">Avg Confidence</div>
                    <div className="text-2xl font-bold text-gray-900">
                      {formatPercentage(dashboardData?.statistics?.avg_confidence || 0)}
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-6">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 bg-yellow-100 rounded-full flex items-center justify-center">
                      <span className="text-yellow-600 text-sm font-semibold">⚠️</span>
                    </div>
                  </div>
                  <div className="ml-4">
                    <div className="text-sm font-medium text-gray-500">Risk Articles</div>
                    <div className="text-2xl font-bold text-gray-900">
                      {dashboardData?.statistics?.articles_with_risks || 0}
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-6">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center">
                      <span className="text-purple-600 text-sm font-semibold">📊</span>
                    </div>
                  </div>
                  <div className="ml-4">
                    <div className="text-sm font-medium text-gray-500">Sources Analyzed</div>
                    <div className="text-2xl font-bold text-gray-900">
                      {dashboardData?.statistics?.sources_analyzed || 0}
                    </div>
                  </div>
                </div>
              </Card>
            </div>

            {/* Trend Analysis */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Sentiment Trends</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Positive</span>
                    <div className="flex items-center space-x-2">
                      <Progress 
                        value={dashboardData?.statistics?.positive_trends || 0} 
                        className="w-24"
                      />
                      <span className="text-sm font-medium">
                        {dashboardData?.statistics?.positive_trends || 0}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Negative</span>
                    <div className="flex items-center space-x-2">
                      <Progress 
                        value={dashboardData?.statistics?.negative_trends || 0} 
                        className="w-24"
                      />
                      <span className="text-sm font-medium">
                        {dashboardData?.statistics?.negative_trends || 0}
                      </span>
                    </div>
                  </div>
                </div>
              </Card>

              <Card className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Stance Trends</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Supportive</span>
                    <div className="flex items-center space-x-2">
                      <Progress 
                        value={dashboardData?.statistics?.supportive_trends || 0} 
                        className="w-24"
                      />
                      <span className="text-sm font-medium">
                        {dashboardData?.statistics?.supportive_trends || 0}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Refutational</span>
                    <div className="flex items-center space-x-2">
                      <Progress 
                        value={dashboardData?.statistics?.refutational_trends || 0} 
                        className="w-24"
                      />
                      <span className="text-sm font-medium">
                        {dashboardData?.statistics?.refutational_trends || 0}
                      </span>
                    </div>
                  </div>
                </div>
              </Card>
            </div>
          </div>
        )}

        {activeTab === 'alerts' && (
          <div className="space-y-4 mt-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900">Recent Alerts</h3>
              <Button onClick={fetchAlerts} variant="outline" size="sm">
                Refresh
              </Button>
            </div>

            {alerts.length === 0 ? (
              <Card className="p-6 text-center">
                <div className="text-gray-500">No recent alerts</div>
              </Card>
            ) : (
              <div className="space-y-3">
                {alerts.map((alert) => (
                  <Card key={alert.id} className="p-4">
                    <div className="flex items-start space-x-3">
                      <div className="flex-shrink-0">
                        <span className="text-2xl">{getAlertTypeIcon(alert.alert_type)}</span>
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2">
                          <Badge className={getSeverityColor(alert.severity)}>
                            {alert.severity}
                          </Badge>
                          <span className="text-sm font-medium text-gray-900">
                            {alert.alert_type.replace('_', ' ').toUpperCase()}
                          </span>
                          <span className="text-sm text-gray-500">
                            {new Date(alert.created_at).toLocaleString()}
                          </span>
                        </div>
                        <div className="mt-1 text-sm text-gray-600">
                          {alert.message}
                        </div>
                        {alert.source_id && (
                          <div className="mt-1 text-xs text-gray-500">
                            Source: {alert.source_id}
                          </div>
                        )}
                        {alert.topic && (
                          <div className="mt-1 text-xs text-gray-500">
                            Topic: {alert.topic}
                          </div>
                        )}
                      </div>
                      <div className="flex-shrink-0">
                        {!alert.is_acknowledged && (
                          <Button size="sm" variant="outline">
                            Acknowledge
                          </Button>
                        )}
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'sources' && (
          <div className="space-y-4 mt-6">
            <h3 className="text-lg font-semibold text-gray-900">Top Sources by Analysis Count</h3>
            {dashboardData?.top_sources?.length > 0 ? (
              <div className="space-y-2">
                {dashboardData.top_sources.map((source, index) => (
                  <Card key={source.source_id} className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                          <span className="text-blue-600 text-sm font-semibold">
                            {index + 1}
                          </span>
                        </div>
                        <div>
                          <div className="font-medium text-gray-900">{source.source_id}</div>
                          <div className="text-sm text-gray-500">
                            {source.analysis_count} analyses
                          </div>
                        </div>
                      </div>
                      <Button variant="outline" size="sm">
                        View Details
                      </Button>
                    </div>
                  </Card>
                ))}
              </div>
            ) : (
              <Card className="p-6 text-center">
                <div className="text-gray-500">No source data available</div>
              </Card>
            )}
          </div>
        )}

        {activeTab === 'topics' && (
          <div className="space-y-4 mt-6">
            <h3 className="text-lg font-semibold text-gray-900">Top Topics by Analysis Count</h3>
            {dashboardData?.top_topics?.length > 0 ? (
              <div className="space-y-2">
                {dashboardData.top_topics.map((topic, index) => (
                  <Card key={topic.topic} className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                          <span className="text-green-600 text-sm font-semibold">
                            {index + 1}
                          </span>
                        </div>
                        <div>
                          <div className="font-medium text-gray-900">{topic.topic}</div>
                          <div className="text-sm text-gray-500">
                            {topic.analysis_count} analyses
                          </div>
                        </div>
                      </div>
                      <Button variant="outline" size="sm">
                        View Details
                      </Button>
                    </div>
                  </Card>
                ))}
              </div>
            ) : (
              <Card className="p-6 text-center">
                <div className="text-gray-500">No topic data available</div>
              </Card>
            )}
          </div>
        )}
      </Tabs>
    </div>
  );
};

export default StanceSentimentBiasDashboard;
