import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';
import { 
  Brain, 
  TrendingUp, 
  Users, 
  AlertTriangle, 
  BarChart3, 
  Target,
  Clock,
  CheckCircle,
  XCircle
} from 'lucide-react';

const MLAnalysisDashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [insights, setInsights] = useState(null);
  const [topics, setTopics] = useState([]);
  const [trends, setTrends] = useState(null);
  const [entities, setEntities] = useState([]);
  const [anomalies, setAnomalies] = useState([]);

  useEffect(() => {
    loadMLInsights();
  }, []);

  const loadMLInsights = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch('/api/ml/insights/summary', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!response.ok) {
        throw new Error('Failed to load ML insights');
      }
      
      const data = await response.json();
      setInsights(data);
      
      // Load additional data for each tab
      if (activeTab === 'topics') {
        await loadTopics();
      } else if (activeTab === 'trends') {
        await loadTrends();
      } else if (activeTab === 'entities') {
        await loadEntities();
      } else if (activeTab === 'anomalies') {
        await loadAnomalies();
      }
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadTopics = async () => {
    try {
      const response = await fetch('/api/ml/topics', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          num_topics: 10,
          method: 'lda'
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setTopics(data.topics || []);
      }
    } catch (err) {
      console.error('Failed to load topics:', err);
    }
  };

  const loadTrends = async () => {
    try {
      const response = await fetch('/api/ml/trends', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          period: 'daily',
          value_type: 'count'
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setTrends(data);
      }
    } catch (err) {
      console.error('Failed to load trends:', err);
    }
  };

  const loadEntities = async () => {
    try {
      // This would typically analyze a sample of recent articles
      const response = await fetch('/api/ml/entities', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          text: "Sample text for entity extraction analysis",
          include_custom: true
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setEntities(data.entities || []);
      }
    } catch (err) {
      console.error('Failed to load entities:', err);
    }
  };

  const loadAnomalies = async () => {
    try {
      const response = await fetch('/api/ml/anomalies', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          contamination: 0.1
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        setAnomalies(data.anomalies || []);
      }
    } catch (err) {
      console.error('Failed to load anomalies:', err);
    }
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    if (tab === 'topics' && topics.length === 0) {
      loadTopics();
    } else if (tab === 'trends' && !trends) {
      loadTrends();
    } else if (tab === 'entities' && entities.length === 0) {
      loadEntities();
    } else if (tab === 'anomalies' && anomalies.length === 0) {
      loadAnomalies();
    }
  };

  const getTrendIcon = (direction) => {
    switch (direction) {
      case 'increasing':
        return <TrendingUp className="h-4 w-4 text-green-500" />;
      case 'decreasing':
        return <TrendingUp className="h-4 w-4 text-red-500 rotate-180" />;
      default:
        return <BarChart3 className="h-4 w-4 text-gray-500" />;
    }
  };

  const getTrendColor = (direction) => {
    switch (direction) {
      case 'increasing':
        return 'text-green-600';
      case 'decreasing':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  if (loading && !insights) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">ML Analysis Dashboard</h2>
        <Button onClick={loadMLInsights} disabled={loading}>
          {loading ? 'Refreshing...' : 'Refresh'}
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <XCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Tabs value={activeTab} onValueChange={handleTabChange} className="space-y-4">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="topics">Topics</TabsTrigger>
          <TabsTrigger value="trends">Trends</TabsTrigger>
          <TabsTrigger value="entities">Entities</TabsTrigger>
          <TabsTrigger value="anomalies">Anomalies</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          {insights && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Articles</CardTitle>
                  <Brain className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{insights.total_articles || 0}</div>
                  <p className="text-xs text-muted-foreground">
                    Available for analysis
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Active Topics</CardTitle>
                  <Target className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{insights.topics?.length || 0}</div>
                  <p className="text-xs text-muted-foreground">
                    Identified themes
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Trend Direction</CardTitle>
                  {getTrendIcon(insights.trend?.direction)}
                </CardHeader>
                <CardContent>
                  <div className={`text-2xl font-bold ${getTrendColor(insights.trend?.direction)}`}>
                    {insights.trend?.direction || 'Unknown'}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Change rate: {((insights.trend?.change_rate || 0) * 100).toFixed(1)}%
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Entities Found</CardTitle>
                  <Users className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{insights.entities?.total || 0}</div>
                  <p className="text-xs text-muted-foreground">
                    {insights.entities?.types?.length || 0} types
                  </p>
                </CardContent>
              </Card>
            </div>
          )}

          {insights?.topics && (
            <Card>
              <CardHeader>
                <CardTitle>Top Topics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {insights.topics.slice(0, 5).map((topic, index) => (
                    <div key={index} className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Badge variant="secondary">#{topic.id}</Badge>
                        <span className="text-sm">{topic.keywords.join(', ')}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Progress value={topic.strength * 100} className="w-20" />
                        <span className="text-xs text-muted-foreground">
                          {(topic.strength * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="topics" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Topic Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="flex items-center justify-center h-32">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                </div>
              ) : (
                <div className="space-y-4">
                  {topics.map((topic, index) => (
                    <div key={index} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-semibold">Topic {topic.topic_id}</h3>
                        <Badge variant="outline">
                          {topic.document_percentage.toFixed(1)}% of docs
                        </Badge>
                      </div>
                      <div className="flex flex-wrap gap-1 mb-2">
                        {topic.keywords.map((keyword, idx) => (
                          <Badge key={idx} variant="secondary" className="text-xs">
                            {keyword}
                          </Badge>
                        ))}
                      </div>
                      <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                        <span>Weight: {topic.weight.toFixed(3)}</span>
                        <span>Coherence: {topic.coherence_score.toFixed(3)}</span>
                        <span>Documents: {topic.document_count}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="trends" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Trend Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="flex items-center justify-center h-32">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                </div>
              ) : trends ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold flex items-center justify-center space-x-2">
                        {getTrendIcon(trends.trend_direction)}
                        <span className={getTrendColor(trends.trend_direction)}>
                          {trends.trend_direction}
                        </span>
                      </div>
                      <p className="text-sm text-muted-foreground">Direction</p>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold">
                        {(trends.trend_strength * 100).toFixed(1)}%
                      </div>
                      <p className="text-sm text-muted-foreground">Strength</p>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>Change Rate:</span>
                      <span className="font-medium">
                        {(trends.change_rate * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Significance:</span>
                      <span className="font-medium">
                        {(trends.significance * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Data Points:</span>
                      <span className="font-medium">{trends.data_points}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Analysis Time:</span>
                      <span className="font-medium">{trends.analysis_time.toFixed(2)}s</span>
                    </div>
                  </div>

                  {trends.forecast && trends.forecast.length > 0 && (
                    <div>
                      <h4 className="font-semibold mb-2">7-Day Forecast</h4>
                      <div className="flex space-x-2">
                        {trends.forecast.map((value, index) => (
                          <div key={index} className="text-center">
                            <div className="text-sm font-medium">Day {index + 1}</div>
                            <div className="text-xs text-muted-foreground">
                              {value.toFixed(0)}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-muted-foreground">No trend data available</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="entities" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Entity Extraction</CardTitle>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="flex items-center justify-center h-32">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                </div>
              ) : (
                <div className="space-y-4">
                  {entities.map((entity, index) => (
                    <div key={index} className="flex items-center justify-between p-2 border rounded">
                      <div className="flex items-center space-x-2">
                        <Badge variant="outline">{entity.label}</Badge>
                        <span className="font-medium">{entity.text}</span>
                        <span className="text-sm text-muted-foreground">
                          {entity.description}
                        </span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Progress value={entity.confidence * 100} className="w-16" />
                        <span className="text-xs text-muted-foreground">
                          {(entity.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="anomalies" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Anomaly Detection</CardTitle>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="flex items-center justify-center h-32">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                </div>
              ) : (
                <div className="space-y-4">
                  {anomalies.length > 0 ? (
                    anomalies.map((anomaly, index) => (
                      <Alert key={index} variant="destructive">
                        <AlertTriangle className="h-4 w-4" />
                        <AlertDescription>
                          <div className="flex items-center justify-between">
                            <span>Anomaly detected on {anomaly.created_at}</span>
                            <Badge variant="destructive">
                              Score: {anomaly.anomaly_score?.toFixed(3) || 'N/A'}
                            </Badge>
                          </div>
                        </AlertDescription>
                      </Alert>
                    ))
                  ) : (
                    <div className="text-center py-8">
                      <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-2" />
                      <p className="text-muted-foreground">No anomalies detected</p>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default MLAnalysisDashboard;
