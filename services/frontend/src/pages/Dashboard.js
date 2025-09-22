import React from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  LineChart, Line, PieChart, Pie, Cell, AreaChart, Area
} from 'recharts';
import { 
  Activity, 
  TrendingUp, 
  Users, 
  FileText, 
  Globe, 
  AlertTriangle,
  Clock,
  Zap,
  Database,
  Wifi
} from 'lucide-react';
import { useApp } from '../contexts/AppContext';

const Dashboard = () => {
  const { dashboardData, loading, wsConnected } = useApp();

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <Activity className="w-8 h-8 animate-spin mx-auto mb-4 text-blue-600" />
          <p className="text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  const { health, articles, alerts, recent_activity } = dashboardData;

  // Prepare chart data
  const languageData = Object.entries(articles.languages || {}).map(([lang, count]) => ({
    language: lang.toUpperCase(),
    count,
    percentage: ((count / articles.total_articles) * 100).toFixed(1)
  }));

  const sentimentData = Object.entries(articles.sentiment_distribution || {}).map(([sentiment, value]) => ({
    name: sentiment.charAt(0).toUpperCase() + sentiment.slice(1),
    value: Math.round(value * 100),
    color: sentiment === 'positive' ? '#10B981' : sentiment === 'negative' ? '#EF4444' : '#6B7280'
  }));

  // Mock time series data for trends
  const trendData = Array.from({ length: 24 }, (_, i) => ({
    hour: `${i}:00`,
    articles: Math.floor(Math.random() * 50) + 10,
    sentiment: Math.random() * 0.4 + 0.3
  }));

  const StatCard = ({ title, value, subtitle, icon: Icon, color = "blue", trend }) => (
    <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100 hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
          {subtitle && <p className="text-sm text-gray-500">{subtitle}</p>}
        </div>
        <div className={`p-3 bg-${color}-100 rounded-lg`}>
          <Icon className={`w-6 h-6 text-${color}-600`} />
        </div>
      </div>
      {trend && (
        <div className="mt-2 flex items-center">
          <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
          <span className="text-sm text-green-600">{trend}</span>
        </div>
      )}
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Dashboard Overview</h1>
          <p className="text-gray-600">Real-time OSINT analytics and monitoring</p>
        </div>
        <div className="flex items-center space-x-2">
          {wsConnected ? (
            <div className="flex items-center text-green-600 bg-green-50 px-3 py-1 rounded-full">
              <Wifi className="w-4 h-4 mr-1" />
              <span className="text-sm font-medium">Live Data</span>
            </div>
          ) : (
            <div className="flex items-center text-orange-600 bg-orange-50 px-3 py-1 rounded-full">
              <Clock className="w-4 h-4 mr-1" />
              <span className="text-sm font-medium">Cached Data</span>
            </div>
          )}
        </div>
      </div>

      {/* System Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="System Status"
          value={health.status}
          subtitle={`Uptime: ${Math.floor(health.uptime / 3600)}h`}
          icon={Activity}
          color="green"
        />
        <StatCard
          title="Total Articles"
          value={articles.total_articles.toLocaleString()}
          subtitle={`${articles.articles_today} today`}
          icon={FileText}
          color="blue"
          trend="+12% from yesterday"
        />
        <StatCard
          title="Active Sources"
          value={articles.sources_count}
          subtitle="RSS feeds monitored"
          icon={Globe}
          color="purple"
        />
        <StatCard
          title="Processing Queue"
          value="0"
          subtitle="No pending tasks"
          icon={Zap}
          color="orange"
        />
      </div>

      {/* Main Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Article Trends */}
        <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Article Processing Trends</h3>
            <div className="text-sm text-gray-500">Last 24 hours</div>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={trendData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" />
              <YAxis />
              <Tooltip />
              <Area type="monotone" dataKey="articles" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.1} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Sentiment Distribution */}
        <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Sentiment Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={sentimentData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }) => `${name}: ${value}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {sentimentData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Secondary Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Language Distribution */}
        <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Language Distribution</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={languageData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="language" />
              <YAxis />
              <Tooltip 
                formatter={(value, name) => [
                  name === 'count' ? `${value} articles` : `${value}%`, 
                  name === 'count' ? 'Articles' : 'Percentage'
                ]}
              />
              <Bar dataKey="count" fill="#8B5CF6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* System Health Metrics */}
        <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">System Health</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium text-gray-600">Database</span>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                health.database === 'healthy' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
              }`}>
                {health.database}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium text-gray-600">Cache</span>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                health.cache === 'healthy' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
              }`}>
                {health.cache}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium text-gray-600">Vector Database</span>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                health.vector_db === 'healthy' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
              }`}>
                {health.vector_db}
              </span>
            </div>
            <div className="pt-4 border-t border-gray-200">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-gray-600">Memory Usage</span>
                <span className="text-sm text-gray-900">75.5%</span>
              </div>
              <div className="mt-1 bg-gray-200 rounded-full h-2">
                <div className="bg-blue-600 h-2 rounded-full" style={{ width: '75.5%' }}></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Alerts and Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Alerts */}
        <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Alerts</h3>
          <div className="space-y-3">
            {alerts.length > 0 ? alerts.slice(0, 5).map((alert, index) => (
              <div key={index} className={`p-3 rounded-lg border-l-4 ${
                alert.severity === 'warning' ? 'bg-yellow-50 border-yellow-400' :
                alert.severity === 'error' ? 'bg-red-50 border-red-400' :
                'bg-blue-50 border-blue-400'
              }`}>
                <div className="flex items-center">
                  <AlertTriangle className={`w-4 h-4 mr-2 ${
                    alert.severity === 'warning' ? 'text-yellow-600' :
                    alert.severity === 'error' ? 'text-red-600' :
                    'text-blue-600'
                  }`} />
                  <span className="font-medium text-sm">{alert.message}</span>
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {new Date(alert.timestamp * 1000).toLocaleString()}
                </div>
              </div>
            )) : (
              <div className="text-center py-4">
                <AlertTriangle className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                <p className="text-gray-500 text-sm">No alerts at this time</p>
              </div>
            )}
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Activity</h3>
          <div className="space-y-3">
            {recent_activity.length > 0 ? recent_activity.slice(0, 5).map((activity, index) => (
              <div key={index} className="flex items-start space-x-3">
                <div className="w-2 h-2 bg-blue-600 rounded-full mt-2 flex-shrink-0"></div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-gray-900">{activity.description}</p>
                  <p className="text-xs text-gray-500">
                    {new Date(activity.timestamp * 1000).toLocaleString()}
                  </p>
                </div>
              </div>
            )) : (
              <div className="text-center py-4">
                <Clock className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                <p className="text-gray-500 text-sm">No recent activity</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
