import React, { useState, useEffect } from 'react';
import { 
  AlertTriangle, Bell, Settings, Filter, Search, RefreshCw, 
  CheckCircle, XCircle, Clock, Eye, EyeOff, Trash2, Archive,
  AlertCircle, Info, Zap, Shield, Database, Server
} from 'lucide-react';

const Alerts = () => {
  const [alerts, setAlerts] = useState([]);
  const [filteredAlerts, setFilteredAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [filters, setFilters] = useState({
    severity: 'all',
    status: 'all',
    type: 'all'
  });
  const [showFilters, setShowFilters] = useState(false);

  // Mock alerts data - in production this would come from API
  useEffect(() => {
    const mockAlerts = [
      {
        id: 1,
        title: 'High Memory Usage Detected',
        description: 'System memory usage has exceeded 85% threshold for the past 10 minutes',
        severity: 'warning',
        type: 'system',
        status: 'active',
        timestamp: new Date(Date.now() - 30 * 60 * 1000),
        source: 'Memory Monitor',
        acknowledged: false,
        acknowledgedBy: null,
        icon: AlertTriangle
      },
      {
        id: 2,
        title: 'Database Connection Pool Full',
        description: 'All database connection slots are in use. New requests may be delayed.',
        severity: 'critical',
        type: 'database',
        status: 'active',
        timestamp: new Date(Date.now() - 15 * 60 * 1000),
        source: 'Database Monitor',
        acknowledged: false,
        acknowledgedBy: null,
        icon: Database
      },
      {
        id: 3,
        title: 'New Articles Processed',
        description: '150 new articles have been successfully processed and indexed',
        severity: 'info',
        type: 'processing',
        status: 'resolved',
        timestamp: new Date(Date.now() - 45 * 60 * 1000),
        source: 'Content Processor',
        acknowledged: true,
        acknowledgedBy: 'admin',
        icon: Info
      },
      {
        id: 4,
        title: 'API Rate Limit Exceeded',
        description: 'External API rate limit reached. Some requests are being throttled.',
        severity: 'warning',
        type: 'api',
        status: 'active',
        timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
        source: 'API Gateway',
        acknowledged: true,
        acknowledgedBy: 'admin',
        icon: Zap
      },
      {
        id: 5,
        title: 'Security Scan Completed',
        description: 'Automated security scan completed successfully. No vulnerabilities detected.',
        severity: 'info',
        type: 'security',
        status: 'resolved',
        timestamp: new Date(Date.now() - 4 * 60 * 60 * 1000),
        source: 'Security Scanner',
        acknowledged: true,
        acknowledgedBy: 'admin',
        icon: Shield
      }
    ];

    setTimeout(() => {
      setAlerts(mockAlerts);
      setFilteredAlerts(mockAlerts);
      setLoading(false);
    }, 1000);
  }, []);

  // Filter alerts based on search and filters
  useEffect(() => {
    let filtered = alerts;

    if (searchTerm) {
      filtered = filtered.filter(alert => 
        alert.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        alert.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        alert.source.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (filters.severity !== 'all') {
      filtered = filtered.filter(alert => alert.severity === filters.severity);
    }

    if (filters.status !== 'all') {
      filtered = filtered.filter(alert => alert.status === filters.status);
    }

    if (filters.type !== 'all') {
      filtered = filtered.filter(alert => alert.type === filters.type);
    }

    setFilteredAlerts(filtered);
  }, [alerts, searchTerm, filters]);

  const handleAcknowledge = (alertId) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId 
        ? { ...alert, acknowledged: true, acknowledgedBy: 'admin' }
        : alert
    ));
  };

  const handleResolve = (alertId) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId 
        ? { ...alert, status: 'resolved' }
        : alert
    ));
  };

  const handleDelete = (alertId) => {
    setAlerts(prev => prev.filter(alert => alert.id !== alertId));
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'critical':
        return { bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-200', icon: 'text-red-600' };
      case 'warning':
        return { bg: 'bg-yellow-50', text: 'text-yellow-700', border: 'border-yellow-200', icon: 'text-yellow-600' };
      case 'info':
        return { bg: 'bg-blue-50', text: 'text-blue-700', border: 'border-blue-200', icon: 'text-blue-600' };
      default:
        return { bg: 'bg-gray-50', text: 'text-gray-700', border: 'border-gray-200', icon: 'text-gray-600' };
    }
  };

  const formatTimeAgo = (timestamp) => {
    const now = new Date();
    const diffInMinutes = Math.floor((now - timestamp) / (1000 * 60));
    
    if (diffInMinutes < 60) {
      return `${diffInMinutes}m ago`;
    }
    
    const diffInHours = Math.floor(diffInMinutes / 60);
    if (diffInHours < 24) {
      return `${diffInHours}h ago`;
    }
    
    const diffInDays = Math.floor(diffInHours / 24);
    return `${diffInDays}d ago`;
  };

  const activeAlerts = alerts.filter(alert => alert.status === 'active');
  const unacknowledgedAlerts = alerts.filter(alert => !alert.acknowledged && alert.status === 'active');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Alerts & Notifications</h1>
          <p className="text-gray-600">
            {activeAlerts.length} active alerts • {unacknowledgedAlerts.length} unacknowledged
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <button 
            onClick={() => setShowFilters(!showFilters)}
            className="flex items-center px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200"
          >
            <Filter className="w-4 h-4 mr-2" />
            Filters
          </button>
          <button className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
            <Settings className="w-4 h-4 mr-2" />
            Configure
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
          <div className="flex items-center">
            <div className="p-2 bg-red-50 rounded-lg">
              <AlertTriangle className="w-6 h-6 text-red-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Critical</p>
              <p className="text-2xl font-bold text-gray-900">
                {alerts.filter(a => a.severity === 'critical' && a.status === 'active').length}
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
          <div className="flex items-center">
            <div className="p-2 bg-yellow-50 rounded-lg">
              <AlertCircle className="w-6 h-6 text-yellow-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Warning</p>
              <p className="text-2xl font-bold text-gray-900">
                {alerts.filter(a => a.severity === 'warning' && a.status === 'active').length}
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
          <div className="flex items-center">
            <div className="p-2 bg-blue-50 rounded-lg">
              <Info className="w-6 h-6 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Info</p>
              <p className="text-2xl font-bold text-gray-900">
                {alerts.filter(a => a.severity === 'info').length}
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
          <div className="flex items-center">
            <div className="p-2 bg-green-50 rounded-lg">
              <CheckCircle className="w-6 h-6 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600">Resolved</p>
              <p className="text-2xl font-bold text-gray-900">
                {alerts.filter(a => a.status === 'resolved').length}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
        <div className="flex flex-col space-y-4">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-3 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search alerts..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          {/* Filters */}
          {showFilters && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <select
                value={filters.severity}
                onChange={(e) => setFilters(prev => ({ ...prev, severity: e.target.value }))}
                className="px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="all">All Severities</option>
                <option value="critical">Critical</option>
                <option value="warning">Warning</option>
                <option value="info">Info</option>
              </select>

              <select
                value={filters.status}
                onChange={(e) => setFilters(prev => ({ ...prev, status: e.target.value }))}
                className="px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="all">All Statuses</option>
                <option value="active">Active</option>
                <option value="resolved">Resolved</option>
              </select>

              <select
                value={filters.type}
                onChange={(e) => setFilters(prev => ({ ...prev, type: e.target.value }))}
                className="px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="all">All Types</option>
                <option value="system">System</option>
                <option value="database">Database</option>
                <option value="api">API</option>
                <option value="security">Security</option>
                <option value="processing">Processing</option>
              </select>
            </div>
          )}
        </div>
      </div>

      {/* Alerts List */}
      <div className="space-y-4">
        {loading ? (
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-8">
            <div className="text-center">
              <RefreshCw className="w-8 h-8 text-gray-400 animate-spin mx-auto mb-4" />
              <p className="text-gray-600">Loading alerts...</p>
            </div>
          </div>
        ) : filteredAlerts.length === 0 ? (
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-8">
            <div className="text-center">
              <Bell className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No alerts found</h3>
              <p className="text-gray-600">
                {searchTerm || filters.severity !== 'all' || filters.status !== 'all' || filters.type !== 'all'
                  ? 'Try adjusting your search or filters'
                  : 'All systems are operating normally'
                }
              </p>
            </div>
          </div>
        ) : (
          filteredAlerts.map((alert) => {
            const severity = getSeverityColor(alert.severity);
            const Icon = alert.icon;
            
            return (
              <div
                key={alert.id}
                className={`bg-white rounded-xl shadow-sm border ${severity.border} p-6 ${severity.bg}`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-4">
                    <div className={`p-2 rounded-lg ${severity.bg}`}>
                      <Icon className={`w-6 h-6 ${severity.icon}`} />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <h3 className="text-lg font-semibold text-gray-900">{alert.title}</h3>
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${severity.text} ${severity.bg}`}>
                          {alert.severity}
                        </span>
                        {alert.acknowledged && (
                          <span className="px-2 py-1 text-xs font-medium bg-green-100 text-green-700 rounded-full">
                            Acknowledged
                          </span>
                        )}
                      </div>
                      <p className="text-gray-600 mb-2">{alert.description}</p>
                      <div className="flex items-center space-x-4 text-sm text-gray-500">
                        <span className="flex items-center">
                          <Clock className="w-4 h-4 mr-1" />
                          {formatTimeAgo(alert.timestamp)}
                        </span>
                        <span>Source: {alert.source}</span>
                        {alert.acknowledgedBy && (
                          <span>Acknowledged by: {alert.acknowledgedBy}</span>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    {!alert.acknowledged && alert.status === 'active' && (
                      <button
                        onClick={() => handleAcknowledge(alert.id)}
                        className="flex items-center px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200"
                      >
                        <Eye className="w-4 h-4 mr-1" />
                        Acknowledge
                      </button>
                    )}
                    {alert.status === 'active' && (
                      <button
                        onClick={() => handleResolve(alert.id)}
                        className="flex items-center px-3 py-1 text-sm bg-green-100 text-green-700 rounded-lg hover:bg-green-200"
                      >
                        <CheckCircle className="w-4 h-4 mr-1" />
                        Resolve
                      </button>
                    )}
                    <button
                      onClick={() => handleDelete(alert.id)}
                      className="flex items-center px-3 py-1 text-sm bg-red-100 text-red-700 rounded-lg hover:bg-red-200"
                    >
                      <Trash2 className="w-4 h-4 mr-1" />
                      Delete
                    </button>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default Alerts;
