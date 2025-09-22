import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useApp } from '../contexts/AppContext';
import { 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  Globe, 
  RefreshCw,
  BarChart3,
  Activity,
  XCircle
} from 'lucide-react';

const ComplianceMonitor = () => {
  const { isAuthenticated, authToken } = useAuth();
  const { } = useApp();
  const [complianceStats, setComplianceStats] = useState(null);
  const [domainStats, setDomainStats] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedUrl, setSelectedUrl] = useState('');
  const [urlCheckResult, setUrlCheckResult] = useState(null);
  const [checkingUrl, setCheckingUrl] = useState(false);

  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost/api';

  useEffect(() => {
    if (isAuthenticated) {
      fetchComplianceStats();
    }
  }, [isAuthenticated]);

  const fetchComplianceStats = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_URL}/compliance/stats`, {
        headers: {
          'Authorization': `Bearer ${authToken}`,
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setComplianceStats(data);
      setDomainStats(data.domain_stats || {});
    } catch (error) {
      console.error('Error fetching compliance stats:', error);
      setError('Failed to load compliance statistics');
    } finally {
      setLoading(false);
    }
  };

  const checkUrlCompliance = async () => {
    if (!selectedUrl.trim()) {
      setError('Please enter a URL to check');
      return;
    }

    try {
      setCheckingUrl(true);
      const response = await fetch(
        `${API_URL}/compliance/check?url=${encodeURIComponent(selectedUrl)}`,
        {
          headers: {
            'Authorization': `Bearer ${authToken}`,
          },
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setUrlCheckResult(data);
    } catch (error) {
      console.error('Error checking URL compliance:', error);
      setError('Error checking URL compliance');
      setUrlCheckResult({
        url: selectedUrl,
        status: 'error',
        can_scrape: false,
        reason: 'Failed to check compliance'
      });
    } finally {
      setCheckingUrl(false);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'allowed':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'disallowed':
        return <XCircle className="h-5 w-5 text-red-500" />;
      case 'rate_limited':
        return <Clock className="h-5 w-5 text-yellow-500" />;
      case 'cached':
        return <RefreshCw className="h-5 w-5 text-blue-500" />;
      default:
        return <AlertTriangle className="h-5 w-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'allowed':
        return 'bg-green-100 text-green-800';
      case 'disallowed':
        return 'bg-red-100 text-red-800';
      case 'rate_limited':
        return 'bg-yellow-100 text-yellow-800';
      case 'cached':
        return 'bg-blue-100 text-blue-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const formatDelay = (delaySeconds) => {
    if (delaySeconds < 1) {
      return `${Math.round(delaySeconds * 1000)}ms`;
    } else if (delaySeconds < 60) {
      return `${Math.round(delaySeconds)}s`;
    } else {
      const minutes = Math.floor(delaySeconds / 60);
      const seconds = Math.round(delaySeconds % 60);
      return `${minutes}m ${seconds}s`;
    }
  };

  if (loading) {
    return (
      <div className="p-6 bg-gray-50 min-h-screen">
        <div className="text-center py-8">
          <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Loading compliance statistics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2 flex items-center">
            <Shield className="h-8 w-8 mr-3 text-blue-600" />
            Compliance Monitor
          </h1>
          <p className="text-gray-600">
            Monitor ethical scraping compliance, rate limiting, and robots.txt adherence
          </p>
        </div>

        {/* URL Checker */}
        <div className="bg-white shadow-md rounded-lg p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
            <Globe className="h-6 w-6 mr-2 text-blue-600" />
            URL Compliance Checker
          </h2>
          <div className="flex space-x-4 mb-4">
            <input
              type="url"
              value={selectedUrl}
              onChange={(e) => setSelectedUrl(e.target.value)}
              placeholder="Enter URL to check compliance..."
              className="flex-1 px-4 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
            />
            <button
              onClick={checkUrlCompliance}
              disabled={checkingUrl || !selectedUrl.trim()}
              className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
            >
              {checkingUrl ? (
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Shield className="h-4 w-4 mr-2" />
              )}
              Check Compliance
            </button>
          </div>

          {urlCheckResult && (
            <div className="mt-4 p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-medium text-gray-900">Compliance Result</h3>
                <span className={`px-2 py-1 rounded-full text-sm font-medium ${getStatusColor(urlCheckResult.status)}`}>
                  {urlCheckResult.status.toUpperCase()}
                </span>
              </div>
              <div className="space-y-2 text-sm">
                <p><strong>URL:</strong> {urlCheckResult.url}</p>
                <p><strong>Can Scrape:</strong> {urlCheckResult.can_scrape ? 'Yes' : 'No'}</p>
                <p><strong>Reason:</strong> {urlCheckResult.reason}</p>
                {urlCheckResult.delay_seconds > 0 && (
                  <p><strong>Delay Required:</strong> {formatDelay(urlCheckResult.delay_seconds)}</p>
                )}
                {urlCheckResult.robots_txt_checked && (
                  <p><strong>Robots.txt Checked:</strong> Yes</p>
                )}
                {urlCheckResult.last_modified && (
                  <p><strong>Last Modified:</strong> {urlCheckResult.last_modified}</p>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Compliance Statistics */}
        {complianceStats && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div className="bg-white shadow-md rounded-lg p-6">
              <div className="flex items-center">
                <BarChart3 className="h-8 w-8 text-blue-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Total Domains</p>
                  <p className="text-2xl font-semibold text-gray-900">
                    {complianceStats.total_domains || 0}
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white shadow-md rounded-lg p-6">
              <div className="flex items-center">
                <Activity className="h-8 w-8 text-green-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Active Domains</p>
                  <p className="text-2xl font-semibold text-gray-900">
                    {Object.keys(domainStats).length}
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-white shadow-md rounded-lg p-6">
              <div className="flex items-center">
                <Clock className="h-8 w-8 text-yellow-600" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Rate Limited</p>
                  <p className="text-2xl font-semibold text-gray-900">
                    {Object.values(domainStats).filter(domain => domain.consecutive_errors > 0).length}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Domain Statistics Table */}
        <div className="bg-white shadow-md rounded-lg overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-xl font-semibold text-gray-900 flex items-center">
              <Activity className="h-6 w-6 mr-2 text-blue-600" />
              Domain Statistics
            </h2>
          </div>
          
          {Object.keys(domainStats).length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <Shield className="h-12 w-12 mx-auto mb-4 text-gray-400" />
              <p>No domain statistics available</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Domain
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Current Delay
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Consecutive Errors
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Last Request
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {Object.entries(domainStats).map(([domain, stats]) => (
                    <tr key={domain}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {domain}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatDelay(stats.current_delay)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          stats.consecutive_errors > 0 
                            ? 'bg-red-100 text-red-800' 
                            : 'bg-green-100 text-green-800'
                        }`}>
                          {stats.consecutive_errors}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {new Date(stats.last_request * 1000).toLocaleString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        {stats.consecutive_errors > 0 ? (
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                            <AlertTriangle className="h-3 w-3 mr-1" />
                            Rate Limited
                          </span>
                        ) : (
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                            <CheckCircle className="h-3 w-3 mr-1" />
                            Active
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Refresh Button */}
        <div className="mt-6 text-center">
          <button
            onClick={fetchComplianceStats}
            className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md shadow-sm text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh Statistics
          </button>
        </div>
      </div>
    </div>
  );
};

export default ComplianceMonitor;
