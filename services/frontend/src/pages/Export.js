import React, { useState } from 'react';
import { 
  Download, FileText, Table, Image, Calendar, Filter, Settings,
  CheckCircle, Clock, AlertCircle, Database, BarChart3, PieChart,
  FileSpreadsheet, FileImage, File, RefreshCw
} from 'lucide-react';

const Export = () => {
  const [selectedDataType, setSelectedDataType] = useState('articles');
  const [selectedFormat, setSelectedFormat] = useState('csv');
  const [dateRange, setDateRange] = useState({
    start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    end: new Date().toISOString().split('T')[0]
  });
  const [filters, setFilters] = useState({
    sentiment: 'all',
    language: 'all',
    source: 'all'
  });
  const [isExporting, setIsExporting] = useState(false);
  const [exportHistory, setExportHistory] = useState([
    {
      id: 1,
      filename: 'articles_export_2025-09-15.csv',
      type: 'articles',
      format: 'csv',
      status: 'completed',
      createdAt: new Date(Date.now() - 2 * 60 * 60 * 1000),
      size: '2.3 MB',
      records: 1250
    },
    {
      id: 2,
      filename: 'analytics_report_2025-09-14.pdf',
      type: 'analytics',
      format: 'pdf',
      status: 'completed',
      createdAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000),
      size: '5.7 MB',
      records: null
    },
    {
      id: 3,
      filename: 'sentiment_analysis_2025-09-13.xlsx',
      type: 'sentiment',
      format: 'xlsx',
      status: 'failed',
      createdAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
      size: null,
      records: null,
      error: 'Export timeout after 5 minutes'
    }
  ]);

  const dataTypes = [
    {
      id: 'articles',
      name: 'Articles',
      description: 'Raw article data with metadata',
      icon: FileText,
      estimatedSize: '~2.1 MB for 1000 articles'
    },
    {
      id: 'analytics',
      name: 'Analytics Data',
      description: 'Sentiment trends, topic distribution',
      icon: BarChart3,
      estimatedSize: '~500 KB for 30 days'
    },
    {
      id: 'sentiment',
      name: 'Sentiment Analysis',
      description: 'Article sentiment scores and trends',
      icon: PieChart,
      estimatedSize: '~300 KB for 1000 records'
    },
    {
      id: 'entities',
      name: 'Named Entities',
      description: 'Extracted people, organizations, locations',
      icon: Database,
      estimatedSize: '~800 KB for 1000 articles'
    }
  ];

  const formats = [
    {
      id: 'csv',
      name: 'CSV',
      description: 'Comma-separated values',
      icon: FileSpreadsheet,
      mimeType: 'text/csv'
    },
    {
      id: 'xlsx',
      name: 'Excel',
      description: 'Microsoft Excel spreadsheet',
      icon: Table,
      mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    },
    {
      id: 'json',
      name: 'JSON',
      description: 'JavaScript Object Notation',
      icon: File,
      mimeType: 'application/json'
    },
    {
      id: 'pdf',
      name: 'PDF Report',
      description: 'Formatted analytics report',
      icon: FileText,
      mimeType: 'application/pdf'
    }
  ];

  const handleExport = async () => {
    setIsExporting(true);
    
    try {
      // Simulate export process
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // Add to export history
      const newExport = {
        id: Date.now(),
        filename: `${selectedDataType}_export_${new Date().toISOString().split('T')[0]}.${selectedFormat}`,
        type: selectedDataType,
        format: selectedFormat,
        status: 'completed',
        createdAt: new Date(),
        size: Math.random() > 0.1 ? `${(Math.random() * 5 + 0.5).toFixed(1)} MB` : null,
        records: Math.random() > 0.1 ? Math.floor(Math.random() * 2000 + 100) : null
      };
      
      setExportHistory(prev => [newExport, ...prev]);
      
      // Trigger download
      const blob = new Blob(['Sample export data'], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = newExport.filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setIsExporting(false);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'processing':
        return <Clock className="w-5 h-5 text-yellow-600" />;
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-red-600" />;
      default:
        return <Clock className="w-5 h-5 text-gray-600" />;
    }
  };

  const formatTimeAgo = (timestamp) => {
    const now = new Date();
    const diffInHours = Math.floor((now - timestamp) / (1000 * 60 * 60));
    
    if (diffInHours < 1) {
      const diffInMinutes = Math.floor((now - timestamp) / (1000 * 60));
      return `${diffInMinutes}m ago`;
    }
    
    if (diffInHours < 24) {
      return `${diffInHours}h ago`;
    }
    
    const diffInDays = Math.floor(diffInHours / 24);
    return `${diffInDays}d ago`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Data Export</h1>
        <p className="text-gray-600">Export OSINT data in various formats for analysis and reporting</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Export Configuration */}
        <div className="lg:col-span-2 space-y-6">
          {/* Data Type Selection */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Select Data Type</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {dataTypes.map((type) => {
                const Icon = type.icon;
                return (
                  <div
                    key={type.id}
                    onClick={() => setSelectedDataType(type.id)}
                    className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                      selectedDataType === type.id
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="flex items-start space-x-3">
                      <Icon className={`w-6 h-6 ${selectedDataType === type.id ? 'text-blue-600' : 'text-gray-600'}`} />
                      <div className="flex-1">
                        <h4 className="font-medium text-gray-900">{type.name}</h4>
                        <p className="text-sm text-gray-600 mb-1">{type.description}</p>
                        <p className="text-xs text-gray-500">{type.estimatedSize}</p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Format Selection */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Export Format</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {formats.map((format) => {
                const Icon = format.icon;
                return (
                  <div
                    key={format.id}
                    onClick={() => setSelectedFormat(format.id)}
                    className={`p-4 rounded-lg border-2 cursor-pointer transition-all text-center ${
                      selectedFormat === format.id
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <Icon className={`w-8 h-8 mx-auto mb-2 ${selectedFormat === format.id ? 'text-blue-600' : 'text-gray-600'}`} />
                    <h4 className="font-medium text-gray-900 text-sm">{format.name}</h4>
                    <p className="text-xs text-gray-600">{format.description}</p>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Filters */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Export Filters</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Date Range */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Date Range</label>
                <div className="grid grid-cols-2 gap-2">
                  <input
                    type="date"
                    value={dateRange.start}
                    onChange={(e) => setDateRange(prev => ({ ...prev, start: e.target.value }))}
                    className="px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  <input
                    type="date"
                    value={dateRange.end}
                    onChange={(e) => setDateRange(prev => ({ ...prev, end: e.target.value }))}
                    className="px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>

              {/* Additional Filters */}
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Sentiment</label>
                  <select
                    value={filters.sentiment}
                    onChange={(e) => setFilters(prev => ({ ...prev, sentiment: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="all">All Sentiments</option>
                    <option value="positive">Positive</option>
                    <option value="neutral">Neutral</option>
                    <option value="negative">Negative</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Language</label>
                  <select
                    value={filters.language}
                    onChange={(e) => setFilters(prev => ({ ...prev, language: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    <option value="all">All Languages</option>
                    <option value="en">English</option>
                    <option value="es">Spanish</option>
                    <option value="fr">French</option>
                  </select>
                </div>
              </div>
            </div>
          </div>

          {/* Export Button */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Ready to Export</h3>
                <p className="text-gray-600">
                  Export {dataTypes.find(t => t.id === selectedDataType)?.name} as {formats.find(f => f.id === selectedFormat)?.name}
                </p>
              </div>
              <button
                onClick={handleExport}
                disabled={isExporting}
                className="flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isExporting ? (
                  <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                ) : (
                  <Download className="w-5 h-5 mr-2" />
                )}
                {isExporting ? 'Exporting...' : 'Start Export'}
              </button>
            </div>
          </div>
        </div>

        {/* Export History Sidebar */}
        <div className="space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Export History</h3>
            <div className="space-y-3">
              {exportHistory.map((exportItem) => (
                <div key={exportItem.id} className="p-4 border border-gray-200 rounded-lg">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(exportItem.status)}
                      <span className="text-sm font-medium text-gray-900">
                        {exportItem.type}
                      </span>
                    </div>
                    <span className="text-xs text-gray-500">
                      {formatTimeAgo(exportItem.createdAt)}
                    </span>
                  </div>
                  
                  <p className="text-sm text-gray-600 mb-2 truncate" title={exportItem.filename}>
                    {exportItem.filename}
                  </p>
                  
                  {exportItem.status === 'completed' && (
                    <div className="flex justify-between text-xs text-gray-500 mb-2">
                      <span>{exportItem.size}</span>
                      {exportItem.records && <span>{exportItem.records} records</span>}
                    </div>
                  )}
                  
                  {exportItem.status === 'failed' && exportItem.error && (
                    <p className="text-xs text-red-600 mb-2">{exportItem.error}</p>
                  )}
                  
                  {exportItem.status === 'completed' && (
                    <button className="w-full flex items-center justify-center px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200">
                      <Download className="w-4 h-4 mr-1" />
                      Download
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Quick Export Templates */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Templates</h3>
            <div className="space-y-2">
              {[
                { name: 'Daily Report', desc: 'Last 24h articles' },
                { name: 'Weekly Analytics', desc: 'Past 7 days data' },
                { name: 'Monthly Summary', desc: 'Full month analysis' }
              ].map((template, index) => (
                <button
                  key={index}
                  className="w-full text-left p-3 border border-gray-200 rounded-lg hover:bg-gray-50"
                >
                  <p className="font-medium text-gray-900 text-sm">{template.name}</p>
                  <p className="text-xs text-gray-600">{template.desc}</p>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Export;
