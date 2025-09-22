import React, { useState, useEffect } from 'react';
import { 
  Search, 
  Filter, 
  Plus, 
  ExternalLink, 
  Calendar,
  Tag,
  TrendingUp,
  TrendingDown,
  Minus,
  Eye,
  Download,
  RefreshCw
} from 'lucide-react';
import { articlesAPI } from '../services/api';

const Articles = () => {
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filters, setFilters] = useState({
    source: '',
    sentiment: '',
    language: '',
    dateRange: '7d'
  });
  const [sortBy, setSortBy] = useState('created_at');
  const [sortOrder, setSortOrder] = useState('desc');
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(20);

  // Fetch articles
  const fetchArticles = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const params = {
        limit: itemsPerPage,
        offset: (currentPage - 1) * itemsPerPage,
        sort: sortBy,
        order: sortOrder,
        ...(searchTerm && { search: searchTerm }),
        ...(filters.source && { source: filters.source }),
        ...(filters.sentiment && { sentiment: filters.sentiment }),
        ...(filters.language && { language: filters.language })
      };

      const response = await articlesAPI.getArticles(params);
      setArticles(response.articles || []);
    } catch (err) {
      console.error('Failed to fetch articles:', err);
      setError('Failed to load articles');
      // Use mock data for demo
      setArticles([
        {
          id: 1,
          title: "Global Economic Trends Indicate Market Volatility",
          url: "https://example.com/article1",
          source: "Financial Times",
          created_at: "2024-09-15T10:30:00Z",
          sentiment_score: 0.2,
          language: "en",
          content_preview: "Economic indicators suggest increased market volatility in the coming quarter...",
          tags: ["economics", "finance", "markets"],
          chars: 1250,
          entities: ["Federal Reserve", "Wall Street", "GDP"]
        },
        {
          id: 2,
          title: "Breakthrough in Renewable Energy Technology",
          url: "https://example.com/article2",
          source: "Science Today",
          created_at: "2024-09-15T09:15:00Z",
          sentiment_score: 0.8,
          language: "en",
          content_preview: "Scientists announce major advancement in solar panel efficiency...",
          tags: ["technology", "renewable energy", "science"],
          chars: 980,
          entities: ["MIT", "Solar Technology", "Clean Energy"]
        },
        {
          id: 3,
          title: "Cybersecurity Threats Rise in Financial Sector",
          url: "https://example.com/article3",
          source: "Tech Security",
          created_at: "2024-09-15T08:45:00Z",
          sentiment_score: -0.4,
          language: "en",
          content_preview: "Financial institutions face increasing sophisticated cyber attacks...",
          tags: ["cybersecurity", "finance", "threats"],
          chars: 1100,
          entities: ["Banking", "Cyber Attacks", "Financial Security"]
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchArticles();
  }, [currentPage, sortBy, sortOrder, filters]);

  // Handle search
  const handleSearch = (e) => {
    e.preventDefault();
    setCurrentPage(1);
    fetchArticles();
  };

  // Get sentiment color and icon
  const getSentimentDisplay = (score) => {
    if (score > 0.1) {
      return { color: 'text-green-600', bg: 'bg-green-100', icon: TrendingUp, label: 'Positive' };
    } else if (score < -0.1) {
      return { color: 'text-red-600', bg: 'bg-red-100', icon: TrendingDown, label: 'Negative' };
    } else {
      return { color: 'text-gray-600', bg: 'bg-gray-100', icon: Minus, label: 'Neutral' };
    }
  };

  // Format date
  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const ArticleCard = ({ article }) => {
    const sentiment = getSentimentDisplay(article.sentiment_score);
    const SentimentIcon = sentiment.icon;

    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
        <div className="flex justify-between items-start mb-3">
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-gray-900 mb-2 line-clamp-2">
              {article.title}
            </h3>
            <div className="flex items-center space-x-4 text-sm text-gray-500 mb-2">
              <span className="font-medium">{article.source}</span>
              <span className="flex items-center">
                <Calendar className="w-4 h-4 mr-1" />
                {formatDate(article.created_at)}
              </span>
              <span>{article.language.toUpperCase()}</span>
            </div>
          </div>
          <div className={`flex items-center px-2 py-1 rounded-full ${sentiment.bg}`}>
            <SentimentIcon className={`w-4 h-4 ${sentiment.color}`} />
            <span className={`ml-1 text-xs font-medium ${sentiment.color}`}>
              {sentiment.label}
            </span>
          </div>
        </div>

        <p className="text-gray-600 text-sm mb-4 line-clamp-3">
          {article.content_preview}
        </p>

        <div className="flex flex-wrap gap-2 mb-4">
          {article.tags?.slice(0, 3).map((tag, index) => (
            <span
              key={index}
              className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-700"
            >
              <Tag className="w-3 h-3 mr-1" />
              {tag}
            </span>
          ))}
        </div>

        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-4 text-sm text-gray-500">
            <span>{article.chars} chars</span>
            <span>{article.entities?.length || 0} entities</span>
          </div>
          <div className="flex items-center space-x-2">
            <button className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded">
              <Eye className="w-4 h-4" />
            </button>
            <button className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded">
              <Download className="w-4 h-4" />
            </button>
            <a
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded"
            >
              <ExternalLink className="w-4 h-4" />
            </a>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Articles</h1>
          <p className="text-gray-600">Manage and explore collected articles</p>
        </div>
        <button className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
          <Plus className="w-4 h-4 mr-2" />
          Add Article
        </button>
      </div>

      {/* Search and Filters */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <form onSubmit={handleSearch} className="flex flex-col lg:flex-row gap-4">
          {/* Search Input */}
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search articles by title, content, or source..."
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
          </div>

          {/* Filters */}
          <div className="flex flex-wrap gap-2">
            <select
              value={filters.source}
              onChange={(e) => setFilters(prev => ({ ...prev, source: e.target.value }))}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All Sources</option>
              <option value="Financial Times">Financial Times</option>
              <option value="Science Today">Science Today</option>
              <option value="Tech Security">Tech Security</option>
            </select>

            <select
              value={filters.sentiment}
              onChange={(e) => setFilters(prev => ({ ...prev, sentiment: e.target.value }))}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All Sentiments</option>
              <option value="positive">Positive</option>
              <option value="neutral">Neutral</option>
              <option value="negative">Negative</option>
            </select>

            <select
              value={filters.language}
              onChange={(e) => setFilters(prev => ({ ...prev, language: e.target.value }))}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="">All Languages</option>
              <option value="en">English</option>
              <option value="es">Spanish</option>
              <option value="fr">French</option>
            </select>

            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Search
            </button>

            <button
              type="button"
              onClick={fetchArticles}
              className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </form>
      </div>

      {/* Sort Options */}
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <span className="text-sm font-medium text-gray-700">Sort by:</span>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="px-3 py-1 border border-gray-300 rounded text-sm focus:ring-2 focus:ring-blue-500"
          >
            <option value="created_at">Date Created</option>
            <option value="title">Title</option>
            <option value="source">Source</option>
            <option value="sentiment_score">Sentiment</option>
          </select>
          <button
            onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
            className="p-1 text-gray-400 hover:text-gray-600"
          >
            {sortOrder === 'asc' ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
          </button>
        </div>
        <div className="text-sm text-gray-500">
          {articles.length} articles found
        </div>
      </div>

      {/* Articles Grid */}
      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-4 text-blue-600" />
            <p className="text-gray-600">Loading articles...</p>
          </div>
        </div>
      ) : error ? (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-600">{error}</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {articles.map((article) => (
            <ArticleCard key={article.id} article={article} />
          ))}
        </div>
      )}

      {/* Pagination */}
      {articles.length > 0 && (
        <div className="flex justify-center items-center space-x-2">
          <button
            onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
            disabled={currentPage === 1}
            className="px-3 py-1 border border-gray-300 rounded text-sm disabled:opacity-50"
          >
            Previous
          </button>
          <span className="px-3 py-1 text-sm text-gray-600">
            Page {currentPage}
          </span>
          <button
            onClick={() => setCurrentPage(prev => prev + 1)}
            disabled={articles.length < itemsPerPage}
            className="px-3 py-1 border border-gray-300 rounded text-sm disabled:opacity-50"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
};

export default Articles;
