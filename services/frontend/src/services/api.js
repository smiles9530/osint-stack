import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost/api';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Clear invalid token
      localStorage.removeItem('auth_token');
      localStorage.removeItem('user');
      // Redirect to login or trigger auth context update
    }
    return Promise.reject(error);
  }
);

// Authentication API
export const authAPI = {
  login: async (username, password) => {
    const response = await api.post('/auth/login', { username, password });
    return response.data;
  },
  
  logout: async () => {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user');
    return { success: true };
  },

  getCurrentUser: () => {
    const user = localStorage.getItem('user');
    return user ? JSON.parse(user) : null;
  },

  isAuthenticated: () => {
    return !!localStorage.getItem('auth_token');
  }
};

// Health API
export const healthAPI = {
  getSystemHealth: async () => {
    const response = await api.get('/healthz');
    return response.data;
  },

  getMetrics: async () => {
    const response = await api.get('/metrics');
    return response.data;
  },

  getQueueStats: async () => {
    const response = await api.get('/queue/stats');
    return response.data;
  }
};

// Articles API
export const articlesAPI = {
  getArticles: async (params = {}) => {
    const response = await api.get('/articles', { params });
    return response.data;
  },

  getArticle: async (id) => {
    const response = await api.get(`/articles/${id}`);
    return response.data;
  },

  submitArticle: async (data) => {
    const response = await api.post('/ingest/fetch_extract', data);
    return response.data;
  },

  generateEmbedding: async (data) => {
    const response = await api.post('/embed', data);
    return response.data;
  }
};

// Analytics API
export const analyticsAPI = {
  getDashboard: async (timePeriod = '24h') => {
    const response = await api.get(`/analytics/dashboard?time_period=${timePeriod}`);
    return response.data;
  },

  getSentimentTrends: async (timePeriod = '7d') => {
    const response = await api.get(`/analytics/sentiment/trends?time_period=${timePeriod}`);
    return response.data;
  },

  getTopicDistribution: async (timePeriod = '7d') => {
    const response = await api.get(`/analytics/topics/distribution?time_period=${timePeriod}`);
    return response.data;
  },

  getEntityFrequency: async (timePeriod = '7d') => {
    const response = await api.get(`/analytics/entities/frequency?time_period=${timePeriod}`);
    return response.data;
  },

  getAnomalies: async (timePeriod = '24h') => {
    const response = await api.get(`/analytics/anomalies?time_period=${timePeriod}`);
    return response.data;
  },

  getQueueStats: async () => {
    try {
      const response = await api.get('/queue/stats');
      return response.data;
    } catch (error) {
      // Return default values if endpoint doesn't exist yet
      return { pending_jobs: 0, failed_jobs: 0, completed_jobs: 0, active_workers: 0 };
    }
  },

  exportData: async (format, dataType, timePeriod) => {
    const response = await api.post(`/analytics/export/${format}?data_type=${dataType}&time_period=${timePeriod}`, {}, {
      responseType: 'blob'
    });
    return response.data;
  }
};

// Users API
export const usersAPI = {
  getUsers: async () => {
    const response = await api.get('/users');
    return response.data;
  },

  createUser: async (userData) => {
    const response = await api.post('/users', userData);
    return response.data;
  }
};

// N8N Workflow Integration
export const workflowAPI = {
  triggerWorkflow: async (type, params) => {
    // Use N8N webhook endpoints we created
    const webhookUrl = `${process.env.REACT_APP_N8N_URL || '/n8n/'}webhook/frontend-data`;
    
    const response = await axios.post(webhookUrl, {
      type,
      params,
      frontend_id: 'osint_frontend',
      timestamp: new Date().toISOString()
    });
    
    return response.data;
  },

  getDashboardData: async () => {
    return workflowAPI.triggerWorkflow('dashboard_data', { time_period: '24h' });
  },

  submitArticleViaWorkflow: async (articleData) => {
    return workflowAPI.triggerWorkflow('article_submit', articleData);
  },

  exportViaWorkflow: async (dataType, timePeriod) => {
    return workflowAPI.triggerWorkflow('analytics_export', { data_type: dataType, time_period: timePeriod });
  },

  getSystemStatus: async () => {
    return workflowAPI.triggerWorkflow('system_status', {});
  }
};

// Individual exports for backward compatibility
export const login = authAPI.login;
export const getHealth = healthAPI.getHealth;
export const getQueueStats = analyticsAPI.getQueueStats;

export default api;
