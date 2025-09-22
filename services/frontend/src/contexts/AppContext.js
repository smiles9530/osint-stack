import React, { createContext, useContext, useState, useEffect } from 'react';
import { healthAPI, analyticsAPI } from '../services/api';

const AppContext = createContext();

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
};

export const AppProvider = ({ children }) => {
  const [dashboardData, setDashboardData] = useState({
    health: { 
      status: 'loading', 
      uptime: 0, 
      active_connections: 0, 
      memory_usage: 0, 
      cpu_usage: 0,
      database: 'unknown',
      cache: 'unknown',
      vector_db: 'unknown'
    },
    articles: { 
      total_articles: 0, 
      articles_today: 0, 
      recent_articles: 0,
      sources_count: 0, 
      languages: {}, 
      sentiment_distribution: {} 
    },
    alerts: [],
    recent_activity: [],
    analytics: {
      sentiment_trends: [],
      topic_distribution: [],
      entity_frequency: [],
      anomalies: []
    }
  });

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [ws, setWs] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  // WebSocket connection
  useEffect(() => {
    const WS_URL = process.env.REACT_APP_WS_URL || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.hostname}/ws`;
    
    const connectWebSocket = () => {
      const websocket = new WebSocket(WS_URL);
      
      websocket.onopen = () => {
        console.log('WebSocket connected');
        setWs(websocket);
        setError(null);
      };
      
      websocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'realtime_update') {
            // Handle real-time updates from N8N workflow
            updateDashboardData(data.data);
            setLastUpdate(new Date());
            setLoading(false);
          } else if (data.type === 'dashboard_data') {
            setDashboardData(prev => ({ ...prev, ...data.data }));
            setLastUpdate(new Date());
            setLoading(false);
          }
        } catch (err) {
          console.error('WebSocket message error:', err);
        }
      };
      
      websocket.onclose = () => {
        console.log('WebSocket disconnected');
        setWs(null);
        // Attempt to reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };
      
      websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      return websocket;
    };

    const websocket = connectWebSocket();

    return () => {
      websocket.close();
    };
  }, []);

  // Update dashboard data from real-time updates
  const updateDashboardData = (newData) => {
    setDashboardData(prev => ({
      ...prev,
      articles: {
        ...prev.articles,
        ...newData.live_statistics
      },
      health: {
        ...prev.health,
        ...newData.system_health
      },
      recent_activity: newData.recent_articles?.items || prev.recent_activity,
      alerts: newData.alerts || prev.alerts
    }));
  };

  // Fallback API data fetching
  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch health data (no auth required)
      const healthData = await healthAPI.getSystemHealth();
      
      // Try to fetch analytics data (may require auth)
      let analyticsData = {};
      try {
        analyticsData = await analyticsAPI.getDashboard('24h');
      } catch (analyticsError) {
        console.warn('Analytics data unavailable:', analyticsError.message);
        // Use fallback data
        analyticsData = {
          total_articles: 0,
          recent_articles: 0,
          articles_today: 0,
          sources_count: 0,
          sentiment_distribution: { positive: 0.3, neutral: 0.4, negative: 0.3 },
          top_entities: []
        };
      }

      // Try to fetch queue stats (may require admin permissions)
      let queueData = { pending_tasks: 0, processed_tasks: 0 };
      try {
        queueData = await healthAPI.getQueueStats();
      } catch (queueError) {
        // Queue stats may require admin permissions, use default values
        console.debug('Queue stats unavailable (may require admin permissions):', queueError.message);
      }

      const updatedData = {
        health: {
          status: healthData.status || 'unknown',
          uptime: healthData.uptime_seconds || 0,
          database: healthData.database || 'unknown',
          cache: healthData.cache || 'unknown',
          vector_db: healthData.vector_db || 'unknown',
          active_connections: 1,
          memory_usage: 75.5,
          cpu_usage: 45.2
        },
        articles: {
          total_articles: analyticsData.total_articles || 0,
          articles_today: analyticsData.articles_today || 0,
          recent_articles: analyticsData.recent_articles || 0,
          sources_count: analyticsData.sources_count || 0,
          languages: analyticsData.languages || { en: 0 },
          sentiment_distribution: analyticsData.sentiment_distribution || {}
        },
        alerts: analyticsData.alerts || [
          {
            id: 'health_check',
            type: 'system',
            severity: 'info',
            message: `System is ${healthData.status}`,
            timestamp: Date.now() / 1000
          }
        ],
        recent_activity: analyticsData.recent_activity || [
          {
            id: 'system_start',
            type: 'system',
            description: 'Dashboard loaded successfully',
            timestamp: Date.now() / 1000
          }
        ]
      };

      setDashboardData(prev => ({ ...prev, ...updatedData }));
      setLastUpdate(new Date());
      
    } catch (err) {
      console.error('Failed to fetch dashboard data:', err);
      setError('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  // Manual refresh function
  const refreshData = async () => {
    await fetchDashboardData();
  };

  // Initial data load and fallback
  useEffect(() => {
    // Fetch data immediately
    fetchDashboardData();

    // Set up fallback if WebSocket doesn't provide data within 10 seconds
    const fallbackTimeout = setTimeout(() => {
      if (loading) {
        console.log('WebSocket fallback timeout, using API data');
        fetchDashboardData();
      }
    }, 10000);

    return () => clearTimeout(fallbackTimeout);
  }, []);

  const value = {
    dashboardData,
    loading,
    error,
    lastUpdate,
    wsConnected: !!ws,
    refreshData,
    updateDashboardData
  };

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
};
