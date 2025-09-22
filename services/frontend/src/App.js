import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { AppProvider } from './contexts/AppContext';
import Layout from './components/Layout';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Articles from './pages/Articles';
import Analytics from './pages/Analytics';
import Search from './pages/Search';
import Sources from './pages/Sources';
import Monitoring from './pages/Monitoring';
import Alerts from './pages/Alerts';
import Export from './pages/Export';
import System from './pages/System';
import Workflows from './pages/Workflows';
import Users from './pages/Users';
import Settings from './pages/Settings';
import MLAnalysisDashboard from './components/MLAnalysisDashboard';
import FileManager from './components/FileManager';
import ComplianceMonitor from './components/ComplianceMonitor';
import './App.css';

// Protected Route Component
const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  return isAuthenticated ? children : <Navigate to="/login" replace />;
};

// Main App Router
const AppRouter = () => {
  return (
    <Router>
      <Routes>
        {/* Public Route */}
        <Route path="/login" element={<Login />} />
        
        {/* Protected Routes */}
        <Route path="/" element={
          <ProtectedRoute>
            <AppProvider>
              <Layout />
            </AppProvider>
          </ProtectedRoute>
        }>
          <Route index element={<Dashboard />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="ml-analysis" element={<MLAnalysisDashboard />} />
          <Route path="articles" element={<Articles />} />
          <Route path="search" element={<Search />} />
          <Route path="sources" element={<Sources />} />
          <Route path="monitoring" element={<Monitoring />} />
          <Route path="alerts" element={<Alerts />} />
          <Route path="export" element={<Export />} />
          <Route path="files" element={<FileManager />} />
          <Route path="compliance" element={<ComplianceMonitor />} />
          <Route path="system" element={<System />} />
          <Route path="workflows" element={<Workflows />} />
          <Route path="users" element={<Users />} />
          <Route path="settings" element={<Settings />} />
        </Route>

        {/* Catch all route */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
};

// Main App Component
const App = () => {
  return (
    <AuthProvider>
      <AppRouter />
    </AuthProvider>
  );
};

export default App;