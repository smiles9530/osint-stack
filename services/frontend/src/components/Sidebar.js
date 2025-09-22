import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  BarChart3,
  FileText,
  Search,
  Settings,
  Users,
  Activity,
  TrendingUp,
  Database,
  Globe,
  AlertTriangle,
  Download,
  Zap,
  Folder,
  Shield,
  Code,
  Brain
} from 'lucide-react';

const Sidebar = ({ isOpen, onClose }) => {
  const navigation = [
    {
      name: 'Overview',
      href: '/',
      icon: BarChart3,
      description: 'Main dashboard'
    },
    {
      name: 'Analytics',
      href: '/analytics',
      icon: TrendingUp,
      description: 'Advanced analytics'
    },
    {
      name: 'ML Analysis',
      href: '/ml-analysis',
      icon: Brain,
      description: 'AI-powered insights'
    },
    {
      name: 'Articles',
      href: '/articles',
      icon: FileText,
      description: 'Article management'
    },
    {
      name: 'Search',
      href: '/search',
      icon: Search,
      description: 'Search & discovery'
    },
    {
      name: 'Sources',
      href: '/sources',
      icon: Globe,
      description: 'Data sources'
    },
    {
      name: 'Monitoring',
      href: '/monitoring',
      icon: Activity,
      description: 'System monitoring'
    },
    {
      name: 'Alerts',
      href: '/alerts',
      icon: AlertTriangle,
      description: 'Alerts & notifications'
    },
    {
      name: 'Export',
      href: '/export',
      icon: Download,
      description: 'Data export'
    },
    {
      name: 'Files',
      href: '/files',
      icon: Folder,
      description: 'File management'
    },
    {
      name: 'Compliance',
      href: '/compliance',
      icon: Shield,
      description: 'Ethical scraping compliance'
    },
    {
      name: 'Cheerio Parser',
      href: '/cheerio',
      icon: Code,
      description: 'HTML parsing and manipulation'
    },
    {
      name: 'System',
      href: '/system',
      icon: Database,
      description: 'System administration'
    },
    {
      name: 'Workflows',
      href: '/workflows',
      icon: Zap,
      description: 'N8N workflows'
    }
  ];

  const bottomNavigation = [
    {
      name: 'Users',
      href: '/users',
      icon: Users,
      description: 'User management'
    },
    {
      name: 'Settings',
      href: '/settings',
      icon: Settings,
      description: 'Application settings'
    }
  ];

  const NavItem = ({ item, onClick }) => (
    <NavLink
      to={item.href}
      onClick={onClick}
      className={({ isActive }) =>
        `group flex items-center px-2 py-2 text-sm font-medium rounded-md transition-colors ${
          isActive
            ? 'bg-blue-100 text-blue-700'
            : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
        }`
      }
    >
      <item.icon
        className={`mr-3 flex-shrink-0 h-5 w-5 transition-colors ${
          isOpen ? 'mr-3' : 'mr-0'
        }`}
        aria-hidden="true"
      />
      {isOpen && (
        <div className="flex-1 min-w-0">
          <div className="font-medium">{item.name}</div>
          <div className="text-xs text-gray-500 truncate">{item.description}</div>
        </div>
      )}
    </NavLink>
  );

  return (
    <>
      {/* Desktop Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-50 transition-all duration-300 transform ${
        isOpen ? 'w-64' : 'w-16'
      } bg-white shadow-lg border-r border-gray-200`}>
        
        {/* Logo */}
        <div className="flex items-center h-16 px-4 border-b border-gray-200">
          <div className="flex items-center">
            <div className="flex-shrink-0 w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
              <Activity className="w-5 h-5 text-white" />
            </div>
            {isOpen && (
              <div className="ml-3">
                <div className="text-lg font-bold text-gray-900">OSINT</div>
                <div className="text-xs text-gray-500">Analytics</div>
              </div>
            )}
          </div>
        </div>

        {/* Navigation */}
        <div className="flex flex-col h-full">
          <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
            {navigation.map((item) => (
              <NavItem 
                key={item.name} 
                item={item} 
                onClick={() => window.innerWidth < 1024 && onClose()}
              />
            ))}
          </nav>

          {/* Bottom navigation */}
          <nav className="px-2 py-4 border-t border-gray-200 space-y-1">
            {bottomNavigation.map((item) => (
              <NavItem 
                key={item.name} 
                item={item} 
                onClick={() => window.innerWidth < 1024 && onClose()}
              />
            ))}
          </nav>
        </div>
      </div>
    </>
  );
};

export default Sidebar;
