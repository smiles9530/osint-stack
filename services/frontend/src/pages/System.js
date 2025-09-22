import React from 'react';
import { Server, Database, Settings, Users } from 'lucide-react';

const System = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">System Administration</h1>
        <p className="text-gray-600">System configuration and administration tools</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {[
          { title: 'Server Configuration', icon: Server, description: 'Manage server settings' },
          { title: 'Database Management', icon: Database, description: 'Database administration' },
          { title: 'System Settings', icon: Settings, description: 'Global system configuration' },
          { title: 'User Administration', icon: Users, description: 'Manage user accounts' }
        ].map((item, index) => (
          <div key={index} className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
            <div className="flex items-center">
              <item.icon className="w-8 h-8 text-blue-600 mr-4" />
              <div>
                <h3 className="text-lg font-semibold text-gray-900">{item.title}</h3>
                <p className="text-gray-600">{item.description}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default System;
