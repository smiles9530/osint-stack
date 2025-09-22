import React from 'react';
import { Settings as SettingsIcon, Bell, Shield, Palette, Globe } from 'lucide-react';

const Settings = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="text-gray-600">Application preferences and configuration</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {[
          { title: 'Notifications', icon: Bell, description: 'Configure alert preferences' },
          { title: 'Security', icon: Shield, description: 'Security and privacy settings' },
          { title: 'Appearance', icon: Palette, description: 'Theme and display options' },
          { title: 'Localization', icon: Globe, description: 'Language and region settings' }
        ].map((setting, index) => (
          <div key={index} className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
            <div className="flex items-center">
              <setting.icon className="w-8 h-8 text-blue-600 mr-4" />
              <div>
                <h3 className="text-lg font-semibold text-gray-900">{setting.title}</h3>
                <p className="text-gray-600">{setting.description}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Settings;
