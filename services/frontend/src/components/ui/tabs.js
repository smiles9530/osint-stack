import React, { useState } from 'react';

export const Tabs = ({ defaultValue, value, onValueChange, className = '', children, ...props }) => {
  const [internalValue, setInternalValue] = useState(defaultValue);
  const currentValue = value !== undefined ? value : internalValue;
  
  const handleValueChange = (newValue) => {
    if (value === undefined) {
      setInternalValue(newValue);
    }
    onValueChange?.(newValue);
  };
  
  return (
    <div className={`w-full ${className}`} {...props}>
      {React.Children.map(children, child => 
        React.cloneElement(child, { 
          value: currentValue, 
          onValueChange: handleValueChange 
        })
      )}
    </div>
  );
};

export const TabsList = ({ className = '', children, ...props }) => (
  <div className={`inline-flex h-10 items-center justify-center rounded-md bg-gray-100 p-1 text-gray-500 ${className}`} {...props}>
    {children}
  </div>
);

export const TabsTrigger = ({ 
  value, 
  onValueChange, 
  className = '', 
  children, 
  ...props 
}) => {
  const handleClick = () => {
    onValueChange?.(value);
  };
  
  return (
    <button
      className={`inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-white transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-950 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 ${className}`}
      onClick={handleClick}
      {...props}
    >
      {children}
    </button>
  );
};

export const TabsContent = ({ 
  value, 
  onValueChange, 
  className = '', 
  children, 
  ...props 
}) => {
  // This component doesn't need to handle value changes
  return (
    <div className={`mt-2 ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-950 focus-visible:ring-offset-2 ${className}`} {...props}>
      {children}
    </div>
  );
};
