import React from 'react';

export const Progress = ({ 
  className = '', 
  value = 0, 
  max = 100, 
  children, 
  ...props 
}) => {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);
  
  return (
    <div 
      className={`relative h-4 w-full overflow-hidden rounded-full bg-gray-200 ${className}`} 
      {...props}
    >
      <div
        className="h-full w-full flex-1 bg-blue-600 transition-all duration-300 ease-in-out"
        style={{ transform: `translateX(-${100 - percentage}%)` }}
      />
      {children}
    </div>
  );
};
