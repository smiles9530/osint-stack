import React from 'react';

export const Alert = ({ 
  className = '', 
  variant = 'default', 
  children, 
  ...props 
}) => {
  const baseClasses = 'relative w-full rounded-lg border p-4';
  
  const variants = {
    default: 'bg-white text-gray-900 border-gray-200',
    destructive: 'bg-red-50 text-red-900 border-red-200',
    warning: 'bg-yellow-50 text-yellow-900 border-yellow-200',
    success: 'bg-green-50 text-green-900 border-green-200'
  };
  
  const classes = `${baseClasses} ${variants[variant]} ${className}`;
  
  return (
    <div className={classes} {...props}>
      {children}
    </div>
  );
};

export const AlertDescription = ({ 
  className = '', 
  children, 
  ...props 
}) => (
  <div className={`text-sm ${className}`} {...props}>
    {children}
  </div>
);
