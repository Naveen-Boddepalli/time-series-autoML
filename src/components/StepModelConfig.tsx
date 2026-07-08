'use client';

import React, { useState, useEffect } from 'react';
import { useStore } from '../store/useStore';

export default function StepModelConfig() {
  const { setModelsToTrain, setStep } = useStore();
  
  const [selectedModels, setSelectedModels] = useState<Record<string, boolean>>({
    arima: true,
    boosting: true,
    lstm: true
  });
  
  const [deviceCapability, setDeviceCapability] = useState('Detecting...');

  useEffect(() => {
    // Basic device detection to set expectations
    const cores = navigator.hardwareConcurrency || 2;
    if (cores >= 8) {
      setDeviceCapability('High-end device detected. Training should be fast.');
    } else if (cores >= 4) {
      setDeviceCapability('Standard device detected. Deep learning might take a moment.');
    } else {
      setDeviceCapability('Low-power device detected. Consider disabling LSTM for faster results.');
    }
  }, []);

  const handleToggle = (model: string) => {
    setSelectedModels(prev => ({
      ...prev,
      [model]: !prev[model]
    }));
  };

  const handleContinue = () => {
    const toTrain = Object.keys(selectedModels).filter(k => selectedModels[k]);
    if (toTrain.length === 0) {
      alert("Please select at least one model to train.");
      return;
    }
    setModelsToTrain(toTrain);
    setStep('TRAIN');
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="bg-white dark:bg-gray-900 p-6 rounded-xl shadow-sm border border-gray-100 dark:border-gray-800">
        <h2 className="text-2xl font-bold mb-2 text-gray-800 dark:text-gray-100">Model Selection</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          All models will train locally on your device. {deviceCapability}
        </p>

        <div className="space-y-4 mb-8">
          {/* ARIMA */}
          <div 
            className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${selectedModels.arima ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'}`}
            onClick={() => handleToggle('arima')}
          >
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-lg dark:text-white">ARIMA (Statsmodels)</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">Classical statistical method. Fast and reliable for simple trends.</p>
              </div>
              <input type="checkbox" checked={selectedModels.arima} readOnly className="w-5 h-5 text-blue-600 rounded" />
            </div>
          </div>

          {/* Boosting */}
          <div 
            className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${selectedModels.boosting ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'}`}
            onClick={() => handleToggle('boosting')}
          >
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-lg dark:text-white">Gradient Boosting (Scikit-Learn)</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">Machine learning approach (XGBoost alternative). Good for non-linear patterns.</p>
              </div>
              <input type="checkbox" checked={selectedModels.boosting} readOnly className="w-5 h-5 text-blue-600 rounded" />
            </div>
          </div>

          {/* LSTM */}
          <div 
            className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${selectedModels.lstm ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'}`}
            onClick={() => handleToggle('lstm')}
          >
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-lg dark:text-white">LSTM (TensorFlow.js)</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">Deep learning model. Computationally intensive but catches complex sequences. Uses WebGL/WebGPU.</p>
              </div>
              <input type="checkbox" checked={selectedModels.lstm} readOnly className="w-5 h-5 text-blue-600 rounded" />
            </div>
          </div>
        </div>

        <div className="flex justify-between items-center pt-4 border-t border-gray-100 dark:border-gray-800">
          <button 
            onClick={() => setStep('EDA')}
            className="px-4 py-2 text-sm font-medium text-gray-600 bg-gray-100 rounded-md hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
          >
            Back
          </button>
          <button 
            onClick={handleContinue}
            className="px-6 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 shadow-sm"
          >
            Start Training
          </button>
        </div>
      </div>
    </div>
  );
}
