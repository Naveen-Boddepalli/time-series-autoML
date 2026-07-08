'use client';

import React, { useState } from 'react';
import { useStore } from '../store/useStore';
import dynamic from 'next/dynamic';
import { Target, AlertCircle, Settings2, ChevronDown } from 'lucide-react';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export default function StepEDA() {
  const { datasetPreview, setEDAOptions, setStep, targetColumn: storeTargetColumn } = useStore();
  
  const columns = datasetPreview?.columns || [];
  const previewData = datasetPreview?.preview || [];
  const dtypes = datasetPreview?.dtypes || {};

  // Auto-select the first numeric column instead of just the first column (which is often a Date string)
  const defaultTarget = columns.find((c: string) => dtypes[c]?.includes('int') || dtypes[c]?.includes('float')) || columns[0] || '';

  const [targetColumn, setTargetColumn] = useState<string>(storeTargetColumn || defaultTarget);
  const [missingValue, setMissingValue] = useState<string>('ffill');
  const [outlier, setOutlier] = useState<string>('none');

  const handleContinue = () => {
    setEDAOptions(targetColumn, missingValue, outlier);
    setStep('MODEL');
  };

  const plotData = previewData.map((row: any, i: number) => ({
    x: i,
    y: row[targetColumn]
  }));

  return (
    <div className="max-w-4xl mx-auto space-y-8 pb-12">
      <div className="bg-white dark:bg-gray-900 p-8 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-800">
        
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100 mb-2">Exploratory Data Analysis</h2>
          <p className="text-gray-500 dark:text-gray-400">Select the target column you want to forecast to preview the raw time-series data.</p>
        </div>
        
        <div className="bg-blue-50/50 dark:bg-blue-900/10 border border-blue-100 dark:border-blue-800/50 rounded-xl p-6 mb-8">
          <label className="flex items-center text-sm font-semibold text-gray-800 dark:text-gray-200 mb-3">
            <Target className="w-4 h-4 mr-2 text-blue-500" />
            Target Column to Forecast
          </label>
          <div className="relative max-w-sm">
            <select 
              className="w-full appearance-none bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 text-gray-900 dark:text-white rounded-lg pl-4 pr-10 py-3 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all font-medium cursor-pointer"
              value={targetColumn} 
              onChange={e => setTargetColumn(e.target.value)}
            >
              {columns.map((c: string) => <option key={c} value={c}>{c}</option>)}
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-gray-500 dark:text-gray-400">
              <ChevronDown className="w-5 h-5" />
            </div>
          </div>
        </div>

        <div className="mb-10">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200">Raw Data Preview</h3>
            <span className="bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 px-3 py-1 text-xs font-medium rounded-full border border-gray-200 dark:border-gray-700">
              Showing first 100 rows
            </span>
          </div>
          <div className="border border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden h-[300px] bg-gray-50 dark:bg-gray-800/30">
             <Plot
              data={[
                {
                  x: plotData.map((d: any) => d.x),
                  y: plotData.map((d: any) => d.y),
                  type: 'scatter',
                  mode: 'lines+markers',
                  marker: {color: '#3b82f6', size: 4},
                  line: {color: '#60a5fa', width: 2}
                },
              ]}
              layout={{
                autosize: true, 
                margin: { t: 20, r: 20, b: 50, l: 60 },
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                xaxis: { 
                  gridcolor: '#e2e8f0',
                  title: { text: 'Time Step (Index)', font: { size: 14 } } 
                },
                yaxis: { 
                  gridcolor: '#e2e8f0',
                  title: { text: targetColumn || 'Value', font: { size: 14 } }
                }
              }}
              style={{width: '100%', height: '100%'}}
              useResizeHandler={true}
            />
          </div>
        </div>

        <div className="border-t border-gray-200 dark:border-gray-700 pt-8 mb-8">
          <div className="mb-6">
            <h3 className="text-lg font-semibold flex items-center text-gray-800 dark:text-gray-200">
              <Settings2 className="w-5 h-5 mr-2 text-gray-500" />
              Preprocessing Strategies
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Configure how the data will be cleaned before training. These strategies will be applied automatically by the pipeline.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="flex items-center text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                <AlertCircle className="w-4 h-4 mr-2 text-orange-500" />
                Missing Value Strategy
              </label>
              <div className="relative">
                <select 
                  className="w-full appearance-none bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 text-gray-900 dark:text-white rounded-lg pl-4 pr-10 py-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all cursor-pointer"
                  value={missingValue} 
                  onChange={e => setMissingValue(e.target.value)}
                >
                  <option value="ffill">Forward Fill (Recommended)</option>
                  <option value="bfill">Backward Fill</option>
                  <option value="mean">Mean Imputation</option>
                  <option value="drop">Drop Rows</option>
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-gray-500 dark:text-gray-400">
                  <ChevronDown className="w-4 h-4" />
                </div>
              </div>
            </div>
            
            <div>
              <label className="flex items-center text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                <AlertCircle className="w-4 h-4 mr-2 text-red-500" />
                Outlier Strategy
              </label>
              <div className="relative">
                <select 
                  className="w-full appearance-none bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-700 text-gray-900 dark:text-white rounded-lg pl-4 pr-10 py-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all cursor-pointer"
                  value={outlier} 
                  onChange={e => setOutlier(e.target.value)}
                >
                  <option value="none">None (Keep All Data)</option>
                  <option value="clip">Clip (IQR Method)</option>
                  <option value="drop">Drop (IQR Method)</option>
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-gray-500 dark:text-gray-400">
                  <ChevronDown className="w-4 h-4" />
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="flex justify-between items-center pt-6 border-t border-gray-100 dark:border-gray-800">
          <button 
            onClick={() => setStep('UPLOAD')}
            className="px-6 py-2.5 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700 transition-colors"
          >
            Back
          </button>
          <button 
            onClick={handleContinue}
            className="px-8 py-2.5 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 shadow-sm transition-colors"
          >
            Continue to Models
          </button>
        </div>
      </div>
    </div>
  );
}
