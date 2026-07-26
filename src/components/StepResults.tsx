'use client';

import React, { useState } from 'react';
import { useStore } from '../store/useStore';
import { Download, RefreshCw, BarChart2, ImageIcon, Activity, Key, Target, Layers } from 'lucide-react';
import dynamic from 'next/dynamic';
import { getPyodideAPI } from '../lib/workerHelper';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export default function StepResults() {
  const { modelResults, datasetPreview, targetColumn, taskType, reset } = useStore();
  
  const availableModels = Object.keys(modelResults);
  const [selectedModel, setSelectedModel] = useState<string>(availableModels[0] || '');
  const [graphDiv, setGraphDiv] = useState<any>(null);
  
  // Live prediction state
  const [userEntry, setUserEntry] = useState<Record<string, string>>({});
  const [livePrediction, setLivePrediction] = useState<{ prediction: string, probabilities: Record<string, number> } | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);

  const previewData = datasetPreview?.preview || [];
  const historicalY = previewData.map((row: any) => row[targetColumn as string]);
  const historicalX = previewData.map((_: any, i: number) => i);
  
  const currentResult = selectedModel && modelResults[selectedModel] ? modelResults[selectedModel] : null;
  const currentMetrics = currentResult?.metrics;
  const forecastY = currentResult?.forecast || [];
  const forecastX = forecastY.map((_: any, i: number) => historicalX.length + i);

  const featureColumns = datasetPreview?.columns.filter(c => c !== targetColumn) || [];

  const handleDownload = () => {
    if (!selectedModel || !currentResult) return;
    
    let csvContent = "type,index,value\n";
    
    historicalY.forEach((val: number, i: number) => {
      csvContent += `historical,${i},${val}\n`;
    });
    
    forecastY.forEach((val: number, i: number) => {
      csvContent += `forecast,${historicalX.length + i},${val}\n`;
    });
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedModel}_forecast.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleDownloadGraph = async () => {
    if (graphDiv) {
      try {
        const Plotly = (await import('plotly.js-dist-min')).default || await import('plotly.js-dist-min');
        Plotly.downloadImage(graphDiv, { format: 'png', filename: `forecast_${selectedModel}` });
      } catch (err) {
        console.error('Failed to save graph', err);
      }
    }
  };

  const handlePredict = async () => {
    if (!selectedModel) return;
    setIsPredicting(true);
    setPredictError(null);
    try {
      const pyodide = getPyodideAPI();
      if (!pyodide) throw new Error("Pyodide worker unavailable");
      
      const parsedEntry: Record<string, number> = {};
      for (const [k, v] of Object.entries(userEntry)) {
        parsedEntry[k] = parseFloat(v);
        if (isNaN(parsedEntry[k])) parsedEntry[k] = 0;
      }
      
      const result = await pyodide.predictUserEntry(selectedModel, parsedEntry);
      setLivePrediction(result);
    } catch (e: any) {
      setPredictError(e.message || "Failed to predict");
    } finally {
      setIsPredicting(false);
    }
  };

  const getModelName = (key: string) => {
    if (key === 'arima') return 'ARIMA';
    if (key === 'boosting') return 'Gradient Boosting';
    if (key === 'lstm') return 'LSTM';
    if (key === 'logistic') return 'Logistic Regression';
    if (key === 'lda') return 'Linear Discriminant Analysis';
    if (key === 'qda') return 'Quadratic Discriminant Analysis';
    return key;
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8 pb-12">
      <div className="bg-white dark:bg-gray-900 p-6 rounded-xl shadow-sm border border-gray-100 dark:border-gray-800">
        
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
          <div>
            <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">
              {taskType === 'regression' ? 'Forecast Results' : 'Classification Results'}
            </h2>
            <p className="text-gray-500 dark:text-gray-400">
              Compare model performance and interact with predictions
            </p>
          </div>
          
          <div className="flex items-center space-x-3 bg-gray-50 dark:bg-gray-800 p-1.5 rounded-lg border border-gray-200 dark:border-gray-700 overflow-x-auto">
            {availableModels.map(model => (
              <button
                key={model}
                onClick={() => setSelectedModel(model)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors whitespace-nowrap ${
                  selectedModel === model 
                    ? 'bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-400 shadow-sm border border-gray-200 dark:border-gray-600' 
                    : 'text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700 hover:text-gray-900 dark:hover:text-gray-200 border border-transparent'
                }`}
              >
                {getModelName(model)}
              </button>
            ))}
          </div>
        </div>

        {/* REGRESSION VIEW */}
        {taskType === 'regression' && currentMetrics && (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
              <div className="p-4 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800/30">
                <div className="text-sm text-blue-600 dark:text-blue-400 font-medium mb-1">RMSE</div>
                <div className="text-2xl font-bold text-blue-900 dark:text-blue-100">{currentMetrics.rmse?.toFixed(4)}</div>
              </div>
              <div className="p-4 rounded-lg bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-100 dark:border-indigo-800/30">
                <div className="text-sm text-indigo-600 dark:text-indigo-400 font-medium mb-1">MAE</div>
                <div className="text-2xl font-bold text-indigo-900 dark:text-indigo-100">{currentMetrics.mae?.toFixed(4)}</div>
              </div>
              <div className="p-4 rounded-lg bg-gray-50 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700 col-span-2 flex items-center">
                 <BarChart2 className="w-8 h-8 text-gray-400 mr-4" />
                 <div>
                   <div className="text-sm text-gray-500 dark:text-gray-400 font-medium mb-1">Forecast Horizon</div>
                   <div className="text-lg font-bold text-gray-800 dark:text-gray-200">{forecastY.length} Steps Ahead</div>
                 </div>
              </div>
            </div>

            <div id="results-plot-container" className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden h-[400px] mb-8 bg-gray-50 dark:bg-gray-800/30 relative">
              <Plot
                  data={[
                    {
                      x: historicalX,
                      y: historicalY,
                      type: 'scatter',
                      mode: 'lines+markers',
                      name: 'Historical',
                      marker: {color: '#64748b', size: 4}, // slate-500
                      line: {color: '#94a3b8', width: 2} // slate-400
                    },
                    {
                      x: [historicalX[historicalX.length - 1], ...forecastX], // connect the lines visually
                      y: [historicalY[historicalY.length - 1], ...forecastY],
                      type: 'scatter',
                      mode: 'lines+markers',
                      name: getModelName(selectedModel) + ' Forecast',
                      marker: {color: '#2563eb', size: 6}, // blue-600
                      line: {color: '#3b82f6', width: 3, dash: 'dot'} // blue-500
                    }
                  ]}
                  layout={{
                    autosize: true, 
                    margin: { t: 30, r: 20, b: 50, l: 60 },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent',
                    showlegend: true,
                    legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1 },
                    xaxis: { 
                      gridcolor: '#e2e8f0', 
                      title: { text: 'Time Step (Index)', font: { size: 14 } } 
                    },
                    yaxis: { 
                      gridcolor: '#e2e8f0', 
                      title: { text: targetColumn || 'Target Value', font: { size: 14 } } 
                    }
                  }}
                  config={{
                    displayModeBar: true,
                    toImageButtonOptions: { filename: `forecast_${selectedModel}`, format: 'png', scale: 2 }
                  }}
                  onInitialized={(figure, graphDiv) => setGraphDiv(graphDiv)}
                  onUpdate={(figure, graphDiv) => setGraphDiv(graphDiv)}
                  style={{width: '100%', height: '100%'}}
                  useResizeHandler={true}
                />
            </div>

            <div className="flex justify-end mb-8">
                <button 
                  onClick={handleDownloadGraph}
                  className="flex items-center space-x-2 px-4 py-2 text-sm font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 rounded-lg dark:bg-blue-900/30 dark:text-blue-300 dark:hover:bg-blue-900/50 transition-colors border border-blue-200 dark:border-blue-800/50"
                >
                  <ImageIcon className="w-4 h-4" />
                  <span>Save Graph Image</span>
                </button>
            </div>
          </>
        )}

        {/* CLASSIFICATION VIEW */}
        {taskType === 'classification' && currentMetrics && (
          <div className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-100 dark:border-green-800/30 flex items-center justify-between">
                <div>
                  <div className="text-sm text-green-600 dark:text-green-400 font-medium mb-1">Accuracy</div>
                  <div className="text-2xl font-bold text-green-900 dark:text-green-100">{(currentMetrics.accuracy * 100).toFixed(1)}%</div>
                </div>
                <Activity className="w-8 h-8 text-green-500/50" />
              </div>
              <div className="p-4 rounded-lg bg-purple-50 dark:bg-purple-900/20 border border-purple-100 dark:border-purple-800/30 flex items-center justify-between">
                <div>
                  <div className="text-sm text-purple-600 dark:text-purple-400 font-medium mb-1">Precision</div>
                  <div className="text-2xl font-bold text-purple-900 dark:text-purple-100">{(currentMetrics.precision * 100).toFixed(1)}%</div>
                </div>
                <Target className="w-8 h-8 text-purple-500/50" />
              </div>
              <div className="p-4 rounded-lg bg-orange-50 dark:bg-orange-900/20 border border-orange-100 dark:border-orange-800/30 flex items-center justify-between">
                <div>
                  <div className="text-sm text-orange-600 dark:text-orange-400 font-medium mb-1">Recall</div>
                  <div className="text-2xl font-bold text-orange-900 dark:text-orange-100">{(currentMetrics.recall * 100).toFixed(1)}%</div>
                </div>
                <Layers className="w-8 h-8 text-orange-500/50" />
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Live Prediction UI */}
              <div className="bg-gray-50 dark:bg-gray-800/30 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-bold text-gray-800 dark:text-gray-100 mb-4 flex items-center">
                  <Key className="w-5 h-5 mr-2 text-blue-500" />
                  Live Prediction
                </h3>
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
                  Enter feature values to predict the class label using the trained model. Blank fields will use the last known values (helpful for lags).
                </p>

                <div className="grid grid-cols-2 gap-4 max-h-[300px] overflow-y-auto pr-2 mb-6">
                  {featureColumns.map(col => (
                    <div key={col}>
                      <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">{col}</label>
                      <input 
                        type="number"
                        placeholder="0.0"
                        value={userEntry[col] || ''}
                        onChange={(e) => setUserEntry(prev => ({...prev, [col]: e.target.value}))}
                        className="w-full text-sm p-2 border border-gray-300 rounded dark:bg-gray-900 dark:border-gray-600 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                      />
                    </div>
                  ))}
                </div>

                <button 
                  onClick={handlePredict}
                  disabled={isPredicting}
                  className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg disabled:opacity-50 transition-colors shadow-sm"
                >
                  {isPredicting ? 'Predicting...' : 'Predict Class'}
                </button>

                {predictError && <div className="mt-4 text-sm text-red-500">{predictError}</div>}

                {livePrediction && (
                  <div className="mt-6 p-4 rounded-lg border border-blue-200 bg-blue-50 dark:bg-blue-900/20 dark:border-blue-800">
                    <div className="text-sm text-blue-600 dark:text-blue-400 font-semibold mb-1">Predicted Label</div>
                    <div className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                      {livePrediction.prediction}
                    </div>
                    
                    <div className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-wider">Class Probabilities</div>
                    <div className="space-y-2">
                      {Object.entries(livePrediction.probabilities).map(([className, prob]) => (
                        <div key={className}>
                          <div className="flex justify-between text-sm mb-1">
                            <span className="text-gray-700 dark:text-gray-300">{className}</span>
                            <span className="font-medium dark:text-gray-300">{(prob * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                            <div className="bg-blue-500 h-1.5 rounded-full" style={{ width: `${prob * 100}%` }}></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Confusion Matrix */}
              <div className="bg-white dark:bg-gray-800/50 rounded-xl p-6 border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-bold text-gray-800 dark:text-gray-100 mb-4">Confusion Matrix</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
                  Performance on the test split.
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm text-left">
                    <thead className="bg-gray-50 dark:bg-gray-700 text-gray-600 dark:text-gray-300">
                      <tr>
                        <th className="p-3 border-b dark:border-gray-600">True \\ Pred</th>
                        {currentMetrics.confusion_matrix?.[0]?.map((_: any, i: number) => (
                          <th key={i} className="p-3 border-b dark:border-gray-600">{currentResult?.classKeyMap?.[i.toString()] || i}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {currentMetrics.confusion_matrix?.map((row: number[], i: number) => (
                        <tr key={i} className="border-b border-gray-100 dark:border-gray-700">
                          <th className="p-3 bg-gray-50 dark:bg-gray-800 font-medium text-gray-700 dark:text-gray-300">
                            {currentResult?.classKeyMap?.[i.toString()] || i}
                          </th>
                          {row.map((val: number, j: number) => {
                            // simple heatmap color intensity based on value
                            const maxVal = Math.max(...currentMetrics.confusion_matrix.flat());
                            const intensity = val / (maxVal || 1);
                            const bgOpacity = Math.max(0.05, intensity * 0.8);
                            return (
                              <td key={j} className="p-3 text-center" style={{ backgroundColor: `rgba(59, 130, 246, ${bgOpacity})` }}>
                                <span className={intensity > 0.5 ? 'text-white font-bold' : 'text-gray-800 dark:text-gray-200 font-medium'}>
                                  {val}
                                </span>
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-8">
           <button 
            onClick={handleDownload}
            disabled={!selectedModel || taskType === 'classification'}
            className="flex items-center justify-center space-x-2 p-3 rounded-lg border-2 border-blue-500 bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:border-blue-700 dark:text-blue-300 hover:bg-blue-100 dark:hover:bg-blue-900/50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Download className="w-5 h-5" />
            <span className="font-semibold">Download Forecast (CSV)</span>
          </button>

           <button 
            onClick={reset}
            className="flex items-center justify-center space-x-2 p-3 rounded-lg border-2 border-gray-300 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          >
            <RefreshCw className="w-5 h-5" />
            <span className="font-semibold">Start New Project</span>
          </button>
        </div>
      </div>
    </div>
  );
}
