import * as Comlink from 'comlink';

let pyodideInstance: any = null;

const pyodideAPI = {
  async init(progressCallback?: (msg: string) => void) {
    if (!pyodideInstance) {
      if (progressCallback) progressCallback("Loading Pyodide runtime...");
      
      // Load Pyodide from CDN to avoid Next.js Webpack module worker issues
      (self as any).importScripts("https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js");
      pyodideInstance = await (self as any).loadPyodide();
      
      if (progressCallback) progressCallback("Loading pandas...");
      await pyodideInstance.loadPackage(["pandas"]);
      if (progressCallback) progressCallback("Pyodide ready.");
    }
    return true;
  },

  async runPython(code: string) {
    if (!pyodideInstance) await this.init();
    const result = await pyodideInstance.runPythonAsync(code);
    return result;
  },

  async getPreview(csvContent: string) {
    if (!pyodideInstance) await this.init();
    
    // Set the variable in Python global scope
    pyodideInstance.globals.set("csv_content", csvContent);
    
    const code = `
import pandas as pd
import io
import json

def generate_preview():
    try:
        df = pd.read_csv(io.StringIO(csv_content))
        
        preview_df = df.head(100)
        
        preview_json = preview_df.to_json(orient='records')
        
        result = {
            "columns": list(df.columns),
            "shape": list(df.shape),
            "dtypes": {k: str(v) for k, v in df.dtypes.items()},
            "preview": json.loads(preview_json)
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

generate_preview()
`;
    const resultJson = await pyodideInstance.runPythonAsync(code);
    return JSON.parse(resultJson);
  },

  async trainModel(modelType: string, csvContent: string, targetCol: string, progressCallback?: (msg: string) => void) {
    if (!pyodideInstance) await this.init(progressCallback);
    
    // Lazy load massive ML libraries only when training, to keep uploads snappy!
    if (modelType === 'arima') {
        if (progressCallback) await progressCallback("Downloading statsmodels library (10-20MB)...");
        await pyodideInstance.loadPackage(['statsmodels']);
    } else if (modelType === 'boosting') {
        if (progressCallback) await progressCallback("Downloading scikit-learn library (30-50MB)...");
        await pyodideInstance.loadPackage(['scikit-learn']);
    }
    
    if (progressCallback) await progressCallback("Fitting model...");
    
    pyodideInstance.globals.set("csv_content", csvContent);
    pyodideInstance.globals.set("target_col", targetCol);
    pyodideInstance.globals.set("model_type", modelType);
    
    const code = `
import pandas as pd
import io
import json
import numpy as np

def run_training():
    df = pd.read_csv(io.StringIO(csv_content))
    
    # Enforce target column is numeric
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    
    # basic preprocessing: ffill
    df = df.ffill().bfill()
    
    # Drop any remaining NaNs
    df = df.dropna(subset=[target_col])
    if len(df) == 0:
        raise ValueError(f"Target column '{target_col}' contains no valid numeric data to forecast.")
        
    # LIMIT DATASET SIZE FOR IN-BROWSER PERFORMANCE (ARIMA gets very slow > 500 rows)
    if len(df) > 500:
        df = df.tail(500).reset_index(drop=True)
        
    y = df[target_col].values
    
    metrics = {}
    forecast = []
    
    if model_type == 'arima':
        from statsmodels.tsa.arima.model import ARIMA
        # simple (1,1,1) for speed
        model = ARIMA(y, order=(1,1,1))
        # Limit maxiter to prevent the WASM thread from hanging infinitely on heavy optimizations
        fitted = model.fit(maxiter=20)
        preds = fitted.fittedvalues
        forecast = fitted.forecast(steps=10).tolist()
        mse = np.mean((y - preds)**2)
        metrics = {"rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(y - preds)))}
    elif model_type == 'boosting':
        from sklearn.ensemble import HistGradientBoostingRegressor
        # create simple lags
        X = np.arange(len(y)).reshape(-1, 1)
        model = HistGradientBoostingRegressor()
        model.fit(X, y)
        preds = model.predict(X)
        X_future = np.arange(len(y), len(y)+10).reshape(-1, 1)
        forecast = model.predict(X_future).tolist()
        mse = np.mean((y - preds)**2)
        metrics = {"rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(y - preds)))}
    
    return json.dumps({
        "metrics": metrics,
        "forecast": forecast
    })

run_training()
`;
    const resultJson = await pyodideInstance.runPythonAsync(code);
    return JSON.parse(resultJson);
  }
};

export type PyodideAPI = typeof pyodideAPI;
Comlink.expose(pyodideAPI);
