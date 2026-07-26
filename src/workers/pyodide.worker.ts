import * as Comlink from 'comlink';

let pyodideInstance: any = null;
let initPromise: Promise<boolean> | null = null;

const pyodideAPI = {
  async init(progressCallback?: (msg: string) => void) {
    if (pyodideInstance) return true;
    
    if (!initPromise) {
      initPromise = (async () => {
        if (progressCallback) progressCallback("Loading Pyodide runtime...");
        
        // Load Pyodide from CDN to avoid Next.js Webpack module worker issues
        (self as any).importScripts("https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js");
        const instance = await (self as any).loadPyodide();
        
        if (progressCallback) progressCallback("Loading pandas...");
        await instance.loadPackage("pandas");
        
        pyodideInstance = instance;
        if (progressCallback) progressCallback("Pyodide ready.");
        return true;
      })();
    }
    
    try {
      await initPromise;
      return true;
    } catch (e) {
      initPromise = null;
      throw e;
    }
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
        
        # Only select numeric columns for correlation and stats
        numeric_df = df.select_dtypes(include=['number'])
        
        corr_matrix = {}
        summary_stats = {}
        
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr().to_dict()
            summary_stats = numeric_df.describe().to_dict()
        
        result = {
            "columns": list(df.columns),
            "shape": list(df.shape),
            "dtypes": {k: str(v) for k, v in df.dtypes.items()},
            "preview": json.loads(preview_json),
            "correlation_matrix": corr_matrix,
            "summary_statistics": summary_stats
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

generate_preview()
`;
    const resultJson = await pyodideInstance.runPythonAsync(code);
    return JSON.parse(resultJson);
  },

  async trainModel(modelType: string, csvContent: string, targetCol: string, modelParams: any = {}, progressCallback?: (msg: string) => void) {
    if (!pyodideInstance) {
        if (progressCallback) progressCallback("Loading Pyodide runtime...");
        await new Promise(resolve => setTimeout(resolve, 100));
        await this.init(progressCallback);
    }
    
    // Lazy load massive ML libraries only when training, to keep uploads snappy!
    if (modelType === 'arima') {
        if (progressCallback) progressCallback("Downloading statsmodels library (10-20MB)...");
        await new Promise(resolve => setTimeout(resolve, 100));
        await pyodideInstance.loadPackage(['statsmodels']);
    } else if (modelType === 'boosting' || modelType === 'linear' || modelType === 'randomForest') {
        if (progressCallback) progressCallback("Downloading scikit-learn library (30-50MB)...");
        await new Promise(resolve => setTimeout(resolve, 100));
        await pyodideInstance.loadPackage(['scikit-learn']);
    }
    
    if (progressCallback) progressCallback("Fitting model...");
    await new Promise(resolve => setTimeout(resolve, 100));
    
    pyodideInstance.globals.set("csv_content", csvContent);
    pyodideInstance.globals.set("target_col", targetCol);
    pyodideInstance.globals.set("model_type", modelType);
    pyodideInstance.globals.set("split_fraction", modelParams.splitFraction || 0.2);
    pyodideInstance.globals.set("rf_estimators", modelParams.rfEstimators || 100);
    pyodideInstance.globals.set("rf_max_depth", modelParams.rfMaxDepth || 0);
    pyodideInstance.globals.set("rf_min_samples_split", modelParams.rfMinSamplesSplit || 2);
    
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
        # Use Nelder-Mead ('nm') optimizer instead of 'lbfgs' to prevent Fortran WASM infinite loops
        fitted = model.fit(method_kwargs={'method': 'nm', 'maxiter': 50})
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
    elif model_type in ['linear', 'randomForest']:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        df['lag1'] = df[target_col].shift(1)
        df['lag2'] = df[target_col].shift(2)
        df['lag3'] = df[target_col].shift(3)
        df = df.dropna()
        
        if len(df) == 0:
            raise ValueError("Not enough data to create lag features.")
            
        X = df.drop(columns=[target_col])
        X = X.select_dtypes(include=['number'])
        y = df[target_col]
        
        split_idx = int(len(df) * (1 - split_fraction))
        if split_idx <= 0 or split_idx >= len(df):
            split_idx = len(df) - 1 # Fallback to 1 element test set if too small
            
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        if model_type == 'linear':
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=int(rf_estimators),
                max_depth=int(rf_max_depth) if rf_max_depth > 0 else None,
                min_samples_split=int(rf_min_samples_split),
                random_state=42
            )
            
        model.fit(X_train, y_train)
        
        if len(y_test) > 0:
            preds = model.predict(X_test)
            MAE = mean_absolute_error(y_test, preds)
            MSE = mean_squared_error(y_test, preds)
            metrics = {"rmse": float(np.sqrt(MSE)), "mae": float(MAE)}
        else:
            metrics = {"rmse": 0.0, "mae": 0.0}
            
        # Forecast 10 steps iteratively
        last_row = df.iloc[-1].copy()
        current_y = last_row[target_col]
        current_lag1 = last_row['lag1']
        current_lag2 = last_row['lag2']
        
        forecast = []
        for _ in range(10):
            next_x = last_row.copy()
            next_x['lag1'] = current_y
            next_x['lag2'] = current_lag1
            next_x['lag3'] = current_lag2
            next_x_df = pd.DataFrame([next_x])[X.columns]
            
            pred = model.predict(next_x_df)[0]
            forecast.append(float(pred))
            
            current_lag2 = current_lag1
            current_lag1 = current_y
            current_y = pred
    
    elif model_type in ['logistic', 'lda', 'qda']:
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
        
        # We need to treat target as categorical, but first create lags if it is a sequence
        df['lag1'] = df[target_col].shift(1)
        df['lag2'] = df[target_col].shift(2)
        df['lag3'] = df[target_col].shift(3)
        df = df.dropna()
        
        if len(df) == 0:
            raise ValueError("Not enough data to create lag features.")
            
        unique_vals = df[target_col].nunique()
        if unique_vals > len(df) * 0.5:
            raise ValueError(f"Target has {unique_vals} unique values out of {len(df)} rows. This appears to be a continuous regression target, not categorical. Please switch to 'Time-Series Regression' mode or use a dataset with discrete classes.")
            
        label_encoder = LabelEncoder()
        df[target_col] = label_encoder.fit_transform(df[target_col])
        
        class_key_map = {str(index): str(label) for index, label in enumerate(label_encoder.classes_)}
        
        X = df.drop(columns=[target_col])
        X = X.select_dtypes(include=['number'])
        y = df[target_col]
        
        split_idx = int(len(df) * (1 - split_fraction))
        if split_idx <= 0 or split_idx >= len(df):
            split_idx = len(df) - 1
            
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        if model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            # use liblinear since lbfgs can be tricky with small datasets, or lbfgs with small max_iter
            model = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000)
        elif model_type == 'lda':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            model = LinearDiscriminantAnalysis()
        elif model_type == 'qda':
            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
            model = QuadraticDiscriminantAnalysis()
            
        model.fit(X_train, y_train)
        
        # Save model globally so we can run live predictions later
        if 'models_cache' not in globals():
            global models_cache
            models_cache = {}
            
        models_cache[model_type] = {
            'model': model,
            'features': list(X.columns),
            'class_map': class_key_map,
            'last_row': df.iloc[-1].to_dict()
        }
        
        if len(y_test) > 0:
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average='weighted', zero_division=0)
            rec = recall_score(y_test, preds, average='weighted', zero_division=0)
            conf = confusion_matrix(y_test, preds).tolist()
            metrics = {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "confusion_matrix": conf}
        else:
            metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "confusion_matrix": []}
            
        # We don't return a 10-step forecast for classification like we do for regression.
        forecast = []
        
        return json.dumps({
            "metrics": metrics,
            "forecast": forecast,
            "classKeyMap": class_key_map
        })
    
    return json.dumps({
        "metrics": metrics,
        "forecast": forecast
    })

run_training()
`;
    const resultJson = await pyodideInstance.runPythonAsync(code);
    return JSON.parse(resultJson);
  },

  async predictUserEntry(modelType: string, entry: Record<string, number>) {
    if (!pyodideInstance) throw new Error("Pyodide is not initialized");
    
    pyodideInstance.globals.set("pred_model", modelType);
    pyodideInstance.globals.set("user_entry_json", JSON.stringify(entry));
    
    const code = `
import json
import pandas as pd
import numpy as np

def run_prediction():
    if 'models_cache' not in globals() or pred_model not in models_cache:
        raise ValueError(f"Model {pred_model} is not loaded in memory.")
        
    cache = models_cache[pred_model]
    model = cache['model']
    features = cache['features']
    class_map = cache['class_map']
    
    # Merge user entry with last row lags if user entry is missing them
    user_entry = json.loads(user_entry_json)
    merged_entry = {}
    
    for f in features:
        if f in user_entry:
            merged_entry[f] = user_entry[f]
        else:
            merged_entry[f] = cache['last_row'].get(f, 0)
            
    user_df = pd.DataFrame([merged_entry])[features]
    
    pred_code = int(model.predict(user_df)[0])
    pred_label = class_map[str(pred_code)]
    
    probs = {}
    if hasattr(model, 'predict_proba'):
        prob_array = model.predict_proba(user_df)[0]
        # match probabilities to classes
        for i, p in enumerate(prob_array):
            if str(i) in class_map:
                probs[class_map[str(i)]] = float(p)
                
    return json.dumps({
        "prediction": pred_label,
        "probabilities": probs
    })

run_prediction()
`;
    const resultJson = await pyodideInstance.runPythonAsync(code);
    return JSON.parse(resultJson);
  }
};

export type PyodideAPI = typeof pyodideAPI;
Comlink.expose(pyodideAPI);
