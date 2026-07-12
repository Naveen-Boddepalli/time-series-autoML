# How to Add New Models & Parameters

This project uses a 100% client-side architecture where all Machine Learning happens in the user's browser using Web Workers. There are two primary engines available:
1. **Pyodide (WebAssembly Python)**: Best for traditional statistical and ML models (ARIMA, XGBoost, Scikit-Learn).
2. **TensorFlow.js (TF.js)**: Best for deep learning and neural networks (LSTM, Transformers).

Follow these steps to add a new model to the application.

---

## Step 1: Update the Model Selection UI
Users need to be able to select your new model from the UI.
1. Open `src/components/StepModelConfig.tsx`.
2. Locate the grid of model toggles inside the return statement.
3. Add a new clickable card for your model. For example:

```tsx
<div 
  onClick={() => toggleModel('prophet')}
  className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${
    modelsToTrain.includes('prophet') ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
  }`}
>
  <div className="flex items-center space-x-3 mb-2">
    <div className={`p-2 rounded-lg ${modelsToTrain.includes('prophet') ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-500'}`}>
      <Activity className="w-5 h-5" />
    </div>
    <h3 className="font-semibold text-gray-800">Prophet</h3>
  </div>
  <p className="text-sm text-gray-500">Facebook's additive regression model for forecasting.</p>
</div>
```

---

## Step 2: Route the Model in the Training Step
Next, tell the application which Web Worker should handle your model.
1. Open `src/components/StepTrain.tsx`.
2. Find the `runTraining` function and the `if (model === ...)` blocks.
3. Route your new model to the appropriate worker (Pyodide or TF.js).

```tsx
if (model === 'arima' || model === 'boosting' || model === 'prophet') {
  const pyodide = getPyodideAPI();
  const result = await pyodide.trainModel(model, rawData.rawContent, targetColumn, Comlink.proxy(...));
  // ...
}
```

---

## Step 3: Implement the Math in the Web Worker

Depending on which engine you chose in Step 2, you'll implement the logic in either the Python or JavaScript worker.

### Option A: Adding a Python Model (Pyodide)
If your model requires Python libraries (e.g., `scikit-learn`, `statsmodels`), implement it in `src/workers/pyodide.worker.ts`.

1. **Load required dependencies**: In the `trainModel` function, ensure any large Python packages your model needs are downloaded before execution.
```typescript
if (modelType === 'prophet') {
    if (progressCallback) progressCallback("Downloading prophet library...");
    await new Promise(resolve => setTimeout(resolve, 100)); // Crucial to prevent UI deadlocks!
    await pyodideInstance.loadPackage(['prophet']);
}
```
*(Note: Always include a `setTimeout(100)` before heavy WebAssembly loads to allow the UI to paint the progress message).*

2. **Write the Python script**: Scroll down to the `code = \`...\`` block and add an `elif` statement for your model.
```python
elif model_type == 'prophet':
    from prophet import Prophet
    # Format data for Prophet (requires 'ds' and 'y' columns)
    prophet_df = pd.DataFrame({'ds': df.index, 'y': y})
    model = Prophet()
    model.fit(prophet_df)
    
    # Generate predictions
    future = model.make_future_dataframe(periods=10)
    forecast_df = model.predict(future)
    
    preds = forecast_df['yhat'][:-10].values
    forecast = forecast_df['yhat'][-10:].values.tolist()
```

### Option B: Adding a Deep Learning Model (TensorFlow.js)
If your model is a neural network, implement it in `src/workers/tfjs.worker.ts`.

1. Add a new asynchronous function to the `TfjsAPI` object.
2. Use `@tensorflow/tfjs` to build and compile your model.

```typescript
async trainMyNewModel(xData, yData, progressCallback) {
  const tf = await import('@tensorflow/tfjs');
  
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [xData[0].length] }));
  model.add(tf.layers.dense({ units: 1 }));
  
  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });
  
  // Train the model
  await model.fit(tf.tensor(xData), tf.tensor(yData), {
    epochs: 20,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (progressCallback) progressCallback(epoch, logs);
      }
    }
  });
  
  // Generate predictions...
}
```

---

## Adding Adjustable Parameters
To allow users to tweak hyperparameters (e.g., Learning Rate, Max Iterations):

1. **Add to Global Store**: Update `src/store/useStore.ts` to hold your new parameters.
2. **Add to UI**: In `src/components/StepModelConfig.tsx`, add range sliders or inputs that update the store.
3. **Pass to Worker**: In `src/components/StepTrain.tsx`, pass the parameters from the store into the `pyodide.trainModel()` or `tfjs.trainLSTM()` function calls.
4. **Inject into Python/TF**: 
   - For Python: Use `pyodideInstance.globals.set("my_param", paramValue)` in `pyodide.worker.ts` so the Python script can read it.
   - For TF.js: Pass the parameter directly into the `model.compile` or `model.fit` functions in `tfjs.worker.ts`.
