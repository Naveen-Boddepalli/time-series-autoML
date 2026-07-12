# Time-Series AutoML Studio

A 100% client-side web application for automated machine learning and forecasting on time-series data. 

**Privacy-First**: Your data never leaves your device. All calculations, data parsing, and model training (including heavy Python libraries and deep learning) are executed directly in your browser using Web Workers, WebAssembly (Pyodide), and WebGL/WebGPU (TensorFlow.js).

## Features
- **Exploratory Data Analysis (EDA)**: Interactive Plotly graphs and data preprocessing (filling missing values, outlier clipping).
- **Multiple Models**: 
  - ARIMA (via Statsmodels & Pyodide)
  - Gradient Boosting (via Scikit-Learn & Pyodide)
  - LSTM Neural Networks (via TensorFlow.js)
- **Zero Server Costs**: Because inference and training happen on the client's GPU/CPU, there is zero cloud compute cost.

## Extending the Platform
Want to add a new model (e.g. Facebook Prophet, Transformers) or expose hyperparameter tuning? 

👉 **[Read the guide on Adding New Models & Parameters](ADDING_MODELS.md)** 👈

## Running Locally

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.
