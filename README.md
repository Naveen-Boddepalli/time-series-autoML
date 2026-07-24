# Time-Series AutoML Studio

A robust, full-stack web application designed for automated machine learning and forecasting on time-series data. 

**Privacy-First & Edge-Powered**: Your data never leaves your device. All calculations, data parsing, and model training (including heavy Python libraries and deep learning) are executed directly in your browser using Web Workers, WebAssembly, and WebGL/WebGPU.

---

## 🚀 Tech Stack

### Frontend & Core
- **Next.js 16** (App Router, Serverless API Routes)
- **React 19**
- **TypeScript**
- **Tailwind CSS 4**
- **Zustand** (State Management)
- **Lucide React** (Icons)

### Machine Learning & Data Processing
- **TensorFlow.js** (Deep Learning via WebGL/WebGPU)
- **Pyodide** (WebAssembly Python environment for Scikit-Learn, Statsmodels, etc.)
- **Plotly.js** (Interactive data visualization)
- **Comlink** (Web Worker communication)
- **IndexedDB (idb)** (Client-side storage)

### Backend & Authentication
- **Neon Database (Serverless PostgreSQL)**
- **Jose** (JWT session management)
- **Bcryptjs** (Password hashing)

---

## ✨ Features

- **Secure Authentication**: Full sign-up, sign-in, and sign-out flow with hashed passwords and secure JWT HTTP-only cookies.
- **Exploratory Data Analysis (EDA)**: Interactive Plotly graphs and data preprocessing tools (filling missing values, outlier clipping).
- **Multiple Models Supported**: 
  - ARIMA (via Statsmodels & Pyodide)
  - Linear Regression & Random Forest (via Scikit-Learn & Pyodide)
  - Gradient Boosting (via Scikit-Learn & Pyodide)
  - LSTM, GRU, Bi-LSTM, & Transformers (via TensorFlow.js)
- **Zero Server Compute Costs**: Because inference and training happen on the client's GPU/CPU, there is zero cloud compute cost for model execution.
- **Background Processing**: ML tasks run in isolated Web Workers, ensuring the UI remains smooth and responsive during heavy training.

---

## 🛠️ Environment Variables

To run the application, you must set up the following environment variables in your `.env.local` file (or your Vercel project settings):

```env
# Connection string for your Serverless PostgreSQL Neon Database
DATABASE_URL="postgres://user:password@endpoint/db_name"

# A secure random string used to sign JWT session cookies
JWT_SECRET="your-super-secret-key"
```

---

## 💻 Running Locally

1. Install dependencies:
```bash
npm install
```

2. Create a `.env.local` file at the root of the project and add your `DATABASE_URL` and `JWT_SECRET`.

3. Run the development server:
```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

4. Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

---

## 🧩 Extending the Platform

Want to add a new model (e.g. Facebook Prophet, custom neural networks) or expose new hyperparameter tuning options? 

👉 **[Read the guide on Adding New Models & Parameters](ADDING_MODELS.md)** 👈
