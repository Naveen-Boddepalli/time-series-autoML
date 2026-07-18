import * as Comlink from 'comlink';
import * as tf from '@tensorflow/tfjs';
// WebGPU and WebGL backends imported for tfjs
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-webgl';

let isInitialized = false;
let initPromise: Promise<boolean> | null = null;

const tfjsAPI = {
  async init(progressCallback?: (msg: string) => void) {
    if (isInitialized) return true;
    
    if (!initPromise) {
      initPromise = (async () => {
        if (progressCallback) progressCallback("Initializing TensorFlow.js backend...");
        try {
          const success = await tf.setBackend('webgpu');
          if (success) {
            if (progressCallback) progressCallback("TensorFlow.js backend set to WebGPU.");
          }
        } catch (e) {
          console.warn("WebGPU not available, falling back to WebGL.", e);
          await tf.setBackend('webgl');
          if (progressCallback) progressCallback("TensorFlow.js backend set to WebGL.");
        }
        await tf.ready();
        isInitialized = true;
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

  async trainLSTM(
    X_train: number[][][], 
    y_train: number[][],
    epochs: number = 10,
    units: number = 50,
    progressCallback?: (epoch: number, logs: any) => void
  ) {
    await this.init();
    
    // Create simple LSTM model
    const model = tf.sequential();
    
    // Assumes input shape [timesteps, features]
    const timesteps = X_train[0].length;
    const features = X_train[0][0].length;
    
    model.add(tf.layers.lstm({
      units: units,
      inputShape: [timesteps, features],
      returnSequences: false
    }));
    model.add(tf.layers.dense({ units: 1 }));
    
    model.compile({
      optimizer: 'adam',
      loss: 'meanSquaredError'
    });
    
    const xs = tf.tensor3d(X_train);
    const ys = tf.tensor2d(y_train);
    
    await model.fit(xs, ys, {
      epochs: epochs,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (progressCallback) {
            progressCallback(epoch, logs);
          }
        }
      }
    });
    
    // Save model to indexedDB
    const modelName = 'lstm-model-' + Date.now();
    await model.save('indexeddb://' + modelName);
    
    // Cleanup tensors
    xs.dispose();
    ys.dispose();
    
    return {
      modelName: modelName
    };
  },
  
  async predictLSTM(modelName: string, X_test: number[][][]) {
    await this.init();
    const model = await tf.loadLayersModel('indexeddb://' + modelName);
    const xs = tf.tensor3d(X_test);
    const preds = model.predict(xs) as tf.Tensor;
    const predictions = await preds.array();
    
    // Cleanup
    xs.dispose();
    preds.dispose();
    
    return predictions;
  },

async trainGRU(
  X_train: number[][][],
  y_train: number[][],
  epochs: number = 10,
  units: number = 50,
  progressCallback?: (epoch: number, logs: any) => void
) {
  await this.init();

  const model = tf.sequential();
  const timesteps = X_train[0].length;
  const features = X_train[0][0].length;

  model.add(tf.layers.gru({
    units: units,
    inputShape: [timesteps, features],
    returnSequences: false
  }));
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

  const xs = tf.tensor3d(X_train);
  const ys = tf.tensor2d(y_train);

  await model.fit(xs, ys, {
    epochs: epochs,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (progressCallback) progressCallback(epoch, logs);
      }
    }
  });

  const modelName = 'gru-model-' + Date.now();
  await model.save('indexeddb://' + modelName);

  xs.dispose();
  ys.dispose();

  return { modelName };
},

async predictGRU(modelName: string, X_test: number[][][]) {
  await this.init();
  const model = await tf.loadLayersModel('indexeddb://' + modelName);
  const xs = tf.tensor3d(X_test);
  const preds = model.predict(xs) as tf.Tensor;
  const predictions = await preds.array();
  xs.dispose();
  preds.dispose();
  return predictions;
},

async trainLSTMGRUHybrid(
  X_train: number[][][],
  y_train: number[][],
  epochs: number = 10,
  units: number = 50,
  progressCallback?: (epoch: number, logs: any) => void
) {
  await this.init();

  const model = tf.sequential();
  const timesteps = X_train[0].length;
  const features = X_train[0][0].length;

  // LSTM layer extracts long-range temporal patterns, passing a full sequence to GRU
  model.add(tf.layers.lstm({
    units: units,
    inputShape: [timesteps, features],
    returnSequences: true
  }));
  // GRU layer compresses that sequence into a final representation
  model.add(tf.layers.gru({
    units: Math.max(8, Math.round(units / 2)),
    returnSequences: false
  }));
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

  const xs = tf.tensor3d(X_train);
  const ys = tf.tensor2d(y_train);

  await model.fit(xs, ys, {
    epochs: epochs,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (progressCallback) progressCallback(epoch, logs);
      }
    }
  });

  const modelName = 'lstm-gru-hybrid-model-' + Date.now();
  await model.save('indexeddb://' + modelName);

  xs.dispose();
  ys.dispose();

  return { modelName };
},

async predictLSTMGRUHybrid(modelName: string, X_test: number[][][]) {
  await this.init();
  const model = await tf.loadLayersModel('indexeddb://' + modelName);
  const xs = tf.tensor3d(X_test);
  const preds = model.predict(xs) as tf.Tensor;
  const predictions = await preds.array();
  xs.dispose();
  preds.dispose();
  return predictions;
},

async trainBiLSTM(
  X_train: number[][][],
  y_train: number[][],
  epochs: number = 10,
  units: number = 50,
  progressCallback?: (epoch: number, logs: any) => void
) {
  await this.init();

  const model = tf.sequential();
  const timesteps = X_train[0].length;
  const features = X_train[0][0].length;

  model.add(tf.layers.bidirectional({
    layer: tf.layers.lstm({ units: units, returnSequences: false }),
    inputShape: [timesteps, features],
    mergeMode: 'concat'
  }));
  model.add(tf.layers.dense({ units: 1 }));

  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

  const xs = tf.tensor3d(X_train);
  const ys = tf.tensor2d(y_train);

  await model.fit(xs, ys, {
    epochs: epochs,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (progressCallback) progressCallback(epoch, logs);
      }
    }
  });

  const modelName = 'bilstm-model-' + Date.now();
  await model.save('indexeddb://' + modelName);

  xs.dispose();
  ys.dispose();

  return { modelName };
},

async predictBiLSTM(modelName: string, X_test: number[][][]) {
  await this.init();
  const model = await tf.loadLayersModel('indexeddb://' + modelName);
  const xs = tf.tensor3d(X_test);
  const preds = model.predict(xs) as tf.Tensor;
  const predictions = await preds.array();
  xs.dispose();
  preds.dispose();
  return predictions;
},

async trainTransformer(
  X_train: number[][][],
  y_train: number[][],
  epochs: number = 10,
  headDim: number = 32,
  progressCallback?: (epoch: number, logs: any) => void
) {
  await this.init();

  const timesteps = X_train[0].length;
  const features = X_train[0][0].length;
  const ffDim = headDim * 2;

  const input = tf.input({ shape: [timesteps, features] });
  const query = tf.layers.dense({ units: headDim }).apply(input) as tf.SymbolicTensor;
  const key = tf.layers.dense({ units: headDim }).apply(input) as tf.SymbolicTensor;
  const value = tf.layers.dense({ units: headDim }).apply(input) as tf.SymbolicTensor;

  // Custom dot-product self-attention since tf.layers.attention is not available in tfjs
  const scores = tf.layers.dot({ axes: [2, 2] }).apply([query, key]) as tf.SymbolicTensor;
  const attnWeights = tf.layers.activation({ activation: 'softmax' }).apply(scores) as tf.SymbolicTensor;
  const attnOutput = tf.layers.dot({ axes: [2, 1] }).apply([attnWeights, value]) as tf.SymbolicTensor;
  
  const attnProjected = tf.layers.dense({ units: features }).apply(attnOutput) as tf.SymbolicTensor;

  const add1 = tf.layers.add().apply([input, attnProjected]) as tf.SymbolicTensor;
  const norm1 = tf.layers.layerNormalization().apply(add1) as tf.SymbolicTensor;

  const ff1 = tf.layers.dense({ units: ffDim, activation: 'relu' }).apply(norm1) as tf.SymbolicTensor;
  const ff2 = tf.layers.dense({ units: features }).apply(ff1) as tf.SymbolicTensor;
  const add2 = tf.layers.add().apply([norm1, ff2]) as tf.SymbolicTensor;
  const norm2 = tf.layers.layerNormalization().apply(add2) as tf.SymbolicTensor;

  const pooled = tf.layers.globalAveragePooling1d().apply(norm2) as tf.SymbolicTensor;
  const output = tf.layers.dense({ units: 1 }).apply(pooled) as tf.SymbolicTensor;

  const model = tf.model({ inputs: input, outputs: output });
  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

  const xs = tf.tensor3d(X_train);
  const ys = tf.tensor2d(y_train);

  await model.fit(xs, ys, {
    epochs: epochs,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (progressCallback) progressCallback(epoch, logs);
      }
    }
  });

  const modelName = 'transformer-model-' + Date.now();
  await model.save('indexeddb://' + modelName);

  xs.dispose();
  ys.dispose();

  return { modelName };
},

async predictTransformer(modelName: string, X_test: number[][][]) {
  await this.init();
  const model = await tf.loadLayersModel('indexeddb://' + modelName);
  const xs = tf.tensor3d(X_test);
  const preds = model.predict(xs) as tf.Tensor;
  const predictions = await preds.array();
  xs.dispose();
  preds.dispose();
  return predictions;
}
};

export type TfjsAPI = typeof tfjsAPI;
Comlink.expose(tfjsAPI);
