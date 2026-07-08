import * as Comlink from 'comlink';
import * as tf from '@tensorflow/tfjs';
// WebGPU and WebGL backends imported for tfjs
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-webgl';

const tfjsAPI = {
  async init(progressCallback?: (msg: string) => void) {
    if (progressCallback) progressCallback("Initializing TensorFlow.js backend...");
    try {
      // Attempt to set backend to webgpu for best performance, fallback to webgl
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
    return true;
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
  }
};

export type TfjsAPI = typeof tfjsAPI;
Comlink.expose(tfjsAPI);
