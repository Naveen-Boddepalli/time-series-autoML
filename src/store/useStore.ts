import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type WizardStep = 'UPLOAD' | 'EDA' | 'MODEL' | 'TRAIN' | 'RESULTS';

export interface DatasetPreview {
  columns: string[];
  shape: number[];
  dtypes: Record<string, string>;
  preview: any[];
  correlation_matrix?: Record<string, Record<string, number>>;
  summary_statistics?: Record<string, Record<string, number>>;
}

interface AppState {
  currentStep: WizardStep;
  datasetId: string | null;
  datasetPreview: DatasetPreview | null; // Preview from pandas
  targetColumn: string | null;
  preprocessing: {
    missingValueStrategy: string;
    outlierStrategy: string;
  };
  taskType: 'regression' | 'classification';
  modelsToTrain: string[];
  modelResults: Record<string, { metrics: any, forecast?: number[], classKeyMap?: Record<string, string> }>;
  modelParams: {
    splitFraction: number;
    rfEstimators: number;
    rfMaxDepth: number;
    rfMinSamplesSplit: number;
  };
  
  setStep: (step: WizardStep) => void;
  setDataset: (id: string, preview: DatasetPreview) => void;
  setEDAOptions: (target: string, missing: string, outliers: string) => void;
  setModelsToTrain: (models: string[]) => void;
  setModelResults: (results: Record<string, { metrics: any, forecast?: number[], classKeyMap?: Record<string, string> }>) => void;
  setModelParams: (params: Partial<AppState['modelParams']>) => void;
  setTaskType: (type: 'regression' | 'classification') => void;
  reset: () => void;
}

export const useStore = create<AppState>()(
  persist(
    (set) => ({
      currentStep: 'UPLOAD',
      datasetId: null,
      datasetPreview: null,
      targetColumn: null,
      preprocessing: {
        missingValueStrategy: 'ffill',
        outlierStrategy: 'none'
      },
      taskType: 'regression',
      modelsToTrain: [],
      modelResults: {},
      modelParams: {
        splitFraction: 0.2,
        rfEstimators: 100,
        rfMaxDepth: 0, // 0 means None
        rfMinSamplesSplit: 2,
      },
      
      setStep: (step) => set({ currentStep: step }),
      setDataset: (id, preview) => set({ datasetId: id, datasetPreview: preview, targetColumn: null }),
      setEDAOptions: (target, missing, outliers) => set({
        targetColumn: target,
        preprocessing: { missingValueStrategy: missing, outlierStrategy: outliers }
      }),
      setTaskType: (type: 'regression' | 'classification') => set({ taskType: type }),
      setModelsToTrain: (models) => set({ modelsToTrain: models }),
      setModelResults: (results) => set({ modelResults: results }),
      setModelParams: (params) => set((state) => ({ modelParams: { ...state.modelParams, ...params } })),
      reset: () => set({
        currentStep: 'UPLOAD',
        datasetId: null,
        datasetPreview: null,
        targetColumn: null,
        preprocessing: {
          missingValueStrategy: 'ffill',
          outlierStrategy: 'none'
        },
        taskType: 'regression',
        modelsToTrain: [],
        modelResults: {},
        modelParams: {
          splitFraction: 0.2,
          rfEstimators: 100,
          rfMaxDepth: 0,
          rfMinSamplesSplit: 2,
        }
      }),
    }),
    {
      name: 'automl-studio-state',
    }
  )
);
