import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type WizardStep = 'UPLOAD' | 'EDA' | 'MODEL' | 'TRAIN' | 'RESULTS';

interface AppState {
  currentStep: WizardStep;
  datasetId: string | null;
  datasetPreview: any | null; // Preview from pandas
  targetColumn: string | null;
  preprocessing: {
    missingValueStrategy: string;
    outlierStrategy: string;
  };
  modelsToTrain: string[];
  modelResults: Record<string, { metrics: {rmse: number, mae: number}, forecast: number[] }>;
  
  setStep: (step: WizardStep) => void;
  setDataset: (id: string, preview: any) => void;
  setEDAOptions: (target: string, missing: string, outliers: string) => void;
  setModelsToTrain: (models: string[]) => void;
  setModelResults: (results: Record<string, { metrics: {rmse: number, mae: number}, forecast: number[] }>) => void;
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
      modelsToTrain: [],
      modelResults: {},
      
      setStep: (step) => set({ currentStep: step }),
      setDataset: (id, preview) => set({ datasetId: id, datasetPreview: preview }),
      setEDAOptions: (target, missing, outliers) => set({
        targetColumn: target,
        preprocessing: { missingValueStrategy: missing, outlierStrategy: outliers }
      }),
      setModelsToTrain: (models) => set({ modelsToTrain: models }),
      setModelResults: (results) => set({ modelResults: results }),
      reset: () => set({
        currentStep: 'UPLOAD',
        datasetId: null,
        datasetPreview: null,
        targetColumn: null,
        preprocessing: {
          missingValueStrategy: 'ffill',
          outlierStrategy: 'none'
        },
        modelsToTrain: [],
        modelResults: {},
      }),
    }),
    {
      name: 'automl-studio-state',
    }
  )
);
