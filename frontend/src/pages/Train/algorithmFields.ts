// Mirrors the hyperparameter widgets in src/ui/pages/train.py — same field
// names, same default values — so **hyperparameters binds correctly against
// train_model / train_lstm / train_sklearn_model / train_kalman_model.
export type FieldSpec =
  | { key: string; label: string; type: 'number'; min: number; max: number; step: number }
  | { key: string; label: string; type: 'select'; options: (string | number)[] }
  | { key: string; label: string; type: 'checkbox' }

export const ALGORITHMS = ['DAE', 'Random Forest', 'XGBoost', 'LightGBM', 'LSTM', 'Kalman Filter'] as const

export const ALGO_FIELDS: Record<string, FieldSpec[]> = {
  DAE: [
    { key: 'masking_ratio', label: 'Masking Ratio', type: 'number', min: 0, max: 0.5, step: 0.01 },
    { key: 'epochs', label: 'Epochs', type: 'number', min: 10, max: 1000, step: 10 },
    { key: 'lr', label: 'Learning Rate', type: 'number', min: 0.0001, max: 0.1, step: 0.0001 },
    { key: 'latent_dim', label: 'Latent Dimension', type: 'number', min: 2, max: 50, step: 1 },
    { key: 'dropout_rate', label: 'Dropout Rate', type: 'number', min: 0, max: 0.5, step: 0.01 },
    { key: 'weight_to_pred', label: 'Weight to Predictor Loss', type: 'number', min: 0.1, max: 10, step: 0.1 },
    { key: 'batch_size', label: 'Batch Size', type: 'select', options: [16, 32, 64, 128, 256] },
    { key: 'auto_train', label: 'Auto-Train (until R² > 0.85)', type: 'checkbox' },
    { key: 'patience', label: 'Early Stop Patience', type: 'number', min: 0, max: 100, step: 5 },
  ],
  'Random Forest': [
    { key: 'n_estimators', label: 'N Estimators', type: 'number', min: 50, max: 1000, step: 50 },
    { key: 'max_depth', label: 'Max Depth (0 = unlimited)', type: 'number', min: 0, max: 50, step: 1 },
    { key: 'min_samples_split', label: 'Min Samples Split', type: 'number', min: 2, max: 20, step: 1 },
    { key: 'max_features', label: 'Max Features per Split', type: 'select', options: ['sqrt', 'log2', 'None'] },
  ],
  XGBoost: [
    { key: 'n_estimators', label: 'N Estimators', type: 'number', min: 50, max: 500, step: 50 },
    { key: 'max_depth', label: 'Max Depth', type: 'number', min: 3, max: 15, step: 1 },
    { key: 'learning_rate', label: 'Learning Rate', type: 'number', min: 0.01, max: 0.3, step: 0.01 },
    { key: 'subsample', label: 'Subsample', type: 'number', min: 0.5, max: 1.0, step: 0.05 },
    { key: 'colsample_bytree', label: 'ColSample by Tree', type: 'number', min: 0.5, max: 1.0, step: 0.05 },
  ],
  LightGBM: [
    { key: 'n_estimators', label: 'N Estimators', type: 'number', min: 50, max: 500, step: 50 },
    { key: 'max_depth', label: 'Max Depth', type: 'number', min: 3, max: 15, step: 1 },
    { key: 'learning_rate', label: 'Learning Rate', type: 'number', min: 0.01, max: 0.3, step: 0.01 },
    { key: 'num_leaves', label: 'Num Leaves', type: 'number', min: 15, max: 127, step: 4 },
    { key: 'subsample', label: 'Subsample', type: 'number', min: 0.5, max: 1.0, step: 0.05 },
  ],
  LSTM: [
    { key: 'hidden_size', label: 'Hidden Size', type: 'select', options: [32, 64, 128, 256] },
    { key: 'n_layers', label: 'LSTM Layers', type: 'number', min: 1, max: 4, step: 1 },
    { key: 'window_size', label: 'Window Size', type: 'number', min: 1, max: 30, step: 1 },
    { key: 'dropout_rate', label: 'Dropout Rate', type: 'number', min: 0, max: 0.5, step: 0.01 },
    { key: 'epochs', label: 'Epochs', type: 'number', min: 10, max: 500, step: 10 },
    { key: 'lr', label: 'Learning Rate', type: 'number', min: 0.0001, max: 0.01, step: 0.0001 },
    { key: 'batch_size', label: 'Batch Size', type: 'select', options: [16, 32, 64, 128] },
    { key: 'patience', label: 'Early Stop Patience', type: 'number', min: 0, max: 100, step: 5 },
  ],
  'Kalman Filter': [
    { key: 'process_noise', label: 'Process Noise (Q)', type: 'number', min: 0.000001, max: 1, step: 0.0001 },
    { key: 'measurement_noise', label: 'Measurement Noise (R)', type: 'number', min: 0.0001, max: 10, step: 0.001 },
    { key: 'initial_covariance', label: 'Initial Covariance (P0)', type: 'number', min: 0.1, max: 10, step: 0.1 },
    { key: 'n_epochs', label: 'Training Passes (Epochs)', type: 'number', min: 1, max: 50, step: 1 },
  ],
}

export const ALGO_DEFAULTS: Record<string, Record<string, unknown>> = {
  DAE: {
    masking_ratio: 0.1,
    epochs: 150,
    lr: 0.001,
    latent_dim: 5,
    dropout_rate: 0.2,
    weight_to_pred: 5.0,
    batch_size: 128,
    auto_train: false,
    patience: 20,
  },
  'Random Forest': {
    n_estimators: 300,
    max_depth: 0,
    min_samples_split: 2,
    max_features: 'sqrt',
    n_jobs: -1,
    random_state: 42,
  },
  XGBoost: {
    n_estimators: 300,
    max_depth: 6,
    learning_rate: 0.1,
    subsample: 0.8,
    colsample_bytree: 0.8,
    n_jobs: -1,
    random_state: 42,
    verbosity: 0,
  },
  LightGBM: {
    n_estimators: 300,
    max_depth: 6,
    learning_rate: 0.05,
    num_leaves: 31,
    subsample: 0.8,
    n_jobs: -1,
    random_state: 42,
    verbose: -1,
  },
  LSTM: {
    hidden_size: 64,
    n_layers: 2,
    window_size: 1,
    dropout_rate: 0.2,
    epochs: 100,
    lr: 0.001,
    batch_size: 64,
    patience: 20,
  },
  'Kalman Filter': {
    process_noise: 0.0001,
    measurement_noise: 0.01,
    initial_covariance: 1.0,
    n_epochs: 10,
    random_state: 42,
  },
}

// Random Forest's max_depth=0 means "unlimited" (None) and max_features="None"
// means the literal Python None — both handled the same way train.py's UI
// dispatch code converts them before calling train_sklearn_model.
export function toApiHyperparameters(algorithm: string, values: Record<string, unknown>): Record<string, unknown> {
  if (algorithm === 'Random Forest') {
    return {
      ...values,
      max_depth: values.max_depth === 0 ? null : values.max_depth,
      max_features: values.max_features === 'None' ? null : values.max_features,
    }
  }
  return values
}
