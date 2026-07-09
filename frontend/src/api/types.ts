export interface DatasetSummary {
  name: string
  uploaded_at: string
  rows: number
  cols: number
  plant?: string | null
  unit?: string | null
  status?: string
}

export interface DatasetPreview {
  name: string
  shape: [number, number]
  columns: string[]
  head: Record<string, unknown>[]
}

export interface SavedModelSummary {
  name: string
  saved_at: string
  input_dim: number
  output_dim: number
  algorithm: string | null
  avg_r2: number | null
  avg_rmse: number | null
  avg_mae: number | null
}

export interface OverviewResponse {
  datasets: DatasetSummary[]
  saved_models: SavedModelSummary[]
}

export interface TargetMetric {
  name: string
  r2: number
  mae: number
}

export interface ModelPerformance {
  model_name: string
  dataset_name: string
  per_target: TargetMetric[]
  avg_r2: number
  grade: string
  emoji: string
  x_cols: string[]
  y_cols: string[]
}

export interface FeatureStat {
  Feature: string
  'Missing Count': number
  'Missing %': number
  Min: number | null
  Mean: number | null
  Max: number | null
}

export interface ProjectSummary {
  project_id: string
  dataset_name: string
  x_cols: string[]
  y_cols: string[]
  created_at: string
  n_train: number
  n_test: number
  config: Record<string, unknown>
}

export interface ApplyPreprocessingResponse {
  project_id: string
  dataset_name: string
  x_cols: string[]
  y_cols: string[]
  n_train: number
  n_test: number
}

export interface JobStatus {
  id: string
  status: 'pending' | 'running' | 'done' | 'error'
  progress: { current?: number; total?: number; message?: string }
  error: string | null
  done: boolean
  result: unknown
}

export interface FeatureConsensusRow {
  Rank: number
  Feature: string
  FinalScore: number
  PredictiveStrength: number
  StabilityScore: number
  SelectionFreq: number
  VIF: number | null
  CorrWithTarget: number | null
  Recommendation: string
  ElasticNetSelected: boolean | null
  MulticollinearWith?: string | null
}

export interface MethodResultOut {
  name: string
  method_id: string
  category: string
  selected_features: string[]
  notes: string
  success: boolean
  raw_scores: Record<string, number>
  all_scores: Record<string, number>
  per_target_scores: Record<string, Record<string, number>>
}

export interface FeatureSelectionResult {
  mode: 'combined' | 'per_target'
  consensus: FeatureConsensusRow[]
  recommended_features: string[]
  optional_features: string[]
  features_to_remove: string[]
  per_feature_reasoning: Record<string, string>
  dataset_info: {
    n_rows: number
    n_raw_features: number
    n_clean_features: number
    n_targets: number
    target_names: string[]
    constant_features: string[]
    missing_pct_x: number
    missing_pct_y: number
    vif_skipped?: boolean
    permutation_skipped?: boolean
  }
  method_results: MethodResultOut[]
  vif: { Feature: string; VIF: number | null; VIF_Level: string }[]
  corr_with_target: Record<string, unknown>[]
  feature_target_map?: Record<string, string[]>
  per_target_summary?: Record<string, { recommended_features: string[]; optional_features: string[] }>
}

export interface FeatureDetail {
  column: string
  empty: boolean
  dtype?: string
  distribution_label?: string
  n_total: number
  n_missing: number
  n_unique?: number
  n_duplicate_rows?: number
  n_outliers_iqr?: number
  mean?: number
  median?: number
  std?: number
  min?: number
  max?: number
  skew?: number
  kurtosis?: number
  histogram?: { counts: number[]; bin_edges: number[] }
  boxplot?: { min: number; q1: number; median: number; q3: number; max: number }
}

export interface TrainingResult {
  model_name: string
  algorithm: string
  avg_r2: number
  avg_rmse: number
  avg_mae: number
  actual_epochs: number | null
  early_stopped: boolean
  epoch_recon_losses: number[]
  epoch_pred_losses: number[]
  val_recon_losses: number[]
  val_pred_losses: number[]
}

export interface PredictResult {
  x_cols: string[]
  y_cols: string[]
  rows: Record<string, unknown>[]
  has_actuals: boolean
  metrics: Record<string, unknown>[] | null
}
