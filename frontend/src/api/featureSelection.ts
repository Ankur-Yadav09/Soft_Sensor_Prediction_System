import { apiClient } from './client'

export interface FeatureSelectionRequest {
  dataset_name: string
  y_cols: string[]
  x_cols?: string[]
  top_k: number
  corr_threshold: number
  vif_threshold: number
  per_target: boolean
}

export async function submitFeatureSelection(body: FeatureSelectionRequest): Promise<string> {
  const { data } = await apiClient.post<{ job_id: string }>('/feature-selection/jobs', body)
  return data.job_id
}
