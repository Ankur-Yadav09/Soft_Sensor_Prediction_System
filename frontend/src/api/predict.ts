import { apiClient } from './client'
import type { PredictResult } from './types'

export interface PredictRequest {
  model_name: string
  source: 'project_test' | 'dataset'
  project_id?: string
  dataset_name?: string
  row_start?: number
  row_end?: number
}

export async function runPredict(body: PredictRequest): Promise<PredictResult> {
  const { data } = await apiClient.post<PredictResult>('/predict', body)
  return data
}
