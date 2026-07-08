import { apiClient } from './client'

export interface TrainingRequest {
  project_id: string
  algorithm: string
  hyperparameters: Record<string, unknown>
}

export async function submitTraining(body: TrainingRequest): Promise<string> {
  const { data } = await apiClient.post<{ job_id: string }>('/training/jobs', body)
  return data.job_id
}
