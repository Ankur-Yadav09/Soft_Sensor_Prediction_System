import { apiClient } from './client'
import type { ModelPerformance, OverviewResponse } from './types'

export async function getOverview(): Promise<OverviewResponse> {
  const { data } = await apiClient.get<OverviewResponse>('/overview')
  return data
}

export async function getModelPerformance(modelName: string): Promise<ModelPerformance> {
  const { data } = await apiClient.get<ModelPerformance>(
    `/models/${encodeURIComponent(modelName)}/performance`,
  )
  return data
}
