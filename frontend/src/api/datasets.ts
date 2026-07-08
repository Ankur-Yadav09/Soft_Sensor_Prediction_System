import { apiClient } from './client'
import type { DatasetPreview, DatasetSummary } from './types'

export async function listDatasets(): Promise<DatasetSummary[]> {
  const { data } = await apiClient.get<{ datasets: DatasetSummary[] }>('/datasets')
  return data.datasets
}

export async function uploadDataset(file: File): Promise<DatasetSummary> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await apiClient.post<DatasetSummary>('/datasets/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function getDatasetPreview(name: string): Promise<DatasetPreview> {
  const { data } = await apiClient.get<DatasetPreview>(
    `/datasets/${encodeURIComponent(name)}/preview`,
  )
  return data
}

export async function deleteDataset(name: string): Promise<void> {
  await apiClient.delete(`/datasets/${encodeURIComponent(name)}`)
}
