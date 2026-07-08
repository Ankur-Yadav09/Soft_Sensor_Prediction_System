import { apiClient } from './client'
import type { DatasetPreview, DatasetSummary } from './types'

export async function listDatasets(): Promise<DatasetSummary[]> {
  const { data } = await apiClient.get<{ datasets: DatasetSummary[] }>('/datasets')
  return data.datasets
}

export interface UploadDatasetOptions {
  datasetName?: string
  plant?: string
  unit?: string
  onProgress?: (percent: number) => void
}

export async function uploadDataset(file: File, options: UploadDatasetOptions = {}): Promise<DatasetSummary> {
  const form = new FormData()
  form.append('file', file)
  if (options.datasetName) form.append('dataset_name', options.datasetName)
  if (options.plant) form.append('plant', options.plant)
  if (options.unit) form.append('unit', options.unit)
  const { data } = await apiClient.post<DatasetSummary>('/datasets/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (options.onProgress && e.total) {
        options.onProgress(Math.round((e.loaded / e.total) * 100))
      }
    },
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
