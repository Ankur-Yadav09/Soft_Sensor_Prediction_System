import { apiClient } from './client'
import type { JobStatus } from './types'

export async function getJob(jobId: string): Promise<JobStatus> {
  const { data } = await apiClient.get<JobStatus>(`/jobs/${encodeURIComponent(jobId)}`)
  return data
}
