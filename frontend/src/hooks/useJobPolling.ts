import { useQuery } from '@tanstack/react-query'
import { getJob } from '../api/jobs'

// Polls GET /api/jobs/{id} every 1.5s until the job reaches a terminal state —
// the React-side half of the job manager's async pattern (see backend/app/jobs/manager.py).
export function useJobPolling(jobId: string | null) {
  return useQuery({
    queryKey: ['jobs', jobId],
    queryFn: () => getJob(jobId as string),
    enabled: !!jobId,
    refetchInterval: (query) => (query.state.data?.done ? false : 1500),
  })
}
