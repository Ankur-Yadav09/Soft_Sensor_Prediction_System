import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { listProjects } from '../../api/preprocess'
import { submitTraining } from '../../api/training'
import { Callout } from '../../components/Callout'
import { LineChart } from '../../components/LineChart'
import { StepHeading } from '../../components/StepHeading'
import { WorkflowStepper } from '../../components/WorkflowStepper'
import { useJobPolling } from '../../hooks/useJobPolling'
import { useActiveProject } from '../../state/ActiveProjectContext'
import { ALGO_DEFAULTS, ALGO_FIELDS, ALGORITHMS, toApiHyperparameters } from './algorithmFields'
import type { TrainingResult } from '../../api/types'

export function TrainPage() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { activeProject: projectId, setActiveProject: setProjectId } = useActiveProject()
  const [algorithm, setAlgorithm] = useState<string>('DAE')
  const [values, setValues] = useState<Record<string, unknown>>(ALGO_DEFAULTS.DAE)
  const [jobId, setJobId] = useState<string | null>(null)

  const projectsQuery = useQuery({ queryKey: ['projects'], queryFn: listProjects })

  // Set algorithm and its defaults together so no render ever sees `algorithm`
  // pointing at a new set of fields while `values` still holds the old
  // algorithm's keys (which caused inputs to briefly read undefined).
  function selectAlgorithm(a: string) {
    setAlgorithm(a)
    setValues(ALGO_DEFAULTS[a])
  }

  const submitMutation = useMutation({
    mutationFn: submitTraining,
    onSuccess: (id) => setJobId(id),
  })

  const jobQuery = useJobPolling(jobId)
  const result = jobQuery.data?.status === 'done' ? (jobQuery.data.result as TrainingResult) : null

  useEffect(() => {
    if (result) {
      queryClient.invalidateQueries({ queryKey: ['overview'] })
    }
  }, [result, queryClient])

  const selectedProject = (projectsQuery.data ?? []).find((p) => p.project_id === projectId)
  const canSubmit = !!projectId && !submitMutation.isPending

  const progress = jobQuery.data?.progress
  const pct =
    progress?.current && progress?.total ? Math.round((progress.current / progress.total) * 100) : null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      <h1>Train Model</h1>

      <WorkflowStepper current="build" />

      <div className="card" style={{ padding: '1.5rem' }}>
        <StepHeading step={1} title="Choose a Project" />
        <select value={projectId} onChange={(e) => setProjectId(e.target.value)} style={{ minWidth: 320 }}>
          <option value="">Select a preprocessed project…</option>
          {(projectsQuery.data ?? []).map((p) => (
            <option key={p.project_id} value={p.project_id}>
              {p.project_id} — {p.dataset_name} ({p.y_cols.join(', ')})
            </option>
          ))}
        </select>
        {selectedProject && (
          <p className="caption" style={{ marginTop: '0.5rem' }}>
            Carried over from Feature Selection. X: {selectedProject.x_cols.join(', ')} · Y:{' '}
            {selectedProject.y_cols.join(', ')} · Train/Test: {selectedProject.n_train}/{selectedProject.n_test}
          </p>
        )}
        {projectsQuery.data?.length === 0 && (
          <div style={{ marginTop: '0.75rem' }}>
            <Callout variant="warning">No projects yet — apply preprocessing on the Preprocessing page first.</Callout>
          </div>
        )}
      </div>

      <div className="card" style={{ padding: '1.5rem' }}>
        <StepHeading step={2} title="Select Model Architecture" />
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '1.25rem' }}>
          {ALGORITHMS.map((a) => (
            <button key={a} className={`chip${a === algorithm ? ' active' : ''}`} onClick={() => selectAlgorithm(a)}>
              {a}
            </button>
          ))}
        </div>

        <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
          {ALGO_FIELDS[algorithm].map((field) => (
            <label key={field.key}>
              <div className="caption">{field.label}</div>
              {field.type === 'number' && (
                <input
                  type="number"
                  min={field.min}
                  max={field.max}
                  step={field.step}
                  value={values[field.key] as number}
                  onChange={(e) => setValues({ ...values, [field.key]: Number(e.target.value) })}
                  style={{ width: 110 }}
                />
              )}
              {field.type === 'select' && (
                <select
                  value={values[field.key] as string | number}
                  onChange={(e) =>
                    setValues({
                      ...values,
                      [field.key]: field.options.every((o) => typeof o === 'number')
                        ? Number(e.target.value)
                        : e.target.value,
                    })
                  }
                >
                  {field.options.map((o) => (
                    <option key={o} value={o}>
                      {o}
                    </option>
                  ))}
                </select>
              )}
              {field.type === 'checkbox' && (
                <input
                  type="checkbox"
                  checked={values[field.key] as boolean}
                  onChange={(e) => setValues({ ...values, [field.key]: e.target.checked })}
                />
              )}
            </label>
          ))}
        </div>

        <button
          style={{ marginTop: '1.5rem' }}
          disabled={!canSubmit}
          onClick={() =>
            submitMutation.mutate({
              project_id: projectId,
              algorithm,
              hyperparameters: toApiHyperparameters(algorithm, values),
            })
          }
        >
          {submitMutation.isPending ? 'Submitting…' : '🚀 Train'}
        </button>

        {submitMutation.isError && (
          <div style={{ marginTop: '0.75rem' }}>
            <Callout variant="error">
              {(submitMutation.error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
                'Failed to submit training job.'}
            </Callout>
          </div>
        )}
      </div>

      {jobId && jobQuery.data && !jobQuery.data.done && (
        <div className="card" style={{ padding: '1.5rem' }}>
          <p className="caption">Training… {progress?.message ?? ''}</p>
          {pct !== null ? (
            <div style={{ background: 'var(--control-bg)', borderRadius: 8, height: 10, overflow: 'hidden' }}>
              <div
                style={{
                  width: `${pct}%`,
                  height: '100%',
                  background: 'linear-gradient(90deg,#3b82f6,#2563eb)',
                  transition: 'width 0.3s',
                }}
              />
            </div>
          ) : (
            <p className="caption">Working — this algorithm doesn't report incremental progress.</p>
          )}
        </div>
      )}

      {jobQuery.data?.status === 'error' && <Callout variant="error">{jobQuery.data.error}</Callout>}

      {result && (
        <div className="card" style={{ padding: '1.5rem' }}>
          <Callout variant="success">
            <strong>{result.algorithm}</strong> trained and saved as <code>{result.model_name}</code>
          </Callout>
          <p className="caption" style={{ marginTop: '0.75rem' }}>
            Avg R² = <code>{result.avg_r2.toFixed(4)}</code> · Avg RMSE ={' '}
            <code>{result.avg_rmse.toFixed(4)}</code> · Avg MAE = <code>{result.avg_mae.toFixed(4)}</code>
            {result.actual_epochs != null && <> · Epochs: {result.actual_epochs}</>}
            {result.early_stopped && <> · Early stopped</>}
          </p>

          {result.epoch_pred_losses.length > 0 && (
            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', marginTop: '1rem' }}>
              {result.epoch_recon_losses.length > 0 && (
                <LineChart
                  title="Reconstruction Loss"
                  xLabel="Epoch"
                  yLabel="Loss"
                  series={[
                    { label: 'Train', color: '#2563eb', data: result.epoch_recon_losses },
                    { label: 'Validation', color: '#10b981', data: result.val_recon_losses },
                  ]}
                />
              )}
              <LineChart
                title="Prediction Loss"
                xLabel="Epoch"
                yLabel="Loss"
                series={[
                  { label: 'Train', color: '#2563eb', data: result.epoch_pred_losses },
                  { label: 'Validation', color: '#10b981', data: result.val_pred_losses },
                ]}
              />
            </div>
          )}

          <button style={{ marginTop: '1.25rem' }} onClick={() => navigate('/predict')}>
            Continue to Predict →
          </button>
        </div>
      )}
    </div>
  )
}
