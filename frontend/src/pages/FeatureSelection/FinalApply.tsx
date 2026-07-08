import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { applyPreprocessing } from '../../api/preprocess'
import { Callout } from '../../components/Callout'
import { SectionBanner } from '../../components/SectionBanner'
import { useActiveProject } from '../../state/ActiveProjectContext'

interface FinalApplyProps {
  datasetName: string
  xCols: string[]
  yCols: string[]
}

const SPLIT_METHODS = ['Random Split', 'Stratified Split', 'Sequential Split']

export function FinalApply({ datasetName, xCols, yCols }: FinalApplyProps) {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const { setActiveProject } = useActiveProject()
  const [imputationMethod, setImputationMethod] = useState('Median')
  const [splitMethod, setSplitMethod] = useState('Random Split')
  const [trainRatio, setTrainRatio] = useState(0.8)

  const applyMutation = useMutation({
    mutationFn: applyPreprocessing,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
      setActiveProject(data.project_id)
    },
  })

  const canApply = xCols.length > 0 && yCols.length > 0 && !applyMutation.isPending

  function apply() {
    applyMutation.mutate({
      dataset_name: datasetName,
      x_cols: xCols,
      y_cols: yCols,
      imputation_method: imputationMethod,
      outlier_method: 'None',
      split_method: splitMethod === 'Sequential Split' ? 'sequential' : 'random',
      test_size: Math.round((1 - trainRatio) * 100) / 100,
      stratify_bins: splitMethod === 'Stratified Split' ? 5 : 0,
    })
  }

  return (
    <div className="card" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <SectionBanner
        icon="🚀"
        title="Apply Preprocessing & Split Dataset"
        subtitle="Finalise feature selection, handle remaining missing values, configure the train/test split, and scale."
      />

      <div>
        <p style={{ fontWeight: 700, color: 'var(--primary)', marginBottom: '0.2rem' }}>Fallback Imputation</p>
        <p className="caption">Fills any NaN values that remain in the selected X and Y columns.</p>
        <select value={imputationMethod} onChange={(e) => setImputationMethod(e.target.value)}>
          <option>Mean</option>
          <option>Median</option>
          <option>Zero</option>
        </select>
      </div>

      <div>
        <p style={{ fontWeight: 700, color: 'var(--primary)', marginBottom: '0.2rem' }}>Dataset Split</p>
        <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-end' }}>
          <label>
            <div className="caption">Split Method</div>
            <select value={splitMethod} onChange={(e) => setSplitMethod(e.target.value)}>
              {SPLIT_METHODS.map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </label>
          <label>
            <div className="caption">Train Ratio: {(trainRatio * 100).toFixed(0)}%</div>
            <input
              type="range"
              min={0.5}
              max={0.95}
              step={0.05}
              value={trainRatio}
              onChange={(e) => setTrainRatio(Number(e.target.value))}
            />
          </label>
        </div>
        {splitMethod === 'Stratified Split' && (
          <p className="caption">Stratification: first Y column will be binned into 5 equal-frequency quantile groups.</p>
        )}
        {splitMethod === 'Sequential Split' && (
          <p className="caption">Sequential: row order is preserved; no shuffling.</p>
        )}
      </div>

      <button onClick={apply} disabled={!canApply} style={{ alignSelf: 'flex-start' }}>
        {applyMutation.isPending ? 'Applying…' : '🚀 Apply Preprocessing & Split Dataset'}
      </button>

      {applyMutation.isError && (
        <Callout variant="error">
          {(applyMutation.error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
            'Failed to apply preprocessing.'}
        </Callout>
      )}

      {applyMutation.data && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <Callout variant="success">
            Preprocessing complete — {applyMutation.data.x_cols.length} X features, {applyMutation.data.y_cols.length}{' '}
            Y target(s). Created project <code>{applyMutation.data.project_id}</code> — {applyMutation.data.n_train}{' '}
            train rows / {applyMutation.data.n_test} test rows.
          </Callout>
          <button style={{ alignSelf: 'flex-start' }} onClick={() => navigate('/train')}>
            Continue to Train Model →
          </button>
        </div>
      )}
    </div>
  )
}
