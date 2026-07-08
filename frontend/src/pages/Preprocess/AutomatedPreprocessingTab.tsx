import { useMutation, useQueryClient } from '@tanstack/react-query'
import { applyAutomatedCleaning } from '../../api/preprocess'
import { Callout } from '../../components/Callout'
import { SectionBanner } from '../../components/SectionBanner'
import type { CleaningResponse } from '../../api/preprocess'

interface AutomatedPreprocessingTabProps {
  datasetName: string
  onCleaned: (result: CleaningResponse) => void
}

export function AutomatedPreprocessingTab({ datasetName, onCleaned }: AutomatedPreprocessingTabProps) {
  const queryClient = useQueryClient()

  const mutation = useMutation({
    mutationFn: () => applyAutomatedCleaning(datasetName),
    onSuccess: (res) => {
      onCleaned(res)
      queryClient.invalidateQueries({ queryKey: ['datasets'] })
      queryClient.invalidateQueries({ queryKey: ['overview'] })
    },
  })

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <SectionBanner
        icon="🤖"
        title="Automated Preprocessing"
        subtitle="One click — runs the optimal cleaning pipeline with best-default settings and shows each step applied."
      />
      <p className="caption">
        Best-default pipeline: <strong>Cast to numeric</strong> → <strong>Remove duplicates</strong> →{' '}
        <strong>Drop high-missing columns (≥50%)</strong> → <strong>Remove constant columns</strong> →{' '}
        <strong>Remove near-zero variance columns (std&lt;0.01)</strong> → <strong>Median imputation</strong> →{' '}
        <strong>IQR capping (1.5×)</strong>
      </p>

      <button onClick={() => mutation.mutate()} disabled={mutation.isPending} style={{ alignSelf: 'flex-start' }}>
        {mutation.isPending ? 'Running…' : '▶ Run Preprocessing'}
      </button>

      {mutation.isError && <Callout variant="error">Automated preprocessing failed.</Callout>}

      {mutation.data && (
        <div>
          <Callout variant="success">
            Automated preprocessing complete. Rows: {mutation.data.before_rows} → {mutation.data.after_rows} · Columns:{' '}
            {mutation.data.after_cols}. Saved as <code>{mutation.data.new_dataset_name}</code>.
          </Callout>
          <ul className="caption">
            {(mutation.data.step_log ?? []).map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
