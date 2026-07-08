import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { WorkflowStepper } from '../../components/WorkflowStepper'
import { useActiveDataset } from '../../state/ActiveDatasetContext'
import { AIRecommendationCard } from './AIRecommendationCard'
import { DataPreviewSection } from './DataPreviewSection'
import { UploadNewDatasetWizard } from './UploadNewDatasetWizard'
import { UseExistingDatasetTab } from './UseExistingDatasetTab'

type SourceTab = 'existing' | 'new'

export function UploadPage() {
  const navigate = useNavigate()
  const [tab, setTab] = useState<SourceTab>('existing')
  const { activeDataset: selectedName, setActiveDataset: setSelectedName } = useActiveDataset()

  function handleUploadAnother() {
    setSelectedName('')
    setTab('new')
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      <div>
        <h1>Connect Process Data</h1>
        <p className="caption">Connect historical process data to begin building an AI-powered Virtual Sensor.</p>
      </div>

      <WorkflowStepper current="connect" />

      <div style={{ display: 'flex', gap: '0.5rem' }}>
        <button
          className={tab === 'existing' ? 'chip active' : 'chip'}
          onClick={() => setTab('existing')}
        >
          1. Use Existing Dataset
        </button>
        <button
          className={tab === 'new' ? 'chip active' : 'chip'}
          onClick={() => setTab('new')}
        >
          2. Upload New Dataset
        </button>
      </div>

      {tab === 'existing' ? (
        <UseExistingDatasetTab selectedName={selectedName} onSelect={setSelectedName} />
      ) : (
        <UploadNewDatasetWizard onUploaded={(summary) => setSelectedName(summary.name)} />
      )}

      {selectedName && (
        <>
          <DataPreviewSection datasetName={selectedName} />
          <AIRecommendationCard datasetName={selectedName} />

          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button onClick={() => navigate('/preprocess')}>Continue to Data Health Assessment →</button>
            <button className="chip" onClick={handleUploadAnother}>Upload Another Dataset</button>
            <button className="chip" onClick={() => navigate('/')}>← Back</button>
          </div>
        </>
      )}
    </div>
  )
}
