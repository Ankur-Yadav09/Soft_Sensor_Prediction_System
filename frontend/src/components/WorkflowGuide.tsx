const STEPS = [
  ['1', 'Connect Process Data', 'Upload an Excel/CSV file or load a previously stored dataset.'],
  ['2', 'Preprocessing', 'Clean, impute, and treat outliers in the raw data.'],
  ['3', 'Feature Selection', 'Pick input (X) and target (Y) columns.'],
  ['4', 'Train Model', 'Train a DAE, tree ensemble, LSTM, or Kalman Filter model.'],
  ['5', 'Predict', 'Run inference and export predictions.'],
]

export function WorkflowGuide() {
  return (
    <div>
      <h3 style={{ marginBottom: '1rem' }}>Workflow Guide</h3>
      <table>
        <thead>
          <tr>
            <th>Step</th>
            <th>Page</th>
            <th>What it does</th>
          </tr>
        </thead>
        <tbody>
          {STEPS.map(([step, page, desc]) => (
            <tr key={step}>
              <td>{step}</td>
              <td>{page}</td>
              <td className="caption">{desc}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
