interface StepHeadingProps {
  step: number
  title: string
}

// Mirrors the "①  Section Title" numbered-step headers used throughout the
// real app's wizard-like pages (Preprocess, Feature Selection).
export function StepHeading({ step, title }: StepHeadingProps) {
  return (
    <div className="step-heading">
      <span className="step-badge">{step}</span>
      <h2>{title}</h2>
    </div>
  )
}
