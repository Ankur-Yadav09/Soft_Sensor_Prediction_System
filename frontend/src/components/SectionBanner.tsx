interface SectionBannerProps {
  icon: string
  title: string
  subtitle: string
}

// The dark "hero card" the real app uses to introduce a major step, e.g.
// "🎯 Target Variable Selection" on Feature Selection, "🔍 Data
// Understanding" on Preprocess — visually distinct from the plain white
// page background.
export function SectionBanner({ icon, title, subtitle }: SectionBannerProps) {
  return (
    <div className="section-banner">
      <span className="icon">{icon}</span>
      <div>
        <h2>{title}</h2>
        <p>{subtitle}</p>
      </div>
    </div>
  )
}
