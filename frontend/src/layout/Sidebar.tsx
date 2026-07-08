import { NavLink } from 'react-router-dom'

// Mirrors config.settings.NAVIGATION_OPTIONS (order + labels). Icons are
// hand-picked here rather than zipped from NAVIGATION_ICONS, since that list
// has 3 leftover entries from the removed What-If/History/Comparison pages.
const NAV_ITEMS = [
  { to: '/', label: 'Overview', icon: '📈' },
  { to: '/upload', label: 'Upload Data', icon: '📤' },
  { to: '/preprocess', label: 'Preprocessing', icon: '⚙️' },
  { to: '/feature-selection', label: 'Feature Selection', icon: '🔍' },
  { to: '/train', label: 'Train Model', icon: '🧠' },
  { to: '/predict', label: 'Predict', icon: '🔮' },
]

export function Sidebar() {
  return (
    <aside
      style={{
        width: 240,
        flexShrink: 0,
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        background: `linear-gradient(180deg, var(--sidebar-top) 0%, var(--sidebar-mid) 50%, var(--sidebar-top) 100%)`,
        borderRight: '1px solid rgba(77, 166, 255, 0.20)',
        padding: '1.5rem 1rem',
      }}
    >
      <div style={{ marginBottom: '1.75rem', padding: '0 0.5rem' }}>
        <div
          style={{
            fontSize: '0.68rem',
            fontWeight: 700,
            letterSpacing: '0.12em',
            textTransform: 'uppercase',
            color: '#7fa8d9',
            marginBottom: '0.3rem',
          }}
        >
          Soft Sensor Platform
        </div>
        <div
          style={{
            fontFamily: 'Outfit, sans-serif',
            fontWeight: 800,
            fontSize: '1.5rem',
            color: '#ffffff',
            lineHeight: 1.1,
          }}
        >
          Multi X-Y
        </div>
        <div style={{ fontSize: '0.75rem', color: '#9db8dc', marginTop: '0.2rem' }}>
          Industrial DAE · ML Dashboard
        </div>
      </div>
      <nav style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            style={({ isActive }) => ({
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              padding: '0.65rem 0.9rem',
              borderRadius: 10,
              textDecoration: 'none',
              color: isActive ? '#ffffff' : '#b7c9e3',
              background: isActive ? 'rgba(255, 255, 255, 0.12)' : 'transparent',
              borderLeft: isActive ? '3px solid #7dd3fc' : '3px solid transparent',
              fontWeight: isActive ? 600 : 500,
              fontSize: '0.92rem',
            })}
          >
            <span>{item.icon}</span>
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>
      <div style={{ flex: 1 }} />
      <div
        style={{
          borderTop: '1px solid rgba(255,255,255,0.12)',
          paddingTop: '0.9rem',
          fontSize: '0.72rem',
          color: '#7f9dc4',
        }}
      >
        v1.0 · FastAPI · React
      </div>
    </aside>
  )
}
