import { NavLink } from 'react-router-dom'

// Mirrors config.settings.NAVIGATION_OPTIONS (order + labels). Icons are
// hand-picked here rather than zipped from NAVIGATION_ICONS, since that list
// has 3 leftover entries from the removed What-If/History/Comparison pages.
const NAV_ITEMS = [
  { to: '/', label: 'Overview', icon: '📈' },
  { to: '/upload', label: 'Connect Process Data', icon: '📤' },
  { to: '/preprocess', label: 'Preprocessing', icon: '⚙️' },
  { to: '/feature-selection', label: 'Feature Selection', icon: '🔍' },
  { to: '/train', label: 'Train Model', icon: '🧠' },
  { to: '/predict', label: 'Predict', icon: '🔮' },
]

export function Sidebar() {
  return (
    <aside
      style={{
        width: 248,
        flexShrink: 0,
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        background: `linear-gradient(180deg, var(--sidebar-top) 0%, var(--sidebar-mid) 50%, var(--sidebar-top) 100%)`,
        borderRight: '1px solid rgba(77, 166, 255, 0.20)',
        padding: '1.5rem 1rem',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '2rem', padding: '0 0.25rem' }}>
        <div
          style={{
            width: 42,
            height: 42,
            flexShrink: 0,
            borderRadius: 12,
            background: 'linear-gradient(135deg, #4da6ff 0%, #2563eb 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 10px rgba(37, 99, 235, 0.35)',
          }}
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
            <path
              d="M3 12h4l2-7 4 14 2-7h6"
              stroke="white"
              strokeWidth="2.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
        <div>
          <div
            style={{
              fontFamily: 'Outfit, sans-serif',
              fontWeight: 800,
              fontSize: '1.35rem',
              color: '#ffffff',
              lineHeight: 1.15,
            }}
          >
            SoftSense AI
          </div>
          <div style={{ fontSize: '0.72rem', color: '#9db8dc', marginTop: '0.1rem' }}>
            Industrial Intelligence
          </div>
        </div>
      </div>
      <nav style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            style={({ isActive }) => ({
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              padding: '0.55rem 0.6rem',
              borderRadius: 12,
              textDecoration: 'none',
              background: isActive ? 'rgba(77, 166, 255, 0.16)' : 'transparent',
              transition: 'background 0.15s ease',
            })}
          >
            {({ isActive }) => (
              <>
                <span
                  style={{
                    width: 32,
                    height: 32,
                    flexShrink: 0,
                    borderRadius: 9,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '1.05rem',
                    background: isActive ? 'linear-gradient(135deg, #4da6ff 0%, #2563eb 100%)' : 'rgba(255,255,255,0.06)',
                    boxShadow: isActive ? '0 2px 6px rgba(37, 99, 235, 0.4)' : 'none',
                  }}
                >
                  {item.icon}
                </span>
                <span
                  style={{
                    color: isActive ? '#ffffff' : '#b7c9e3',
                    fontWeight: isActive ? 700 : 500,
                    fontSize: '0.94rem',
                  }}
                >
                  {item.label}
                </span>
              </>
            )}
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
