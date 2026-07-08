import { useState } from 'react'
import type { ReactNode } from 'react'

interface Tab {
  label: string
  content: ReactNode
}

interface TabsProps {
  tabs: Tab[]
  defaultIndex?: number
}

export function Tabs({ tabs, defaultIndex = 0 }: TabsProps) {
  const [active, setActive] = useState(defaultIndex)
  return (
    <div>
      <div style={{ display: 'flex', gap: '0.25rem', borderBottom: '1px solid var(--border)', marginBottom: '1.25rem', flexWrap: 'wrap' }}>
        {tabs.map((tab, i) => (
          <button
            key={tab.label}
            onClick={() => setActive(i)}
            style={{
              background: 'transparent',
              boxShadow: 'none',
              color: active === i ? 'var(--primary)' : 'var(--text-caption)',
              fontWeight: active === i ? 700 : 500,
              borderRadius: 0,
              borderBottom: active === i ? '2px solid var(--primary)' : '2px solid transparent',
              padding: '0.6rem 0.9rem',
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div>{tabs[active].content}</div>
    </div>
  )
}
