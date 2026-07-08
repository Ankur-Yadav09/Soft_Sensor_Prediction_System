import { useEffect, useRef, useState } from 'react'

interface MultiSelectDropdownProps {
  options: string[]
  selected: Set<string>
  onChange: (next: Set<string>) => void
  placeholder?: string
}

// A collapsed, click-to-expand multi-select — matches the interaction
// pattern of Streamlit's st.multiselect (closed box with a placeholder +
// chevron, opens a checkable list, selections shown as removable tags in
// the closed box) since there's no native HTML equivalent of that widget.
export function MultiSelectDropdown({ options, selected, onChange, placeholder = 'Choose options' }: MultiSelectDropdownProps) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [])

  function toggle(option: string) {
    const next = new Set(selected)
    if (next.has(option)) next.delete(option)
    else next.add(option)
    onChange(next)
  }

  function remove(option: string, e: React.MouseEvent) {
    e.stopPropagation()
    const next = new Set(selected)
    next.delete(option)
    onChange(next)
  }

  return (
    <div ref={rootRef} style={{ position: 'relative', minWidth: 320 }}>
      <div
        onClick={() => setOpen((o) => !o)}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '0.5rem',
          background: 'var(--control-bg)',
          border: '1px solid var(--border)',
          borderRadius: 8,
          padding: '0.4rem 0.7rem',
          minHeight: 38,
          cursor: 'pointer',
        }}
      >
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem', flex: 1 }}>
          {selected.size === 0 && <span style={{ color: 'var(--text-caption)' }}>{placeholder}</span>}
          {[...selected].map((s) => (
            <span
              key={s}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.3rem',
                background: 'var(--primary)',
                color: 'white',
                borderRadius: 6,
                padding: '0.15rem 0.4rem',
                fontSize: '0.8rem',
              }}
            >
              {s}
              <span onClick={(e) => remove(s, e)} style={{ cursor: 'pointer', fontWeight: 700 }}>
                ×
              </span>
            </span>
          ))}
        </div>
        <span style={{ color: 'var(--text-caption)' }}>{open ? '▲' : '▼'}</span>
      </div>

      {open && (
        <div
          className="card"
          style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            right: 0,
            marginTop: 4,
            maxHeight: 260,
            overflowY: 'auto',
            zIndex: 20,
            padding: '0.4rem 0',
          }}
        >
          {options.map((o) => (
            <label
              key={o}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.4rem 0.8rem',
                cursor: 'pointer',
              }}
              onMouseDown={(e) => e.preventDefault()}
            >
              <input type="checkbox" checked={selected.has(o)} onChange={() => toggle(o)} />
              {o}
            </label>
          ))}
        </div>
      )}
    </div>
  )
}
