import type { ReactNode } from 'react'

const ICONS: Record<string, string> = {
  info: 'ℹ️',
  success: '✅',
  warning: '⚠️',
  error: '⚠️',
}

interface CalloutProps {
  variant: 'info' | 'success' | 'warning' | 'error'
  children: ReactNode
}

export function Callout({ variant, children }: CalloutProps) {
  return (
    <div className={`callout ${variant}`}>
      {ICONS[variant]} {children}
    </div>
  )
}
