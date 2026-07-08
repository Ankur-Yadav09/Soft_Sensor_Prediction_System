import { createContext, useContext, useState } from 'react'
import type { ReactNode } from 'react'

const STORAGE_KEY = 'softsense.activeDataset'

interface ActiveDatasetContextValue {
  activeDataset: string
  setActiveDataset: (name: string) => void
}

const ActiveDatasetContext = createContext<ActiveDatasetContextValue | null>(null)

export function ActiveDatasetProvider({ children }: { children: ReactNode }) {
  const [activeDataset, setActiveDatasetState] = useState(() => localStorage.getItem(STORAGE_KEY) ?? '')

  function setActiveDataset(name: string) {
    setActiveDatasetState(name)
    if (name) localStorage.setItem(STORAGE_KEY, name)
    else localStorage.removeItem(STORAGE_KEY)
  }

  return (
    <ActiveDatasetContext.Provider value={{ activeDataset, setActiveDataset }}>
      {children}
    </ActiveDatasetContext.Provider>
  )
}

// Carries the dataset chosen on Connect Process Data across Preprocessing and
// Feature Selection so those pages don't ask the user to pick it again.
export function useActiveDataset() {
  const ctx = useContext(ActiveDatasetContext)
  if (!ctx) throw new Error('useActiveDataset must be used within ActiveDatasetProvider')
  return ctx
}
