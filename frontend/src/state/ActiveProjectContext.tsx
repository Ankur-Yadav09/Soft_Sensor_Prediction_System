import { createContext, useContext, useState } from 'react'
import type { ReactNode } from 'react'

const STORAGE_KEY = 'softsense.activeProject'

interface ActiveProjectContextValue {
  activeProject: string
  setActiveProject: (id: string) => void
}

const ActiveProjectContext = createContext<ActiveProjectContextValue | null>(null)

export function ActiveProjectProvider({ children }: { children: ReactNode }) {
  const [activeProject, setActiveProjectState] = useState(() => localStorage.getItem(STORAGE_KEY) ?? '')

  function setActiveProject(id: string) {
    setActiveProjectState(id)
    if (id) localStorage.setItem(STORAGE_KEY, id)
    else localStorage.removeItem(STORAGE_KEY)
  }

  return (
    <ActiveProjectContext.Provider value={{ activeProject, setActiveProject }}>
      {children}
    </ActiveProjectContext.Provider>
  )
}

// Carries the project created by "Apply Preprocessing & Split Dataset" on
// Feature Selection into Train Model so its "Choose a Project" dropdown
// doesn't ask the user to pick what they just created.
export function useActiveProject() {
  const ctx = useContext(ActiveProjectContext)
  if (!ctx) throw new Error('useActiveProject must be used within ActiveProjectProvider')
  return ctx
}
