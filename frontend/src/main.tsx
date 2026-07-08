import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { RouterProvider } from 'react-router-dom'
import { router } from './routes'
import { ActiveDatasetProvider } from './state/ActiveDatasetContext'
import { ActiveProjectProvider } from './state/ActiveProjectContext'
import './theme/theme.css'

const queryClient = new QueryClient()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <ActiveDatasetProvider>
        <ActiveProjectProvider>
          <RouterProvider router={router} />
        </ActiveProjectProvider>
      </ActiveDatasetProvider>
    </QueryClientProvider>
  </StrictMode>,
)
