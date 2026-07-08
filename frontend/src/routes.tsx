import { createBrowserRouter } from 'react-router-dom'
import { Layout } from './layout/Layout'
import { FeatureSelectionPage } from './pages/FeatureSelection/FeatureSelectionPage'
import { OverviewPage } from './pages/Overview/OverviewPage'
import { PredictPage } from './pages/Predict/PredictPage'
import { PreprocessPage } from './pages/Preprocess/PreprocessPage'
import { TrainPage } from './pages/Train/TrainPage'
import { UploadPage } from './pages/Upload/UploadPage'

export const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      { path: '/', element: <OverviewPage /> },
      { path: '/upload', element: <UploadPage /> },
      { path: '/preprocess', element: <PreprocessPage /> },
      { path: '/feature-selection', element: <FeatureSelectionPage /> },
      { path: '/train', element: <TrainPage /> },
      { path: '/predict', element: <PredictPage /> },
    ],
  },
])
