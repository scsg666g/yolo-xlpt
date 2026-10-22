import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'

const Dashboard = lazy(() => import('./components/Dashboard').then((module) => ({ default: module.Dashboard })))
const TaskCanvas = lazy(() => import('./components/TaskCanvas').then((module) => ({ default: module.TaskCanvas })))
const ExperimentCompare = lazy(() => import('./components/ExperimentCompare').then((module) => ({ default: module.ExperimentCompare })))
const ModelTests = lazy(() => import('./components/ModelTests').then((module) => ({ default: module.ModelTests })))
const ComputeResources = lazy(() => import('./components/ComputeResources').then((module) => ({ default: module.ComputeResources })))
const Datasets = lazy(() => import('./components/Datasets').then((module) => ({ default: module.Datasets })))

function PageLoading() {
  return <div className="empty">页面资源加载中…</div>
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Suspense fallback={<PageLoading />}><Dashboard /></Suspense>} />
        <Route path="compare" element={<Suspense fallback={<PageLoading />}><ExperimentCompare /></Suspense>} />
        <Route path="model-tests" element={<Suspense fallback={<PageLoading />}><ModelTests /></Suspense>} />
        <Route path="compute" element={<Suspense fallback={<PageLoading />}><ComputeResources /></Suspense>} />
        <Route path="datasets" element={<Suspense fallback={<PageLoading />}><Datasets /></Suspense>} />
        <Route path="tasks/:taskId/canvas" element={<Suspense fallback={<PageLoading />}><TaskCanvas /></Suspense>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
