import { Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { Dashboard } from './components/Dashboard'
import { TaskCanvas } from './components/TaskCanvas'
import { ExperimentCompare } from './components/ExperimentCompare'
import { ComputeResources } from './components/ComputeResources'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="compare" element={<ExperimentCompare />} />
        <Route path="compute" element={<ComputeResources />} />
        <Route path="tasks/:taskId/canvas" element={<TaskCanvas />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
