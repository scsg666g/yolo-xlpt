import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import { initializeAppearance } from './appearance'
import '@xyflow/react/dist/style.css'
import './styles.css'
import './compare.css'
import './compute.css'
import './datasets.css'
import './editor.css'
import './appearance.css'

initializeAppearance()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
)
