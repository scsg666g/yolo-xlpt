import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import '@xyflow/react/dist/style.css'
import './styles.css'
import './compare.css'
import './editor.css'

const savedTheme = localStorage.getItem('yolo-forge-theme')
document.documentElement.dataset.theme = savedTheme === 'dark' ? 'dark' : 'light'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
)
