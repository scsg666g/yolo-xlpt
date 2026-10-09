import { useState } from 'react'
import { Activity, Boxes, Cpu, LayoutDashboard, Moon, Settings2, Sun } from 'lucide-react'
import { NavLink, Outlet } from 'react-router-dom'

export function Layout() {
  const [theme, setTheme] = useState<'light' | 'dark'>(() =>
    document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light',
  )

  function changeTheme(nextTheme: 'light' | 'dark') {
    setTheme(nextTheme)
    document.documentElement.dataset.theme = nextTheme
    localStorage.setItem('yolo-forge-theme', nextTheme)
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Boxes size={19} /></div>
          <div><strong>YOLO Forge</strong><span>训练控制台</span></div>
        </div>
        <div className="theme-picker" aria-label="界面主题">
          <span>界面主题</span>
          <div>
            <button className={theme === 'light' ? 'active' : ''} onClick={() => changeTheme('light')} title="明亮主题" aria-pressed={theme === 'light'}><Sun size={14} />明亮</button>
            <button className={theme === 'dark' ? 'active' : ''} onClick={() => changeTheme('dark')} title="深色主题" aria-pressed={theme === 'dark'}><Moon size={14} />深色</button>
          </div>
        </div>
        <nav>
          <NavLink to="/" end><LayoutDashboard size={18} />任务中心</NavLink>
          <NavLink to="/compare"><Activity size={18} />实验对比</NavLink>
          <span className="nav-muted"><Cpu size={18} />算力资源</span>
        </nav>
        <div className="sidebar-footer">
          <div className="service-state"><i />训练服务在线</div>
          <button className="icon-text"><Settings2 size={16} />平台设置</button>
        </div>
      </aside>
      <main className="main"><Outlet /></main>
    </div>
  )
}
