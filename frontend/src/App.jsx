import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import Home from './pages/Home'
import Demo from './pages/Demo'

function Navbar() {
  const linkClass = ({ isActive }) =>
    `font-mono text-sm tracking-wider transition-colors duration-200 px-3 py-1.5 rounded-md ${
      isActive
        ? 'text-accent-blue bg-blue-500/10'
        : 'text-neutral-500 hover:text-neutral-300'
    }`

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-accent-blue/10 bg-neutral-900/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
        <div className="flex items-center gap-3">
          {/* Logo signal icon */}
          <svg className="w-5 h-5 text-accent-blue" viewBox="0 0 20 20" fill="none">
            <path d="M10 2 L10 18 M6 6 L6 14 M14 6 L14 14 M2 10 L18 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
          </svg>
          <span className="font-display font-semibold text-neutral-100 tracking-tight">SemComm</span>
          <span className="label-tag bg-blue-500/10 text-accent-blue border border-accent-blue/20">v0.1</span>
        </div>
        <div className="flex items-center gap-1">
          <NavLink to="/" end className={linkClass}>Home</NavLink>
          <NavLink to="/demo" className={linkClass}>Demo</NavLink>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="ml-2 font-mono text-sm text-neutral-600 hover:text-neutral-400 transition-colors"
          >
            GitHub ↗
          </a>
        </div>
      </div>
    </nav>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <main className="pt-14">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/demo" element={<Demo />} />
        </Routes>
      </main>
    </BrowserRouter>
  )
}
