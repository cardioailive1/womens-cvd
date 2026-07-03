import { NavLink, Navigate, Route, Routes } from 'react-router-dom';
import { useAuth } from './auth';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Patients from './pages/Patients';
import Assessment from './pages/Assessment';
import Interop from './pages/Interop';
import Diagnostics from './pages/Diagnostics';
import Alerts from './pages/Alerts';
import Reports from './pages/Reports';
import Users from './pages/Users';
import Account from './pages/Account';

function Logo() {
  return (
    <svg width="34" height="34" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
      <ellipse cx="100" cy="100" rx="70" ry="30" fill="none" stroke="#fff" strokeWidth="7" transform="rotate(0 100 100)" />
      <ellipse cx="100" cy="100" rx="70" ry="30" fill="none" stroke="#fff" strokeWidth="7" transform="rotate(60 100 100)" />
      <ellipse cx="100" cy="100" rx="70" ry="30" fill="none" stroke="#fff" strokeWidth="7" transform="rotate(120 100 100)" />
      <path d="M100 85 C100 75,90 70,85 75 C80 80,80 85,85 90 L100 105 L115 90 C120 85,120 80,115 75 C110 70,100 75,100 85Z" fill="#E74C3C" />
    </svg>
  );
}

function Shell() {
  const { user, logout } = useAuth();
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <Logo />
          <div><h1>CARDIO AI</h1><small>Women's Heart Health</small></div>
        </div>
        <nav className="nav">
          <NavLink to="/" end>📊 Dashboard</NavLink>
          <NavLink to="/patients">👥 Patients</NavLink>
          <NavLink to="/assessment">🔬 Risk Assessment</NavLink>
          <NavLink to="/diagnostics">🫀 Diagnostics Guide</NavLink>
          <NavLink to="/alerts">🚨 Alerts</NavLink>
          <NavLink to="/reports">📋 Reports</NavLink>
          <NavLink to="/interop">🔗 FHIR / HL7</NavLink>
          {user?.role === 'ADMIN' && <NavLink to="/users">🛡️ Users</NavLink>}
          <NavLink to="/account">⚙️ Account</NavLink>
        </nav>
        <div className="sidebar-foot">
          <div className="who">{user?.fullName}<br />{user?.role}</div>
          <button className="logout" onClick={logout}>Sign out</button>
        </div>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/patients" element={<Patients />} />
          <Route path="/assessment" element={<Assessment />} />
          <Route path="/diagnostics" element={<Diagnostics />} />
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/interop" element={<Interop />} />
          {user?.role === 'ADMIN' && <Route path="/users" element={<Users />} />}
          <Route path="/account" element={<Account />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  const { user, loading } = useAuth();
  if (loading) return <div className="spinner" />;
  if (!user) return <Login />;
  if (user.mustChangePassword) return <Account forced />;
  return <Shell />;
}
