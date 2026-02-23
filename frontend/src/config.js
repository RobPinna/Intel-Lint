export const API_BASE = String(import.meta.env.VITE_API_BASE || '').trim().replace(/\/+$/, '')

export function apiUrl(path) {
  const normalized = String(path || '')
  const withSlash = normalized.startsWith('/') ? normalized : `/${normalized}`
  return API_BASE ? `${API_BASE}${withSlash}` : withSlash
}
