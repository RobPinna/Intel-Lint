import { useEffect, useMemo, useRef, useState } from 'react'
import { API_BASE, apiUrl } from './config'

function toUiError(err) {
  if (err?.name === 'AbortError') return null
  if (err instanceof TypeError && /fetch/i.test(String(err.message || ''))) {
    const target = API_BASE || 'same-origin'
    return `Cannot reach backend at ${target}. Start backend and verify /api/health responds.`
  }
  return String(err)
}

async function fetchJson(path, options = undefined) {
  const response = await fetch(apiUrl(path), options)
  if (!response.ok) {
    const body = await response.text()
    throw new Error(body || `HTTP ${response.status}`)
  }
  return response.json()
}

export default function SetupPage({ onContinue, onGateRefresh }) {
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [pulling, setPulling] = useState(false)
  const [error, setError] = useState('')
  const [health, setHealth] = useState(null)
  const [status, setStatus] = useState(null)
  const [settings, setSettings] = useState(null)
  const [form, setForm] = useState({
    ENGINE: 'ollama',
    OLLAMA_URL: 'http://localhost:11434',
    MODEL: 'FenkoHQ/Foundation-Sec-8B',
  })
  const [pullJob, setPullJob] = useState(null)
  const pollTimerRef = useRef(null)

  const stopPolling = () => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }
  }

  const loadData = async () => {
    setLoading(true)
    setError('')
    try {
      const [healthData, settingsData, statusData] = await Promise.all([
        fetchJson('/api/health'),
        fetchJson('/api/setup/settings'),
        fetchJson('/api/setup/ollama/status'),
      ])
      setHealth(healthData)
      setSettings(settingsData)
      setStatus(statusData)
      setForm({
        ENGINE: String(settingsData.ENGINE || 'ollama').toLowerCase(),
        OLLAMA_URL: settingsData.OLLAMA_URL || 'http://localhost:11434',
        MODEL: settingsData.MODEL || 'FenkoHQ/Foundation-Sec-8B',
      })
    } catch (err) {
      setError(toUiError(err))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
    return stopPolling
  }, [])

  const engine = String(form.ENGINE || 'ollama').toLowerCase()
  const canContinue = useMemo(() => {
    if (engine === 'placeholder') return true
    return Boolean(status?.reachable && status?.model_present)
  }, [engine, status])

  const saveSettings = async () => {
    setSaving(true)
    setError('')
    try {
      const payload = {
        ENGINE: form.ENGINE,
        OLLAMA_URL: form.OLLAMA_URL,
        MODEL: form.MODEL,
      }
      const updated = await fetchJson('/api/setup/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      setSettings(updated)
      const nextStatus = await fetchJson('/api/setup/ollama/status')
      setStatus(nextStatus)
      onGateRefresh?.()
    } catch (err) {
      setError(toUiError(err))
    } finally {
      setSaving(false)
    }
  }

  const pollPullJob = (jobId) => {
    stopPolling()
    pollTimerRef.current = setInterval(async () => {
      try {
        const state = await fetchJson(`/api/setup/ollama/pull/${jobId}`)
        setPullJob((prev) => ({
          ...(prev || {}),
          job_id: jobId,
          state: state.state,
          last_line: state.last_line || '',
          exit_code: state.exit_code,
        }))
        if (state.state === 'done' || state.state === 'error') {
          stopPolling()
          setPulling(false)
          const nextStatus = await fetchJson('/api/setup/ollama/status')
          setStatus(nextStatus)
          onGateRefresh?.()
        }
      } catch (err) {
        stopPolling()
        setPulling(false)
        setError(toUiError(err))
      }
    }, 1000)
  }

  const startPull = async () => {
    setPulling(true)
    setError('')
    try {
      const payload = {
        model: form.MODEL,
        ollama_url: form.OLLAMA_URL,
      }
      const job = await fetchJson('/api/setup/ollama/pull', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      setPullJob({
        job_id: job.job_id,
        state: job.state,
        last_line: job.last_line || '',
      })
      pollPullJob(job.job_id)
    } catch (err) {
      setPulling(false)
      setError(toUiError(err))
    }
  }

  return (
    <div className="mx-auto min-h-screen max-w-5xl p-4 md:p-8">
      <header className="mb-6 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-3xl font-bold text-ink md:text-4xl">Intel Lint Setup</h1>
          <p className="mt-1 text-sm text-slate-600">Local-first onboarding for Ollama mode</p>
        </div>
        <button
          onClick={onContinue}
          disabled={!canContinue}
          className="rounded-lg border border-ink px-4 py-2 text-sm font-semibold text-ink disabled:cursor-not-allowed disabled:opacity-40"
        >
          Continue to app
        </button>
      </header>

      {engine === 'placeholder' && (
        <p className="mb-4 rounded-lg border border-amber-200 bg-amber-50 p-2 text-sm text-amber-700">
          Placeholder engine is tests/smoke only.
        </p>
      )}

      {error && <p className="mb-4 rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</p>}

      <section className="mb-5 grid gap-3 md:grid-cols-3">
        <article className="rounded-xl border border-white/70 bg-white/85 p-4 shadow-panel">
          <p className="text-xs uppercase tracking-wide text-slate-500">Backend</p>
          <p className="mt-1 text-sm font-semibold text-slate-800">{health?.ok ? 'Reachable' : loading ? 'Checking...' : 'Unavailable'}</p>
        </article>
        <article className="rounded-xl border border-white/70 bg-white/85 p-4 shadow-panel">
          <p className="text-xs uppercase tracking-wide text-slate-500">Ollama</p>
          <p className="mt-1 text-sm font-semibold text-slate-800">{status?.reachable ? 'Reachable' : loading ? 'Checking...' : 'Unreachable'}</p>
          <p className="mt-1 text-xs text-slate-600 break-all">{status?.ollama_url || form.OLLAMA_URL}</p>
        </article>
        <article className="rounded-xl border border-white/70 bg-white/85 p-4 shadow-panel">
          <p className="text-xs uppercase tracking-wide text-slate-500">Model</p>
          <p className="mt-1 text-sm font-semibold text-slate-800">{status?.model || form.MODEL}</p>
          <p className="mt-1 text-xs text-slate-600">{status?.model_present ? 'Present locally' : loading ? 'Checking...' : 'Not present'}</p>
        </article>
      </section>

      <section className="mb-5 rounded-2xl border border-white/70 bg-white/85 p-5 shadow-panel">
        <h2 className="mb-4 text-lg font-semibold text-slate-800">Settings</h2>
        <div className="grid gap-3 md:grid-cols-3">
          <label className="flex flex-col gap-1 text-sm text-slate-700">
            ENGINE
            <select
              value={form.ENGINE}
              onChange={(e) => setForm((prev) => ({ ...prev, ENGINE: e.target.value }))}
              className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm"
            >
              <option value="ollama">ollama</option>
              <option value="placeholder">placeholder</option>
            </select>
          </label>
          <label className="flex flex-col gap-1 text-sm text-slate-700">
            OLLAMA_URL
            <input
              type="text"
              value={form.OLLAMA_URL}
              onChange={(e) => setForm((prev) => ({ ...prev, OLLAMA_URL: e.target.value }))}
              className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm"
              placeholder="http://localhost:11434"
            />
          </label>
          <label className="flex flex-col gap-1 text-sm text-slate-700">
            MODEL
            <input
              type="text"
              value={form.MODEL}
              onChange={(e) => setForm((prev) => ({ ...prev, MODEL: e.target.value }))}
              className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm"
              placeholder="FenkoHQ/Foundation-Sec-8B"
            />
          </label>
        </div>
        <div className="mt-4">
          <button
            onClick={saveSettings}
            disabled={saving}
            className="rounded-lg bg-coral px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
          >
            {saving ? 'Saving...' : 'Save settings'}
          </button>
        </div>
      </section>

      <section className="rounded-2xl border border-white/70 bg-white/85 p-5 shadow-panel">
        <h2 className="mb-3 text-lg font-semibold text-slate-800">Pull model</h2>
        <button
          onClick={startPull}
          disabled={pulling}
          className="rounded-lg border border-ink px-4 py-2 text-sm font-semibold text-ink disabled:opacity-50"
        >
          {pulling ? 'Pulling...' : 'Pull model'}
        </button>
        {pullJob && (
          <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
            <p>
              <span className="font-semibold">State:</span> {pullJob.state}
            </p>
            <p className="mt-1 break-all">
              <span className="font-semibold">Last line:</span> {pullJob.last_line || '-'}
            </p>
            {pullJob.exit_code !== undefined && pullJob.exit_code !== null && (
              <p className="mt-1">
                <span className="font-semibold">Exit code:</span> {pullJob.exit_code}
              </p>
            )}
          </div>
        )}
      </section>
    </div>
  )
}
