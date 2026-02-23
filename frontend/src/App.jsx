import { useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import SetupPage from './SetupPage'
import { API_BASE, apiUrl } from './config'

const TABS = ['Bias', 'Claims']

const SCORE_CLASSES = {
  SUPPORTED: 'bg-emerald-100 text-emerald-700',
  PLAUSIBLE: 'bg-amber-100 text-amber-700',
  SPECULATIVE: 'bg-slate-200 text-slate-700',
}

function formatMs(ms) {
  if (!ms || ms < 0) return '0.0s'
  return `${(ms / 1000).toFixed(1)}s`
}

function formatRewriteMarkdown(input) {
  if (!input || !input.trim()) return '# Intel Lint Rewrite\n\n_No rewrite available._\n'
  const trimmed = input.trim()
  const withHeading = /^\s*#/.test(trimmed) ? trimmed : `# Intel Lint Rewrite\n\n${trimmed}`
  return `${withHeading}\n`
}

const REWRITE_MD_COMPONENTS = {
  h1: ({ children }) => <h1 className="mb-5 text-2xl font-bold text-slate-900">{children}</h1>,
  h2: ({ children }) => <h2 className="mb-3 mt-6 text-xl font-semibold text-slate-800">{children}</h2>,
  h3: ({ children }) => <h3 className="mb-2 mt-4 text-lg font-semibold text-slate-700">{children}</h3>,
  p: ({ children }) => <p className="mb-4 leading-7 text-slate-700">{children}</p>,
  ul: ({ children }) => <ul className="mb-4 list-disc space-y-2 pl-6 text-slate-700">{children}</ul>,
  ol: ({ children }) => <ol className="mb-4 list-decimal space-y-2 pl-6 text-slate-700">{children}</ol>,
  li: ({ children }) => <li className="leading-7">{children}</li>,
}

function clampSpan(span, textLength) {
  const start = Math.max(0, Math.min(textLength, Number(span.start || 0)))
  const end = Math.max(0, Math.min(textLength, Number(span.end || 0)))
  if (end <= start) return null
  return { start, end }
}

function normalizeSpanToWord(text, span) {
  if (!span) return null
  let { start, end } = span
  const len = text.length
  while (start > 0 && /\S/.test(text[start - 1])) start -= 1
  while (end < len && /\S/.test(text[end])) end += 1
  return { start, end }
}

function buildAnnotatedSegments(text, claims) {
  if (!text) return []
  const markers = new Uint8Array(text.length)
  const labelStarts = new Map()

  const registerLabel = (offset, claimId) => {
    if (!claimId || offset < 0 || offset >= text.length) return
    const existing = labelStarts.get(offset) || new Set()
    existing.add(claimId)
    labelStarts.set(offset, existing)
  }

  for (const claim of claims || []) {
    const claimId = String(claim.claim_id || '').trim()
    // Highlight the whole claim span to avoid partial sentence highlighting when evidence is short.
    const claimSpan = normalizeSpanToWord(text, clampSpan(claim, text.length))
    if (claimSpan) {
      for (let i = claimSpan.start; i < claimSpan.end; i += 1) markers[i] |= 1
      registerLabel(claimSpan.start, claimId)
    }

    for (const ev of claim.evidence || []) {
      const span = normalizeSpanToWord(text, clampSpan(ev, text.length))
      if (!span) continue
      for (let i = span.start; i < span.end; i += 1) markers[i] |= 1
      registerLabel(span.start, claimId)
    }
    for (const flag of claim.bias_flags || []) {
      const biasCode = String(flag.bias_code || claimId).trim()
      for (const ev of flag.evidence || []) {
        const span = normalizeSpanToWord(text, clampSpan(ev, text.length))
        if (!span) continue
        for (let i = span.start; i < span.end; i += 1) markers[i] |= 2
        registerLabel(span.start, biasCode)
      }
    }
  }

  if (text.length === 0) return []

  const segments = []
  let currentType = markers[0] || 0
  let segmentStart = 0
  for (let i = 1; i < text.length; i += 1) {
    const type = markers[i] || 0
    if (type !== currentType) {
      segments.push({
        text: text.slice(segmentStart, i),
        type: currentType,
        labels: Array.from(labelStarts.get(segmentStart) || []).sort(),
      })
      segmentStart = i
      currentType = type
    }
  }
  segments.push({
    text: text.slice(segmentStart),
    type: currentType,
    labels: Array.from(labelStarts.get(segmentStart) || []).sort(),
  })
  return segments
}

function delay(ms, signal) {
  return new Promise((resolve, reject) => {
    const id = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort)
      resolve()
    }, ms)
    const onAbort = () => {
      clearTimeout(id)
      reject(new DOMException('Aborted', 'AbortError'))
    }
    if (signal) signal.addEventListener('abort', onAbort, { once: true })
  })
}

function toUiError(err) {
  if (err?.name === 'AbortError') return null
  if (err instanceof TypeError && /fetch/i.test(String(err.message || ''))) {
    const target = API_BASE || 'same-origin'
    return `Cannot reach backend at ${target}. Start backend and verify API endpoints respond.`
  }
  return String(err)
}

function normalizeRoutePath(pathname) {
  return pathname === '/setup' ? '/setup' : '/'
}

async function fetchSetupGateState() {
  const [settingsRes, statusRes] = await Promise.all([
    fetch(apiUrl('/api/setup/settings')),
    fetch(apiUrl('/api/setup/ollama/status')),
  ])
  if (!settingsRes.ok) throw new Error(await settingsRes.text())
  if (!statusRes.ok) throw new Error(await statusRes.text())
  const settings = await settingsRes.json()
  const status = await statusRes.json()
  return { settings, status }
}

function applyBiasCorrections(claim) {
  let sentence = claim.text || ''
  const replacements = {
    alarmist: 'significant',
    certainty: 'likely',
    hype: 'notable',
  }
  for (const flag of claim.bias_flags || []) {
    const neutral = replacements[flag.tag] || 'neutral'
    for (const ev of flag.evidence || []) {
      const quote = (ev.quote || '').trim()
      if (!quote) continue
      sentence = sentence.replaceAll(quote, neutral)
    }
  }
  return sentence
}

function normalizeSentence(text) {
  let clean = String(text || '').replace(/\s+/g, ' ').trim()
  if (!clean) return ''
  clean = clean
    .replace(/^#{1,6}\s+/, '')
    .replace(/^>\s+/, '')
    .replace(/^[-*+]\s+/, '')
    .replace(/^\d+[.)]\s+/, '')
    .trim()
  if (!clean) return ''
  if (/[.!?]$/.test(clean)) return clean
  return `${clean}.`
}

function normalizeAnchorCode(value) {
  return String(value || '')
    .replace(/[\[\]\s]/g, '')
    .trim()
    .toUpperCase()
}

function parseSourceSections(sourceText) {
  if (!sourceText || !sourceText.trim()) return []
  const sections = []
  const seen = new Set()
  const lineRegex = /([^\r\n]*)(\r\n|\n|\r|$)/g
  let match = null
  while ((match = lineRegex.exec(sourceText)) !== null) {
    const rawLine = match[1] || ''
    const lineStart = match.index
    const trimmed = rawLine.trim()
    if (!trimmed) {
      if (match[2] === '') break
      continue
    }

    let title = ''
    let headingMatch = trimmed.match(/^#{1,6}\s+(.+?)\s*$/)
    if (headingMatch) {
      title = headingMatch[1]
    } else {
      headingMatch = trimmed.match(/^\d+(?:\.\d+)*[).:-]?\s+(.+?)\s*$/)
      if (headingMatch) {
        title = headingMatch[1]
      } else {
        headingMatch = trimmed.match(/^([A-Z][A-Za-z0-9 /&-]{2,90}):\s*$/)
        if (headingMatch) {
          title = headingMatch[1]
        } else {
          const letters = trimmed.replace(/[^A-Za-z]/g, '')
          const isUpperHeading = letters.length >= 4 && trimmed === trimmed.toUpperCase() && trimmed.length <= 90
          if (isUpperHeading) title = trimmed
        }
      }
    }

    if (!title) {
      if (match[2] === '') break
      continue
    }

    const normalized = title.replace(/\s+/g, ' ').replace(/[:\s]+$/, '').trim()
    const key = normalized.toLowerCase()
    if (!normalized || seen.has(key)) {
      if (match[2] === '') break
      continue
    }
    seen.add(key)
    sections.push({ title: normalized, start: lineStart })
    if (match[2] === '') break
  }

  return sections.sort((a, b) => a.start - b.start)
}

function claimStartOffset(claim) {
  const spans = []
  for (const ev of claim.evidence || []) spans.push(Number(ev.start))
  for (const flag of claim.bias_flags || []) {
    for (const ev of flag.evidence || []) spans.push(Number(ev.start))
  }
  const valid = spans.filter((value) => Number.isFinite(value) && value >= 0)
  if (valid.length === 0) return null
  return Math.min(...valid)
}

function sectionKey(title) {
  return String(title || '').trim().toLowerCase()
}

function mergeSectionTitles(sourceSections) {
  const standard = ['Executive Summary', 'Key Findings', 'Recommended Actions']
  const merged = []
  const seen = new Set()

  for (const title of standard) {
    const key = sectionKey(title)
    if (!seen.has(key)) {
      seen.add(key)
      merged.push({ title, start: -1 })
    }
  }

  for (const section of sourceSections) {
    const key = sectionKey(section.title)
    if (!seen.has(key)) {
      seen.add(key)
      merged.push({ title: section.title, start: section.start })
    }
  }

  return merged
}

function buildExecutiveSummary(correctedClaims) {
  if (correctedClaims.length === 0) {
    return ['No validated claims were available to produce a rewritten summary.']
  }
  return correctedClaims.slice(0, 3).map((item) => normalizeSentence(item.corrected))
}

function buildScopeParagraphs(sourceText, sourceSections, correctedClaims) {
  const paragraphs = []
  if (sourceSections.length > 0) {
    const labels = sourceSections.slice(0, 4).map((section) => section.title)
    paragraphs.push(`This rewrite follows the structure and intent of the source report sections: ${labels.join(', ')}.`)
  }
  if (sourceText && sourceText.trim()) {
    paragraphs.push('Content has been normalized for clarity while preserving evidence-grounded meaning from the original text.')
  }
  if (correctedClaims.length === 0) {
    paragraphs.push('No extracted claims were available, so scope remains limited to high-level normalization.')
  }
  return paragraphs
}

function buildAssessmentParagraphs(correctedClaims) {
  if (correctedClaims.length === 0) {
    return ['No actionable assessment could be produced because no validated claims were available.']
  }
  return correctedClaims.slice(0, 4).map((item) => normalizeSentence(item.corrected))
}

function buildOperationalImpactParagraphs(correctedClaims) {
  if (correctedClaims.length === 0) {
    return ['Operational impact could not be determined from the available material.']
  }
  const highlights = correctedClaims.slice(0, 2).map((item) => normalizeSentence(item.corrected))
  return [
    `The observed activity may create operational risk across monitoring, response, and recovery workflows. ${highlights.join(' ')}`.trim(),
    'Prioritization should focus on controls directly related to the described behaviors and validated evidence spans.',
  ]
}

function buildMitigationActions(correctedClaims) {
  const suggested = []
  for (const item of correctedClaims) {
    for (const fix of item.suggestedFixes || []) {
      const normalized = normalizeSentence(fix)
      if (normalized) suggested.push(normalized)
    }
  }
  const unique = Array.from(new Set(suggested)).slice(0, 6)
  if (unique.length > 0) return unique
  return [
    'Use language grounded in cited evidence and avoid absolute statements without source support.',
    'Separate confirmed observations from assumptions, and mark uncertainty levels explicitly.',
    'Prioritize remediation recommendations that are directly tied to observed evidence spans.',
  ]
}

async function generateRewriteFromCorrections(claims, sourceText, signal) {
  const sourceSections = parseSourceSections(sourceText)
  const sections = mergeSectionTitles(sourceSections)
  const correctedClaims = []
  for (let i = 0; i < claims.length; i += 1) {
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError')
    const claim = claims[i]
    correctedClaims.push({
      corrected: applyBiasCorrections(claim),
      start: claimStartOffset(claim),
      suggestedFixes: (claim.bias_flags || []).map((flag) => String(flag.suggested_fix || '').trim()).filter(Boolean),
    })
    await delay(60, signal)
  }

  const lines = ['# Intel Lint Rewrite', '']
  lines.push('## Executive Summary')
  lines.push('')
  const executiveParagraphs = buildExecutiveSummary(correctedClaims)
  if (executiveParagraphs.length === 1 && executiveParagraphs[0].toLowerCase().startsWith('no validated claims')) {
    lines.push(executiveParagraphs[0])
    lines.push('')
  } else {
    for (const paragraph of executiveParagraphs) {
      lines.push(`- ${paragraph}`)
    }
    lines.push('')
  }

  lines.push('## Scope')
  lines.push('')
  const scopeParagraphs = buildScopeParagraphs(sourceText, sourceSections, correctedClaims)
  for (const paragraph of scopeParagraphs) {
    lines.push(`- ${paragraph}`)
  }
  lines.push('')

  lines.push('## Assessment')
  lines.push('')
  const assessmentParagraphs = buildAssessmentParagraphs(correctedClaims)
  for (const paragraph of assessmentParagraphs) {
    lines.push(`- ${paragraph}`)
  }
  lines.push('')

  lines.push('## Operational Impact')
  lines.push('')
  const impactParagraphs = buildOperationalImpactParagraphs(correctedClaims)
  for (const paragraph of impactParagraphs) {
    lines.push(`- ${paragraph}`)
  }
  lines.push('')

  const bodySections = sections
    .filter((section) => {
      const key = sectionKey(section.title).replace(/\s+/g, ' ').trim()
      return (
        key !== 'executive summary' &&
        key !== 'key findings' &&
        key !== 'scope' &&
        key !== 'assessment' &&
        key !== 'operational impact' &&
        key !== 'recommended actions' &&
        key !== 'mitigations'
      )
    })
    .sort((a, b) => a.start - b.start)

  if (bodySections.length > 0 && correctedClaims.length > 0) {
    lines.push('## Source-Aligned Notes')
    lines.push('')
    for (let i = 0; i < bodySections.length; i += 1) {
      const section = bodySections[i]
      const nextSection = bodySections[i + 1]
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError')
      lines.push(`### ${section.title}`)
      lines.push('')
      const scoped = correctedClaims.filter((item) => {
        if (item.start === null) return false
        if (item.start < section.start) return false
        if (nextSection && item.start >= nextSection.start) return false
        return true
      })
      const scopedSlice = scoped.length > 0 ? scoped.slice(0, 3) : correctedClaims.slice(0, 2)
      for (const item of scopedSlice) {
        lines.push(`- ${normalizeSentence(item.corrected)}`)
      }
      lines.push('')
      await delay(40, signal)
    }
  }

  lines.push('## Mitigations')
  lines.push('')
  const actions = buildMitigationActions(correctedClaims)
  for (const action of actions) {
    lines.push(`- ${normalizeSentence(action)}`)
  }
  lines.push('')

  lines.push('## Method Notes')
  lines.push('')
  lines.push(`- Total claims reviewed: ${correctedClaims.length}.`)
  lines.push('- Rewrite generated from validated claims and bias corrections without re-running extraction.')
  lines.push('')

  return `${lines.join('\n').trim()}\n`
}

export default function App() {
  const [route, setRoute] = useState(() => normalizeRoutePath(window.location.pathname))
  const [gateState, setGateState] = useState({ loading: true, error: '', settings: null, status: null })
  const [text, setText] = useState('')
  const [analyzedText, setAnalyzedText] = useState('')
  const [activeTab, setActiveTab] = useState('Bias')
  const [showAnnotatedLeft, setShowAnnotatedLeft] = useState(false)
  const [loadingAnalysis, setLoadingAnalysis] = useState(false)
  const [loadingRewrite, setLoadingRewrite] = useState(false)
  const [analysisProgress, setAnalysisProgress] = useState(0)
  const [rewriteProgress, setRewriteProgress] = useState(0)
  const [analysisElapsedMs, setAnalysisElapsedMs] = useState(0)
  const [rewriteElapsedMs, setRewriteElapsedMs] = useState(0)
  const [analysisLastMs, setAnalysisLastMs] = useState(0)
  const [rewriteLastMs, setRewriteLastMs] = useState(0)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [rewriteMd, setRewriteMd] = useState('')
  const [ollamaStatus, setOllamaStatus] = useState(null)
  const [stabilityRunning, setStabilityRunning] = useState(false)
  const [stabilityResult, setStabilityResult] = useState(null)
  const [pendingRightAnchor, setPendingRightAnchor] = useState(null)
  const [pendingLeftAnchor, setPendingLeftAnchor] = useState(null)

  const analysisAbortRef = useRef(null)
  const rewriteAbortRef = useRef(null)
  const leftAnnotatedRef = useRef(null)
  const rightPanelRef = useRef(null)

  const navigateTo = (nextPath) => {
    const path = normalizeRoutePath(nextPath)
    if (window.location.pathname !== path) {
      window.history.pushState({}, '', path)
    }
    setRoute(path)
  }

  useEffect(() => {
    const onPopState = () => setRoute(normalizeRoutePath(window.location.pathname))
    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [])

  const refreshGateState = async () => {
    try {
      const data = await fetchSetupGateState()
      setGateState({ loading: false, error: '', ...data })
      const engine = String(data?.settings?.ENGINE || '').toLowerCase()
      const needsSetup = engine === 'ollama' && (!data?.status?.reachable || !data?.status?.model_present)
      if (needsSetup) navigateTo('/setup')
    } catch (err) {
      setGateState({ loading: false, error: toUiError(err) || 'setup check failed', settings: null, status: null })
    }
  }

  useEffect(() => {
    refreshGateState()
  }, [])

  useEffect(() => {
    if (route !== '/') return
    const engine = String(gateState?.settings?.ENGINE || '').toLowerCase()
    if (engine !== 'ollama') return
    if (!gateState?.status?.reachable || !gateState?.status?.model_present) {
      navigateTo('/setup')
    }
  }, [route, gateState])

  useEffect(() => {
    const checkOllama = async () => {
      try {
        const res = await fetch(apiUrl('/health/ollama'))
        if (!res.ok) return
        const data = await res.json()
        setOllamaStatus(data)
      } catch {
        setOllamaStatus(null)
      }
    }
    checkOllama()
  }, [])

  const biasTags = useMemo(() => {
    if (!result?.claims?.claims) return []
    const counts = new Map()
    for (const claim of result.claims.claims) {
      for (const flag of claim.bias_flags || []) {
        counts.set(flag.tag, (counts.get(flag.tag) || 0) + 1)
      }
    }
    return Array.from(counts.entries())
  }, [result])

  const biasFlat = useMemo(() => {
    const items = []
    const claims = result?.claims?.claims || []
    if (claims.length === 0) return items

    const overlapsClaim = (ev, claim) => {
      const s = Number(ev.start || 0)
      const e = Number(ev.end || 0)
      return e > s && e > Number(claim.start || 0) && s < Number(claim.end || 0)
    }

    for (const claim of claims) {
      for (const flag of claim.bias_flags || []) {
        let linked = null
        for (const other of claims) {
          if ((flag.evidence || []).some((ev) => overlapsClaim(ev, other))) {
            linked = other.claim_id
            break
          }
        }
        items.push({
          claimId: linked,
          biasCode: flag.bias_code,
          tag: flag.tag,
          suggestedFix: flag.suggested_fix,
          evidence: flag.evidence || [],
        })
      }
    }
    return items
  }, [result])

  const kpi = useMemo(() => {
    const claims = result?.claims?.claims || []
    const totalClaims = claims.length
    const speculative = claims.filter((claim) => claim.score_label === 'SPECULATIVE').length
    const speculativePct = totalClaims > 0 ? Math.round((speculative / totalClaims) * 100) : 0
    const totalBias = biasTags.reduce((sum, [, count]) => sum + Number(count || 0), 0)
    const biasParts = biasTags.map(([tag, count]) => `${tag}: ${count}`)
    return {
      totalBias,
      totalClaims,
      speculativePct,
      biasByCategory: biasParts.length > 0 ? biasParts.join(' | ') : 'none',
    }
  }, [result, biasTags])

  const annotatedSegments = useMemo(() => {
    if (!analyzedText || !result?.claims?.claims) return []
    return buildAnnotatedSegments(analyzedText, result.claims.claims)
  }, [analyzedText, result])

  const rewriteReady = Boolean(rewriteMd?.trim())

  if (route === '/setup') {
    return <SetupPage onContinue={() => navigateTo('/')} onGateRefresh={refreshGateState} />
  }

  const stopAnalysis = () => analysisAbortRef.current?.abort()
  const stopRewrite = () => rewriteAbortRef.current?.abort()

  const handleAnnotatedAnchorClick = (rawCode) => {
    const code = normalizeAnchorCode(rawCode)
    if (!code) return
    const targetTab = code.startsWith('B') ? 'Bias' : 'Claims'
    setActiveTab(targetTab)
    setPendingRightAnchor(code)
  }

  const handleRightAnchorClick = (rawCode) => {
    const code = normalizeAnchorCode(rawCode)
    if (!code) return
    setShowAnnotatedLeft(true)
    setPendingLeftAnchor(code)
  }

  useEffect(() => {
    if (!pendingRightAnchor || !result) return
    const targetTab = pendingRightAnchor.startsWith('B') ? 'Bias' : 'Claims'
    if (activeTab !== targetTab) return
    const container = rightPanelRef.current
    if (!container) return
    const selector = pendingRightAnchor.startsWith('B')
      ? `[data-bias-code="${pendingRightAnchor}"]`
      : `[data-claim-id="${pendingRightAnchor}"]`
    const node = container.querySelector(selector)
    if (!node) {
      setPendingRightAnchor(null)
      return
    }
    node.scrollIntoView({ behavior: 'smooth', block: 'center' })
    setPendingRightAnchor(null)
  }, [pendingRightAnchor, activeTab, result])

  useEffect(() => {
    if (!pendingLeftAnchor || !showAnnotatedLeft) return
    const container = leftAnnotatedRef.current
    if (!container) return
    const node = container.querySelector(`[data-anchor="${pendingLeftAnchor}"]`)
    if (!node) {
      setPendingLeftAnchor(null)
      return
    }
    node.scrollIntoView({ behavior: 'smooth', block: 'center' })
    setPendingLeftAnchor(null)
  }, [pendingLeftAnchor, showAnnotatedLeft, annotatedSegments])

  async function sha256Hex(str) {
    const data = new TextEncoder().encode(str)
    if (window.crypto?.subtle) {
      const hash = await window.crypto.subtle.digest('SHA-256', data)
      return Array.from(new Uint8Array(hash))
        .map((b) => b.toString(16).padStart(2, '0'))
        .join('')
    }
    // Fallback: simple deterministic checksum (not cryptographic)
    let hash = 0
    for (let i = 0; i < data.length; i += 1) {
      hash = (hash * 31 + data[i]) >>> 0
    }
    return hash.toString(16)
  }

  function normalizePayload(payload) {
    const claims = payload?.claims?.claims || []
    const normClaims = claims
      .map((claim) => {
        const ev = [...(claim.evidence || [])].sort((a, b) => a.start - b.start || a.end - b.end || a.quote.localeCompare(b.quote || ''))
        const flags = [...(claim.bias_flags || [])].map((f) => ({
          ...f,
          evidence: [...(f.evidence || [])].sort((a, b) => a.start - b.start || a.end - b.end || a.quote.localeCompare(b.quote || '')),
        }))
        flags.sort((a, b) => a.tag.localeCompare(b.tag || '') || a.suggested_fix.localeCompare(b.suggested_fix || ''))
        return {
          claim_id: claim.claim_id,
          text: claim.text,
          score_label: claim.score_label,
          evidence: ev,
          bias_flags: flags,
        }
      })
      .sort((a, b) => (a.claim_id || '').localeCompare(b.claim_id || ''))
    return { claims: normClaims }
  }

  function biasCounts(payload) {
    const counts = new Map()
    for (const claim of payload?.claims?.claims || []) {
      for (const flag of claim.bias_flags || []) {
        const tag = flag.tag || 'unknown'
        counts.set(tag, (counts.get(tag) || 0) + 1)
      }
    }
    return Object.fromEntries([...counts.entries()].sort((a, b) => a[0].localeCompare(b[0])))
  }

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError('Insert text to analyze.')
      return
    }

    stopRewrite()
    setLoadingAnalysis(true)
    setAnalysisProgress(2)
    setAnalysisElapsedMs(0)
    setError('')
    setRewriteMd('')

    const controller = new AbortController()
    analysisAbortRef.current = controller

    const started = Date.now()
    const estimatedMs = Math.max(5000, Math.min(120000, text.length * 7))

    const progressTimer = setInterval(() => {
      const elapsed = Date.now() - started
      setAnalysisElapsedMs(elapsed)
      const pct = Math.min(92, Math.round((elapsed / estimatedMs) * 100))
      setAnalysisProgress((prev) => (pct > prev ? pct : prev + (prev < 90 ? 1 : 0)))
    }, 200)

    try {
      const res = await fetch(apiUrl('/analyze'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, generate_rewrite: false }),
        signal: controller.signal,
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      const elapsed = Date.now() - started
      setResult(data)
      setAnalyzedText(text)
      setAnalysisLastMs(elapsed)
      setAnalysisElapsedMs(elapsed)
      setAnalysisProgress(100)
      setShowAnnotatedLeft(true)
      setActiveTab('Bias')
    } catch (err) {
      if (err?.name === 'AbortError') {
        setError('Analysis stopped by user.')
      } else {
        setError(toUiError(err))
      }
    } finally {
      clearInterval(progressTimer)
      analysisAbortRef.current = null
      setTimeout(() => {
        setLoadingAnalysis(false)
        setAnalysisProgress(0)
      }, 200)
    }
  }

  const handleGenerateRewrite = async () => {
    const claims = result?.claims?.claims || []
    if (claims.length === 0) {
      setError('Run analysis first.')
      return
    }

    setLoadingRewrite(true)
    setRewriteProgress(2)
    setRewriteElapsedMs(0)
    setError('')

    const controller = new AbortController()
    rewriteAbortRef.current = controller

    const started = Date.now()
    const estimatedMs = Math.max(1200, claims.length * 220)
    const progressTimer = setInterval(() => {
      const elapsed = Date.now() - started
      setRewriteElapsedMs(elapsed)
      const pct = Math.min(96, Math.round((elapsed / estimatedMs) * 100))
      setRewriteProgress((prev) => (pct > prev ? pct : prev + (prev < 95 ? 1 : 0)))
    }, 120)

    try {
      const md = await generateRewriteFromCorrections(claims, analyzedText, controller.signal)
      const elapsed = Date.now() - started
      setRewriteMd(md)
      setRewriteElapsedMs(elapsed)
      setRewriteLastMs(elapsed)
      setRewriteProgress(100)
      setActiveTab('Rewrite')
    } catch (err) {
      if (err?.name === 'AbortError') {
        setError('Rewrite generation stopped by user.')
      } else {
        setError(toUiError(err))
      }
    } finally {
      clearInterval(progressTimer)
      rewriteAbortRef.current = null
      setTimeout(() => {
        setLoadingRewrite(false)
        setRewriteProgress(0)
      }, 200)
    }
  }

  const handleStabilityTest = async (runs = 5) => {
    if (!text.trim()) {
      setError('Insert text to analyze before running stability test.')
      return
    }
    setError('')
    setStabilityRunning(true)
    setStabilityResult(null)
    const digests = []
    const summaries = []
    for (let i = 0; i < runs; i += 1) {
      try {
        const res = await fetch(apiUrl('/analyze'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, generate_rewrite: false }),
        })
        if (!res.ok) throw new Error(await res.text())
        const payload = await res.json()
        const digest = await sha256Hex(JSON.stringify(normalizePayload(payload)))
        digests.push(digest)
        summaries.push({
          run: i + 1,
          digest,
          bias: biasCounts(payload),
          claims: (payload.claims?.claims || []).map((c) => c.claim_id),
        })
      } catch (err) {
        setStabilityRunning(false)
        setError(toUiError(err))
        return
      }
    }
    const unique = Array.from(new Set(digests))
    const baselineBias = summaries[0]?.bias || {}
    const baselineClaims = summaries[0]?.claims || []
    const divergences = summaries.filter(
      (s) => JSON.stringify(s.bias) !== JSON.stringify(baselineBias) || JSON.stringify(s.claims) !== JSON.stringify(baselineClaims)
    )
    setStabilityResult({
      runs,
      uniqueCount: unique.length,
      digests: unique,
      divergences,
    })
    setStabilityRunning(false)
  }

  const handleDownloadZip = async () => {
    try {
      const res = await fetch(apiUrl('/download/latest'))
      if (!res.ok) throw new Error(await res.text())
      const blob = await res.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'intel-lint-latest.zip'
      document.body.appendChild(a)
      a.click()
      a.remove()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      setError(toUiError(err))
    }
  }

  return (
    <div className="mx-auto min-h-screen max-w-7xl p-4 md:p-8">
      <header className="mb-6 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-3xl font-bold text-ink md:text-4xl">Intel Lint</h1>
          <p className="mt-1 text-sm text-slate-600">LLM-first claim analysis with evidence guardrails</p>
          {ollamaStatus && (
            <p className="mt-2 text-xs text-slate-500">
              Ollama: {ollamaStatus.reachable ? 'reachable' : 'unreachable'}
              {ollamaStatus.reachable && ` | processor: ${ollamaStatus.processor_summary || 'unknown'}`}
            </p>
          )}
          {gateState.error && <p className="mt-2 text-xs text-red-600">{gateState.error}</p>}
        </div>
        <button
          onClick={() => navigateTo('/setup')}
          className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700"
        >
          Setup
        </button>
      </header>

      <section className="mb-4 grid gap-3 md:grid-cols-4">
        <div className="rounded-xl border border-white/70 bg-white/85 p-4 shadow-panel">
          <p className="text-xs uppercase tracking-wide text-slate-500">Total Bias Flags</p>
          <p className="mt-1 text-2xl font-bold text-coral">{kpi.totalBias}</p>
        </div>
        <div className="rounded-xl border border-white/70 bg-white/85 p-4 shadow-panel">
          <p className="text-xs uppercase tracking-wide text-slate-500">Bias by Category</p>
          <p className="mt-1 text-sm font-medium text-slate-700 whitespace-pre-wrap break-words">{kpi.biasByCategory}</p>
        </div>
        <div className="rounded-xl border border-white/70 bg-white/85 p-4 shadow-panel">
          <p className="text-xs uppercase tracking-wide text-slate-500">Total Claims</p>
          <p className="mt-1 text-2xl font-bold text-ink">{kpi.totalClaims}</p>
        </div>
        <div className="rounded-xl border border-white/70 bg-white/85 p-4 shadow-panel">
          <p className="text-xs uppercase tracking-wide text-slate-500">% Speculative</p>
          <p className="mt-1 text-2xl font-bold text-coral">{kpi.speculativePct}%</p>
        </div>
      </section>

      <div className="grid gap-4 md:grid-cols-2">
        <section className="rounded-2xl border border-white/70 bg-white/80 p-5 shadow-panel backdrop-blur">
          <div className="mb-3 flex items-center gap-2">
            {!loadingAnalysis && (
              <button
                onClick={handleAnalyze}
                className="inline-flex items-center gap-2 rounded-lg bg-coral px-4 py-2 text-sm font-semibold text-white transition hover:opacity-90"
              >
                Analyze
              </button>
            )}
            {!loadingAnalysis && result && (
              <button
                onClick={() => setShowAnnotatedLeft((value) => !value)}
                className="inline-flex items-center gap-2 rounded-lg border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700"
              >
                {showAnnotatedLeft ? 'Edit input' : 'Show annotated'}
              </button>
            )}
            {loadingAnalysis && (
              <button
                onClick={stopAnalysis}
                className="inline-flex items-center gap-2 rounded-lg bg-slate-700 px-4 py-2 text-sm font-semibold text-white"
              >
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/60 border-t-white" />
                Stop analysis
              </button>
            )}
          </div>

          {loadingAnalysis && (
            <div className="mb-3">
              <div className="mb-1 flex items-center justify-between text-xs text-slate-500">
                <span>Analyzing</span>
                <span>{analysisProgress}% | {formatMs(analysisElapsedMs)}</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200">
                <div className="h-full rounded-full bg-tide transition-all duration-300" style={{ width: `${analysisProgress}%` }} />
              </div>
            </div>
          )}

          {!loadingAnalysis && analysisLastMs > 0 && (
            <p className="mb-3 text-xs text-slate-500">Last analysis time: {formatMs(analysisLastMs)}</p>
          )}

          {showAnnotatedLeft && result ? (
            <div ref={leftAnnotatedRef} className="h-[460px] overflow-auto rounded-xl border border-slate-200 bg-white p-4">
              <div className="mb-3 flex flex-wrap gap-2 text-xs">
                <span className="rounded-full bg-tide/20 px-2 py-1 text-tide">Claim evidence</span>
                <span className="rounded-full bg-coral/20 px-2 py-1 text-coral">Bias evidence</span>
                <span className="rounded-full bg-amber-200 px-2 py-1 text-amber-800">Claim + Bias overlap</span>
              </div>
              <pre className="whitespace-pre-wrap break-words rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-800">
                {annotatedSegments.map((segment, idx) => {
                  let klass = ''
                  if (segment.type === 1) klass = 'bg-tide/20'
                  if (segment.type === 2) klass = 'bg-coral/20'
                  if (segment.type === 3) klass = 'bg-amber-200'
                  return (
                    <span key={`left-segment-${idx}`}>
                      {segment.labels.length > 0 && (
                        <span className="mr-1 inline-flex flex-wrap gap-1 rounded bg-slate-800 px-1 py-0.5 text-[10px] font-semibold text-white">
                          {segment.labels.map((label) => (
                            <button
                              key={`label-${idx}-${label}`}
                              type="button"
                              data-anchor={normalizeAnchorCode(label)}
                              onClick={() => handleAnnotatedAnchorClick(label)}
                              className="rounded px-1 text-white underline-offset-2 hover:underline focus:outline-none focus:ring-1 focus:ring-white"
                            >
                              [{label}]
                            </button>
                          ))}
                        </span>
                      )}
                      <span className={klass}>{segment.text}</span>
                    </span>
                  )
                })}
              </pre>
            </div>
          ) : (
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste text to analyze..."
              className="h-[460px] w-full resize-none rounded-xl border border-slate-300 bg-mist p-3 text-sm outline-none focus:ring-2 focus:ring-tide"
            />
          )}
        </section>

        <section className="rounded-2xl border border-white/70 bg-white/85 p-5 shadow-panel backdrop-blur">
          <div className="mb-4 flex flex-wrap items-center gap-2">
            {TABS.map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`rounded-full px-3 py-1.5 text-sm font-medium ${
                  activeTab === tab ? 'bg-ink text-white' : 'bg-slate-100 text-slate-600'
                }`}
              >
                {tab}
              </button>
            ))}

            {rewriteReady && (
              <button
                onClick={() => setActiveTab('Rewrite')}
                className={`rounded-full px-3 py-1.5 text-sm font-medium ${
                  activeTab === 'Rewrite' ? 'bg-ink text-white' : 'bg-slate-100 text-slate-600'
                }`}
              >
                Rewrite
              </button>
            )}

            {!loadingRewrite && (
              <button
                onClick={handleGenerateRewrite}
                disabled={!result || loadingAnalysis}
                className="rounded-lg border border-tide px-3 py-1.5 text-sm font-medium text-tide disabled:opacity-50"
              >
                Generate rewrite
              </button>
            )}

            {loadingRewrite && (
              <button
                onClick={stopRewrite}
                className="rounded-lg border border-slate-600 px-3 py-1.5 text-sm font-medium text-slate-700"
              >
                Stop generation
              </button>
            )}

            <button onClick={handleDownloadZip} className="ml-auto rounded-lg border border-ink px-3 py-1.5 text-sm font-medium text-ink">
              Download ZIP
            </button>
          </div>

          {loadingRewrite && (
            <div className="mb-3">
              <div className="mb-1 flex items-center justify-between text-xs text-slate-500">
                <span>Generating rewrite</span>
                <span>{rewriteProgress}% | {formatMs(rewriteElapsedMs)}</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200">
                <div className="h-full rounded-full bg-coral transition-all duration-300" style={{ width: `${rewriteProgress}%` }} />
              </div>
            </div>
          )}

          {!loadingRewrite && rewriteLastMs > 0 && (
            <p className="mb-3 text-xs text-slate-500">Last rewrite generation time: {formatMs(rewriteLastMs)}</p>
          )}

          {error && <p className="mb-3 whitespace-pre-wrap break-words rounded-lg bg-red-50 p-2 text-sm text-red-600">{error}</p>}
          {!error && result?.warning && (
            <p className="mb-3 whitespace-pre-wrap break-words rounded-lg bg-amber-50 p-2 text-sm text-amber-700">
              {result.warning}
            </p>
          )}
          {stabilityResult && (
            <div className="mb-3 rounded-lg border border-indigo-100 bg-indigo-50 p-3 text-sm text-indigo-800">
              <p className="font-semibold">Stability (runs: {stabilityResult.runs})</p>
              <p className="mt-1">Unique digests: {stabilityResult.uniqueCount}</p>
              {stabilityResult.uniqueCount > 1 && (
                <>
                  <p className="mt-1 text-xs">Digests: {stabilityResult.digests.map((d) => d.slice(0, 12)).join(', ')}</p>
                  {stabilityResult.divergences.length > 0 && (
                    <div className="mt-2 space-y-1 text-xs">
                      {stabilityResult.divergences.map((d) => (
                        <p key={`div-${d.run}`}>
                          Run {d.run}: bias {JSON.stringify(d.bias)} | claims {d.claims.join(', ')}
                        </p>
                      ))}
                    </div>
                  )}
                </>
              )}
              {stabilityResult.uniqueCount === 1 && <p className="mt-1 text-xs">All runs identical.</p>}
            </div>
          )}

          <div ref={rightPanelRef} className="max-h-[500px] overflow-auto rounded-xl border border-slate-200 bg-white p-4">
            {!result && <p className="text-sm text-slate-500">Run analysis to view outputs.</p>}

            {result && activeTab === 'Rewrite' && rewriteReady && (
              <article className="max-w-none break-words rounded-lg border border-slate-200 bg-slate-50 p-4">
                <ReactMarkdown components={REWRITE_MD_COMPONENTS}>{formatRewriteMarkdown(rewriteMd)}</ReactMarkdown>
              </article>
            )}

            {result && activeTab === 'Claims' && (
              <div className="space-y-3">
                {result.claims.claims.map((claim) => {
                  const evidenceStatus = claim.evidence_status || (claim.evidence.length > 0 ? 'anchored' : 'missing')
                  const evidenceTexts =
                    Array.isArray(claim.evidence_texts) && claim.evidence_texts.length > 0
                      ? claim.evidence_texts
                      : (claim.evidence || []).map((ev) => ev.quote)
                  const evidenceNote = claim.evidence_note || '--- no strong evidence reported ---'
                  return (
                    <article
                      key={claim.claim_id}
                      data-claim-id={normalizeAnchorCode(claim.claim_id)}
                      className="rounded-xl border border-slate-200 bg-slate-50 p-3 shadow-sm"
                    >
                      <div className="flex flex-wrap items-center gap-2">
                        <button
                          type="button"
                          onClick={() => handleRightAnchorClick(claim.claim_id)}
                          className="rounded-full bg-slate-800 px-2 py-0.5 text-xs font-semibold text-white underline-offset-2 hover:underline focus:outline-none focus:ring-2 focus:ring-slate-400"
                        >
                          {claim.claim_id}
                        </button>
                        <span
                          className={`rounded-full px-2 py-1 text-xs font-semibold ${SCORE_CLASSES[claim.score_label] || SCORE_CLASSES.SPECULATIVE}`}
                        >
                          {claim.score_label}
                        </span>
                      </div>
                      <p className="mt-2 text-sm font-medium text-slate-900 whitespace-pre-wrap break-words">{claim.text}</p>
                      <div className="mt-2 space-y-1">
                        <p className="text-[11px] uppercase tracking-wide text-slate-500">Evidence</p>
                        {evidenceStatus === 'missing' && <p className="text-xs text-slate-500">{evidenceNote}</p>}
                        {evidenceStatus !== 'missing' &&
                          evidenceTexts.slice(0, 3).map((text, idx) => (
                            <p
                              key={`${claim.claim_id}-evidence-${idx}`}
                              className="rounded-md bg-white px-2 py-1 text-xs text-slate-700 shadow-[inset_0_0_0_1px_rgba(0,0,0,0.04)]"
                            >
                              "{text}"
                            </p>
                          ))}
                      </div>
                    </article>
                  )
                })}
              </div>
            )}

            {result && activeTab === 'Bias' && (
              <div>
                <div className="mb-4 flex flex-wrap gap-2">
                  {biasTags.length === 0 && <span className="text-sm text-slate-500">No bias tags detected.</span>}
                  {biasTags.map(([tag, count]) => (
                    <span key={tag} className="rounded-full bg-coral/15 px-3 py-1 text-sm text-coral">
                      {tag} ({count})
                    </span>
                  ))}
                </div>
                <div className="space-y-3">
                  {biasFlat.length === 0 && <span className="text-sm text-slate-500">No bias flags.</span>}
                  {biasFlat.map((flag, idx) => {
                    const key = `${flag.claimId}-${flag.tag}-${idx}`
                    return (
                      <div
                        key={key}
                        data-bias-code={flag.biasCode ? normalizeAnchorCode(flag.biasCode) : undefined}
                        className="rounded-md border border-coral/20 bg-coral/10 p-3 text-xs text-slate-700"
                      >
                        <p className="font-semibold text-coral">
                          {flag.biasCode && (
                            <button
                              type="button"
                              onClick={() => handleRightAnchorClick(flag.biasCode)}
                              className="mr-1 rounded px-1 underline-offset-2 hover:underline focus:outline-none focus:ring-1 focus:ring-coral"
                            >
                              [{flag.biasCode}]
                            </button>
                          )}
                          <span>{flag.tag}</span>
                        </p>
                        {flag.claimId && (
                          <p className="text-[11px] text-amber-700">
                            Overlaps{' '}
                            <button
                              type="button"
                              onClick={() => handleRightAnchorClick(flag.claimId)}
                              className="rounded px-1 underline-offset-2 hover:underline focus:outline-none focus:ring-1 focus:ring-amber-600"
                            >
                              {flag.claimId}
                            </button>
                          </p>
                        )}
                        <p className="mt-1 whitespace-pre-wrap break-words">
                          <span className="font-medium">Suggested fix:</span> {flag.suggestedFix}
                        </p>
                        <div className="mt-1 space-y-1">
                          {flag.evidence.length === 0 && <p className="text-slate-400">no evidence provided</p>}
                          {flag.evidence.map((ev, evIdx) => (
                            <p key={`${key}-ev-${evIdx}`} className="whitespace-pre-wrap break-words" title={`span ${ev.start}:${ev.end}`}>
                              "{ev.quote}"
                            </p>
                          ))}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  )
}
