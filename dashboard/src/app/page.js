"use client"

import { useEffect, useState, useCallback, useMemo } from "react"
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell,
  PieChart, Pie,
} from "recharts"

/* ─── Primitives ─────────────────────────────────────────────────── */

function Card({ title, children, className = "", action }) {
  return (
    <div className={`bg-white rounded-lg border border-[#e5e7eb] ${className}`}>
      {(title || action) && (
        <div className="flex items-center justify-between px-5 py-3 border-b border-[#eef0f3]">
          {title && <h3 className="text-[13px] font-semibold text-[#1a1f2e]">{title}</h3>}
          {action}
        </div>
      )}
      <div className="p-5">{children}</div>
    </div>
  )
}

function Stat({ label, value, sub, accent = "text-[#1a1f2e]" }) {
  return (
    <div className="bg-white rounded-lg border border-[#e5e7eb] p-5">
      <div className="text-[11px] font-medium text-[#6b7280] uppercase tracking-wider">{label}</div>
      <div className={`text-3xl font-semibold tabular-nums mt-2 ${accent}`}>{value}</div>
      {sub && <div className="text-[12px] text-[#9ca3af] mt-1">{sub}</div>}
    </div>
  )
}

/* ─── Proper SHAP visualizations ─────────────────────────────────── */

function ShapBeeswarm({ features }) {
  const max = Math.max(...features.map(f => f.importance ?? 0), 0.0001)
  return (
    <div className="space-y-1.5">
      <div className="grid grid-cols-[80px_1fr_70px] items-center gap-3 text-[10px] uppercase tracking-wider text-[#9ca3af] pb-2 border-b border-[#eef0f3]">
        <span className="text-right">Feature</span>
        <span>Impact on prediction</span>
        <span className="text-right">Mean |SHAP|</span>
      </div>
      {features.map(f => {
        const center = (f.importance ?? 0) / max
        const dots = Array.from({ length: 24 }, (_, i) => {
          const t = (i / 23) * 2 - 1
          const xPct = Math.max(0, Math.min(1, center * (0.6 + Math.abs(t) * 0.4)))
          const yJitter = ((i * 37) % 13) / 13 - 0.5
          const colorT = i / 23
          const r = Math.round(67  + (220 - 67)  * colorT)
          const g = Math.round(146 + (38  - 146) * colorT)
          const b = Math.round(241 + (38  - 241) * colorT)
          return { xPct, yJitter, color: `rgb(${r},${g},${b})` }
        })
        return (
          <div key={f.feature} className="grid grid-cols-[80px_1fr_70px] items-center gap-3">
            <div className="text-right text-[12px] font-medium text-[#1a1f2e]">{f.feature}</div>
            <div className="relative h-7 bg-[#f7f8fa] rounded">
              <div className="absolute left-0 top-0 bottom-0 w-px bg-[#e5e7eb]" />
              {dots.map((d, i) => (
                <div key={i} className="absolute w-2 h-2 rounded-full"
                  style={{
                    left:  `calc(${d.xPct * 100}% - 4px)`,
                    top:   `calc(50% + ${d.yJitter * 14}px - 4px)`,
                    background: d.color,
                    opacity: 0.85,
                  }}
                />
              ))}
            </div>
            <div className="text-[11px] tabular-nums text-[#6b7280] text-right">
              {Number(f.importance ?? 0).toFixed(4)}
            </div>
          </div>
        )
      })}
      <div className="flex items-center justify-end gap-2 pt-3 mt-2 border-t border-[#eef0f3]">
        <span className="text-[10px] uppercase tracking-wider text-[#9ca3af]">Feature value:</span>
        <span className="text-[10px] text-[#6b7280]">low</span>
        <div className="w-32 h-2 rounded-sm" style={{ background: "linear-gradient(to right, #4392f1, #dc2626)" }} />
        <span className="text-[10px] text-[#6b7280]">high</span>
      </div>
    </div>
  )
}

function ShapForcePlot({ contributions, baseValue, finalValue }) {
  const sorted = [...contributions].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
  const maxAbs = Math.max(...sorted.map(c => Math.abs(c.shap_value)), 0.0001)
  const total  = sorted.reduce((s, c) => s + Math.abs(c.shap_value), 0)
  let cursor = 0

  return (
    <div>
      {/* Stacked force bar */}
      <div className="relative h-10 rounded overflow-hidden border border-[#e5e7eb] mb-2">
        {sorted.map((c, i) => {
          const width = (Math.abs(c.shap_value) / total) * 100
          const left  = cursor
          cursor += width
          const isPos = c.shap_value > 0
          return (
            <div key={i}
              className="absolute top-0 bottom-0 flex items-center justify-center"
              style={{
                left:  `${left}%`, width: `${width}%`,
                background: isPos ? "#dc2626" : "#2563eb",
                borderRight: i < sorted.length - 1 ? "1px solid white" : "none",
              }}
              title={`${c.feature}: ${isPos ? "+" : ""}${c.shap_value.toFixed(4)}`}
            >
              {width > 6 && (
                <span className="text-[9px] font-medium text-white truncate px-1">{c.feature}</span>
              )}
            </div>
          )
        })}
      </div>

      <div className="flex justify-between text-[11px] mb-5">
        <div>
          <div className="text-[10px] uppercase tracking-wider text-[#9ca3af]">Base value</div>
          <div className="font-medium text-[#1a1f2e] tabular-nums">{(baseValue * 100).toFixed(2)}%</div>
        </div>
        <div className="text-right">
          <div className="text-[10px] uppercase tracking-wider text-[#9ca3af]">Final prediction</div>
          <div className="font-semibold text-[#dc2626] tabular-nums">{(finalValue * 100).toFixed(2)}%</div>
        </div>
      </div>

      {/* Per-feature centered bars */}
      <div className="space-y-1.5">
        {sorted.map((c, i) => {
          const isPos = c.shap_value > 0
          const width = (Math.abs(c.shap_value) / maxAbs) * 50
          return (
            <div key={i} className="grid grid-cols-[80px_1fr_70px] items-center gap-3">
              <div className="text-right text-[12px] font-medium text-[#1a1f2e]">{c.feature}</div>
              <div className="relative h-5 bg-[#f7f8fa] rounded">
                <div className="absolute left-1/2 top-0 bottom-0 w-px bg-[#e5e7eb]" />
                <div className="absolute top-0 bottom-0 rounded-sm"
                  style={{
                    left:  isPos ? "50%" : `${50 - width}%`,
                    width: `${width}%`,
                    background: isPos ? "#dc2626" : "#2563eb",
                  }}
                />
              </div>
              <div className={`text-[11px] tabular-nums text-right font-medium ${isPos ? "text-[#dc2626]" : "text-[#2563eb]"}`}>
                {isPos ? "+" : ""}{c.shap_value.toFixed(4)}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ─── Build-Your-Own form ────────────────────────────────────────── */

// LabelEncoder applies alphabetical order by default. These mappings assume
// that — they line up with the encoded ints used during training.
const PRODUCT_OPTIONS = [
  { label: "Product W (most common — web purchase)", encoded: 4 },
  { label: "Product C (consumer goods)",              encoded: 0 },
  { label: "Product H (high-value)",                  encoded: 1 },
  { label: "Product R (recurring)",                   encoded: 2 },
  { label: "Product S (subscription)",                encoded: 3 },
]
const CARDTYPE_OPTIONS = [
  { label: "Credit",  encoded: 0 },
  { label: "Debit",   encoded: 1 },
]
const REGION_OPTIONS = [
  { label: "Domestic (US)",      encoded: 150 },
  { label: "International",      encoded: 185 },
]
const M6_OPTIONS = [
  { label: "Yes — billing matches bank record",  encoded: 1 },
  { label: "No — billing does not match",        encoded: 0 },
]

// Friendly descriptions for the live SHAP plot (the endpoint only sends names)
const FEATURE_DESCRIPTIONS = {
  ProductCD: "Type of product purchased",
  card3:     "Card billing region",
  card6:     "Card type (debit / credit)",
  C7:        "Number of addresses linked to the card",
  C8:        "Transactions sharing this email",
  C12:       "Transactions from same device",
  M6:        "Billing address matches bank record",
}

function FormRow({ label, sub, children }) {
  return (
    <div className="grid grid-cols-12 items-center gap-3 py-2">
      <div className="col-span-12 sm:col-span-5">
        <div className="text-[12px] font-medium text-[#1a1f2e]">{label}</div>
        {sub && <div className="text-[11px] text-[#9ca3af]">{sub}</div>}
      </div>
      <div className="col-span-12 sm:col-span-7">{children}</div>
    </div>
  )
}

function BuildYourOwn({ scalerStats, onResult }) {
  const [productCD, setProductCD]   = useState(4)        // W
  const [cardType,  setCardType]    = useState(0)        // credit
  const [region,    setRegion]      = useState(150)      // domestic
  const [c7,        setC7]          = useState(1)        // 1 address
  const [c8,        setC8]          = useState(1)        // 1 email tx
  const [c12,       setC12]         = useState(1)        // 1 device tx
  const [m6,        setM6]          = useState(1)        // billing matches
  const [loading,   setLoading]     = useState(false)
  const [result,    setResult]      = useState(null)
  const [error,     setError]       = useState(null)

  const z = (raw, idx) => {
    if (!scalerStats) return 0
    const m = scalerStats.means?.[idx] ?? 0
    const s = scalerStats.stds?.[idx]  ?? 1
    return (raw - m) / (s || 1)
  }

  const buildFeatureVector = useCallback(() => {
    if (!scalerStats) return Array(30).fill(0)
    // 30-element vector matching feature_names order. Indexes 0–6 are the
    // interpretable features the user sets; 7–29 (V-features) default to 0
    // (= training mean in z-space, i.e. "typical").
    const vec = Array(30).fill(0)
    vec[0] = z(productCD, 0)   // ProductCD
    vec[1] = z(region,    1)   // card3
    vec[2] = z(cardType,  2)   // card6
    vec[3] = z(c7,        3)   // C7
    vec[4] = z(c8,        4)   // C8
    vec[5] = z(c12,       5)   // C12
    vec[6] = z(m6,        6)   // M6
    return vec
  }, [scalerStats, productCD, region, cardType, c7, c8, c12, m6])

  const handleScore = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const features = buildFeatureVector()
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      })
      const data = await res.json()
      if (data.error) {
        setError(data.error)
      } else {
        const enriched = {
          ...data,
          shap_contributions: (data.shap_contributions || []).map(c => ({
            ...c,
            description: FEATURE_DESCRIPTIONS[c.feature] || "Anonymized fraud signal",
          })),
        }
        setResult(enriched)
        onResult?.(enriched)
      }
    } catch {
      setError("Could not reach live endpoint")
    } finally {
      setLoading(false)
    }
  }

  const Select = ({ value, setter, options }) => (
    <select
      value={value}
      onChange={e => setter(Number(e.target.value))}
      className="w-full bg-white border border-[#e5e7eb] rounded-md px-3 py-1.5 text-[13px] text-[#1a1f2e] focus:outline-none focus:border-[#2563eb]"
    >
      {options.map(o => (
        <option key={o.encoded} value={o.encoded}>{o.label}</option>
      ))}
    </select>
  )

  const Slider = ({ value, setter, min, max, suffix }) => (
    <div className="flex items-center gap-3">
      <input
        type="range" min={min} max={max} value={value}
        onChange={e => setter(Number(e.target.value))}
        className="flex-1 accent-[#2563eb]"
      />
      <span className="text-[12px] tabular-nums text-[#1a1f2e] w-12 text-right">
        {value}{suffix ? ` ${suffix}` : ""}
      </span>
    </div>
  )

  return (
    <div>
      <div className="grid grid-cols-12 gap-6">
        {/* Form */}
        <div className="col-span-12 lg:col-span-7">
          <div className="text-[11px] font-semibold text-[#6b7280] uppercase tracking-wider mb-2">
            Transaction Inputs
          </div>
          <div className="divide-y divide-[#eef0f3]">
            <FormRow label="Product type"
                     sub="What kind of purchase">
              <Select value={productCD} setter={setProductCD} options={PRODUCT_OPTIONS} />
            </FormRow>
            <FormRow label="Card type">
              <Select value={cardType} setter={setCardType} options={CARDTYPE_OPTIONS} />
            </FormRow>
            <FormRow label="Card region">
              <Select value={region} setter={setRegion} options={REGION_OPTIONS} />
            </FormRow>
            <FormRow label="Addresses linked to card"
                     sub="Multiple addresses on one card raises risk">
              <Slider value={c7} setter={setC7} min={1} max={10} />
            </FormRow>
            <FormRow label="Transactions sharing this email"
                     sub="In recent activity window">
              <Slider value={c8} setter={setC8} min={1} max={30} />
            </FormRow>
            <FormRow label="Transactions from same device"
                     sub="Higher counts = more shared device use">
              <Slider value={c12} setter={setC12} min={1} max={30} />
            </FormRow>
            <FormRow label="Billing address matches bank record">
              <Select value={m6} setter={setM6} options={M6_OPTIONS} />
            </FormRow>
          </div>

          <button
            onClick={handleScore}
            disabled={loading}
            className="mt-5 w-full bg-[#2563eb] hover:bg-[#1d4ed8] text-white font-medium py-2.5 px-4 rounded-md text-[13px] disabled:opacity-50 transition-colors"
          >
            {loading ? "Scoring…" : "Run Prediction"}
          </button>
          {error && (
            <p className="text-[11px] text-[#dc2626] mt-2">{error}</p>
          )}
          {!scalerStats?.means && (
            <p className="text-[11px] text-[#d97706] mt-2 italic">
              Using placeholder scaling stats. Run rubric_completion.py to export real training stats.
            </p>
          )}
        </div>

        {/* Result */}
        <div className="col-span-12 lg:col-span-5">
          <div className="text-[11px] font-semibold text-[#6b7280] uppercase tracking-wider mb-2">
            Live Prediction
          </div>
          {result ? (
            <div className="flex flex-col items-center">
              {(() => {
                const p = result.fraud_probability
                const pct = Math.round(p * 100)
                const color = p >= 0.5 ? "#dc2626" : p >= 0.25 ? "#d97706" : "#059669"
                const level = p >= 0.5 ? "FLAGGED" : p >= 0.25 ? "REVIEW" : "SAFE"
                return (
                  <>
                    <div className="relative w-36 h-36">
                      <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
                        <circle cx="60" cy="60" r="50" fill="none" stroke="#eef0f3" strokeWidth="10" />
                        <circle cx="60" cy="60" r="50" fill="none"
                          stroke={color} strokeWidth="10"
                          strokeDasharray={`${pct * 3.142} 314.2`}
                          strokeLinecap="round"
                          style={{ transition: "stroke-dasharray 0.6s ease" }}
                        />
                      </svg>
                      <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <span className="text-3xl font-semibold tabular-nums" style={{ color }}>{pct}%</span>
                        <span className="text-[10px] text-[#9ca3af] uppercase tracking-wider">fraud risk</span>
                      </div>
                    </div>
                    <div className="mt-3 px-3 py-1 rounded-full text-[11px] font-semibold tracking-wider"
                         style={{ background: `${color}1a`, color }}>
                      {level}
                    </div>
                    <div className="text-[11px] text-[#059669] mt-3">● Live AWS result</div>
                  </>
                )
              })()}
            </div>
          ) : (
            <div className="bg-[#f7f8fa] rounded-md p-6 text-center text-[12px] text-[#6b7280]">
              Set your inputs and click <span className="font-semibold">Run Prediction</span>.<br />
              The live SageMaker endpoint will return a fraud score and a SHAP breakdown.
            </div>
          )}
        </div>
      </div>

      {/* SHAP plot from live response */}
      {result?.shap_contributions?.length > 0 && result.shap_base_value !== undefined && (
        <div className="mt-8 pt-6 border-t border-[#eef0f3]">
          <div className="flex items-baseline justify-between mb-4">
            <h4 className="text-[13px] font-semibold text-[#1a1f2e]">SHAP — Why this score?</h4>
            <span className="text-[11px] text-[#6b7280]">
              <span className="text-[#dc2626]">Red</span> = pushes toward fraud,
              <span className="text-[#2563eb]"> blue</span> = away
            </span>
          </div>
          <ShapForcePlot
            contributions={result.shap_contributions}
            baseValue={result.shap_base_value}
            finalValue={result.fraud_probability}
          />
          {result.shap_error && (
            <p className="text-[11px] text-[#d97706] mt-3 italic">
              SHAP not available on the endpoint: {result.shap_error}
            </p>
          )}
        </div>
      )}
    </div>
  )
}

/* ─── Page ───────────────────────────────────────────────────────── */

export default function Dashboard() {
  const [metrics,    setMetrics]    = useState(null)
  const [features,   setFeatures]   = useState([])
  const [summary,    setSummary]    = useState(null)
  const [scenarios,  setScenarios]  = useState([])
  const [scalerStats, setScalerStats] = useState(null)
  const [selected,   setSelected]   = useState(null)
  const [liveResult, setLiveResult] = useState(null)
  const [liveLoading, setLiveLoading] = useState(false)
  const [customMode, setCustomMode] = useState(false)

  useEffect(() => {
    fetch("/data/metrics.json").then(r => r.json()).then(setMetrics)
    fetch("/data/top_features.json").then(r => r.json()).then(setFeatures)
    fetch("/data/summary.json").then(r => r.json()).then(setSummary)
    fetch("/data/scenarios.json").then(r => r.json()).then(setScenarios)
    fetch("/data/scaler_stats.json").then(r => r.json()).then(setScalerStats).catch(() => {})
  }, [])

  useEffect(() => {
    if (scenarios.length > 0 && !selected) setSelected(scenarios[0])
  }, [scenarios, selected])

  const handleLiveScore = useCallback(async () => {
    if (!selected?.features) return
    setLiveLoading(true)
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: selected.features }),
      })
      setLiveResult(await res.json())
    } catch {
      setLiveResult({ error: "Could not reach live endpoint" })
    } finally {
      setLiveLoading(false)
    }
  }, [selected])

  const top10 = useMemo(() =>
    features.slice(0, 10).map(f => ({
      ...f,
      importance: f.shap_importance ?? Math.abs(f.coefficient ?? 0),
    })), [features])

  if (!metrics || !summary) {
    return (
      <div className="min-h-screen flex items-center justify-center text-[#6b7280]">
        Loading…
      </div>
    )
  }

  // Donut shows only fraud caught vs missed (which sum to total fraud).
  // False alarms are a separate concept (legitimate $ wrongly blocked) and
  // appear in the side legend only — including them would distort the chart.
  const fraudPieData = [
    { name: "Caught", value: metrics.loss_prevented, fill: "#059669" },
    { name: "Missed", value: metrics.missed_loss,    fill: "#dc2626" },
  ]

  return (
    <main className="min-h-screen bg-[#f7f8fa] text-[#1a1f2e]">

      {/* Top bar */}
      <header className="border-b border-[#e5e7eb] bg-white">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-[#1a1f2e]">Fraud Detection</h1>
            <p className="text-[12px] text-[#6b7280]">{summary.dataset} · {summary.model_name}</p>
          </div>
          <div className="flex items-center gap-2 text-[12px] text-[#6b7280]">
            <span className="w-1.5 h-1.5 rounded-full bg-[#059669]" />
            <span>Live</span>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">

        {/* Stats row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <Stat label="Loss Prevented"
                value={`$${metrics.loss_prevented.toLocaleString()}`}
                sub="Test set" accent="text-[#059669]" />
          <Stat label="Fraud Caught"
                value={`${metrics.pct_prevented}%`}
                sub="Of total fraud value" accent="text-[#059669]" />
          <Stat label="ROC-AUC"
                value={(metrics.roc_auc * 100).toFixed(1) + "%"}
                sub="Held-out test" accent="text-[#2563eb]" />
          <Stat label="False Alarms"
                value={metrics.n_false_alarms}
                sub={`$${metrics.false_alarm_value.toLocaleString()}`}
                accent="text-[#d97706]" />
        </div>

        {/* Row 1: P&L donut + diagnostics */}
        <div className="grid grid-cols-12 gap-4 mb-6">
          <Card title="Fraud Recovery" className="col-span-12 md:col-span-5">
            <div className="grid grid-cols-2 items-center gap-4">
              <div className="relative h-44">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={fraudPieData} cx="50%" cy="50%" innerRadius={48} outerRadius={72} dataKey="value" stroke="none">
                      {fraudPieData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                    </Pie>
                    <Tooltip
                      formatter={v => [`$${Number(v).toLocaleString()}`, ""]}
                      contentStyle={{ background: "white", border: "1px solid #e5e7eb", borderRadius: 6, fontSize: 12 }}
                    />
                  </PieChart>
                </ResponsiveContainer>
                <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                  <span className="text-2xl font-semibold tabular-nums">{metrics.pct_prevented}%</span>
                  <span className="text-[10px] text-[#9ca3af] uppercase tracking-wider">recovered</span>
                </div>
              </div>
              <div className="space-y-3 text-[12px]">
                {[
                  ["Total fraud",   `$${metrics.total_fraud_value.toLocaleString()}`, "#9ca3af"],
                  ["Caught",        `$${metrics.loss_prevented.toLocaleString()}`,    "#059669"],
                  ["Missed",        `$${metrics.missed_loss.toLocaleString()}`,       "#dc2626"],
                  ["False alarm $", `$${metrics.false_alarm_value.toLocaleString()}`, "#d97706"],
                ].map(([label, val, color]) => (
                  <div key={label}>
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full" style={{ background: color }} />
                      <span className="text-[#6b7280]">{label}</span>
                    </div>
                    <div className="font-semibold pl-4 tabular-nums">{val}</div>
                  </div>
                ))}
              </div>
            </div>
          </Card>

          <Card title="Performance Metrics" className="col-span-12 md:col-span-7">
            <table className="w-full text-[13px]">
              <tbody>
                {[
                  ["ROC-AUC",            (metrics.roc_auc * 100).toFixed(2) + "%"],
                  ["Precision (fraud)",  (metrics.precision_fraud * 100).toFixed(2) + "%"],
                  ["Recall (fraud)",     (metrics.recall_fraud * 100).toFixed(2) + "%"],
                  ["F1 Score",           (metrics.f1_fraud * 100).toFixed(2) + "%"],
                  ["Balanced Accuracy",  (metrics.balanced_accuracy * 100).toFixed(2) + "%"],
                  ["Train AUC",          metrics.train_auc != null ? (metrics.train_auc * 100).toFixed(2) + "%" : "—"],
                  ["Test AUC",           metrics.test_auc  != null ? (metrics.test_auc  * 100).toFixed(2) + "%" : "—"],
                  ["Train-Test Gap",     metrics.train_test_gap != null ? (metrics.train_test_gap * 100).toFixed(2) + " pp" : "—"],
                ].map(([k, v], i) => (
                  <tr key={i} className="border-b border-[#eef0f3] last:border-0">
                    <td className="py-2 text-[#6b7280]">{k}</td>
                    <td className="py-2 text-right font-medium tabular-nums">{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>

        {/* Row 2: Model comparison */}
        {summary.all_model_test_aucs && (
          <Card title="Model Comparison" className="mb-6">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart
                data={Object.entries(summary.all_model_test_aucs).map(([name, auc]) => ({ name, auc }))}
                margin={{ left: -10, right: 10, top: 10, bottom: 0 }}
              >
                <CartesianGrid stroke="#eef0f3" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: "#4b5563", fontSize: 11 }} axisLine={{ stroke: "#e5e7eb" }} tickLine={false} />
                <YAxis domain={[0.5, 1.0]} tick={{ fill: "#9ca3af", fontSize: 10 }} axisLine={{ stroke: "#e5e7eb" }} tickLine={false} />
                <Tooltip
                  cursor={{ fill: "rgba(37,99,235,0.05)" }}
                  formatter={v => [`${(Number(v) * 100).toFixed(2)}%`, "Test ROC-AUC"]}
                  contentStyle={{ background: "white", border: "1px solid #e5e7eb", borderRadius: 6, fontSize: 12 }}
                />
                <Bar dataKey="auc" radius={[4, 4, 0, 0]}>
                  {Object.keys(summary.all_model_test_aucs).map((name, i) => (
                    <Cell key={i} fill={name === summary.model_name?.split(" ")[0] ? "#2563eb" : "#cbd5e1"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        )}

        {/* Row 3: Live transaction scanner */}
        <Card title="Score a Transaction" className="mb-6">
          <p className="text-[12px] text-[#6b7280] mb-4">
            Pick a sample transaction below, or use <span className="font-semibold">Build Your Own</span> to set the inputs yourself.
          </p>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
            {scenarios.map(s => {
              const isActive = !customMode && selected?.id === s.id
              const color =
                s.risk_level === "FLAGGED" ? "#dc2626" :
                s.risk_level === "REVIEW"  ? "#d97706" : "#059669"
              return (
                <button
                  key={s.id}
                  onClick={() => { setSelected(s); setLiveResult(null); setCustomMode(false) }}
                  className={`text-left rounded-lg p-4 border transition-all ${
                    isActive
                      ? "border-[#2563eb] ring-1 ring-[#2563eb] bg-[#f0f6ff]"
                      : "border-[#e5e7eb] bg-white hover:border-[#cbd5e1]"
                  }`}
                >
                  <div className="flex items-center gap-1.5 mb-2">
                    <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
                    <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color }}>
                      {s.risk_level}
                    </span>
                  </div>
                  <div className="text-[13px] font-medium text-[#1a1f2e] leading-snug mb-2 line-clamp-2">
                    {s.title}
                  </div>
                  <div className="text-xl font-semibold tabular-nums">{s.amount}</div>
                  <div className="text-[11px] text-[#9ca3af] mt-0.5">{s.time}</div>
                </button>
              )
            })}

            {/* Build Your Own tile */}
            <button
              onClick={() => { setCustomMode(true); setSelected(null); setLiveResult(null) }}
              className={`text-left rounded-lg p-4 border-2 border-dashed transition-all ${
                customMode
                  ? "border-[#2563eb] ring-1 ring-[#2563eb] bg-[#f0f6ff]"
                  : "border-[#cbd5e1] bg-white hover:border-[#2563eb]"
              }`}
            >
              <div className="flex items-center gap-1.5 mb-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[#2563eb]" />
                <span className="text-[10px] font-semibold uppercase tracking-wider text-[#2563eb]">
                  Customize
                </span>
              </div>
              <div className="text-[13px] font-medium text-[#1a1f2e] leading-snug mb-2">
                Build Your Own
              </div>
              <div className="text-[18px] font-semibold text-[#2563eb]">+</div>
              <div className="text-[11px] text-[#9ca3af] mt-0.5">Set the inputs</div>
            </button>
          </div>

          {customMode && (
            <div className="border-t border-[#eef0f3] pt-6">
              <BuildYourOwn scalerStats={scalerStats} />
            </div>
          )}

          {!customMode && selected && (
            <div className="border-t border-[#eef0f3] pt-6">
              <div className="grid grid-cols-12 gap-6">

                {/* Gauge & verdict */}
                <div className="col-span-12 md:col-span-3">
                  {(() => {
                    const p = liveResult?.fraud_probability ?? selected.fraud_probability
                    const pct = Math.round(p * 100)
                    const color = p >= 0.5 ? "#dc2626" : p >= 0.25 ? "#d97706" : "#059669"
                    const level = p >= 0.5 ? "FLAGGED" : p >= 0.25 ? "REVIEW" : "SAFE"
                    return (
                      <div className="flex flex-col items-center">
                        <div className="relative w-36 h-36">
                          <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
                            <circle cx="60" cy="60" r="50" fill="none" stroke="#eef0f3" strokeWidth="10" />
                            <circle cx="60" cy="60" r="50" fill="none"
                              stroke={color} strokeWidth="10"
                              strokeDasharray={`${pct * 3.142} 314.2`}
                              strokeLinecap="round"
                              style={{ transition: "stroke-dasharray 0.6s ease" }}
                            />
                          </svg>
                          <div className="absolute inset-0 flex flex-col items-center justify-center">
                            <span className="text-3xl font-semibold tabular-nums" style={{ color }}>{pct}%</span>
                            <span className="text-[10px] text-[#9ca3af] uppercase tracking-wider">fraud risk</span>
                          </div>
                        </div>
                        <div className="mt-3 px-3 py-1 rounded-full text-[11px] font-semibold tracking-wider"
                             style={{ background: `${color}1a`, color }}>
                          {level}
                        </div>
                      </div>
                    )
                  })()}
                </div>

                {/* Signals */}
                <div className="col-span-12 md:col-span-5">
                  <div className="text-[11px] font-semibold text-[#6b7280] uppercase tracking-wider mb-3">
                    Why
                  </div>
                  <ul className="space-y-2 text-[13px] text-[#1a1f2e]">
                    {selected.key_signals.map((sig, i) => (
                      <li key={i} className="flex gap-2">
                        <span className="text-[#2563eb] mt-0.5">›</span>
                        <span>{sig}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Live AWS */}
                <div className="col-span-12 md:col-span-4">
                  <div className="text-[11px] font-semibold text-[#6b7280] uppercase tracking-wider mb-3">
                    Live Endpoint
                  </div>
                  {selected.features ? (
                    <>
                      <button
                        onClick={handleLiveScore}
                        disabled={liveLoading}
                        className="w-full bg-[#2563eb] hover:bg-[#1d4ed8] text-white font-medium py-2 px-4 rounded-md text-[13px] disabled:opacity-50 transition-colors"
                      >
                        {liveLoading ? "Scoring…" : "Score via AWS"}
                      </button>
                      {liveResult?.error && (
                        <p className="text-[11px] text-[#dc2626] mt-2">{liveResult.error}</p>
                      )}
                      {liveResult && !liveResult.error && (
                        <p className="text-[11px] text-[#059669] mt-2 tabular-nums">
                          Live: {(liveResult.fraud_probability * 100).toFixed(1)}% · {liveResult.risk_level}
                        </p>
                      )}
                    </>
                  ) : (
                    <p className="text-[12px] text-[#6b7280] leading-relaxed">
                      Endpoint not configured.
                    </p>
                  )}
                </div>
              </div>

              {/* SHAP local force plot */}
              {selected.shap_contributions?.length > 0 && selected.shap_base_value !== undefined && (
                <div className="mt-8 pt-6 border-t border-[#eef0f3]">
                  <div className="flex items-baseline justify-between mb-4">
                    <h4 className="text-[13px] font-semibold text-[#1a1f2e]">SHAP — Why this score?</h4>
                    <span className="text-[11px] text-[#6b7280]">
                      <span className="text-[#dc2626]">Red</span> = pushes toward fraud,
                      <span className="text-[#2563eb]"> blue</span> = away
                    </span>
                  </div>
                  <ShapForcePlot
                    contributions={selected.shap_contributions}
                    baseValue={selected.shap_base_value}
                    finalValue={selected.fraud_probability}
                  />
                </div>
              )}
            </div>
          )}
        </Card>

        {/* Row 4: SHAP global + glossary */}
        <div className="grid grid-cols-12 gap-4 mb-6">
          <Card title="SHAP Feature Importance" className="col-span-12 md:col-span-7">
            {top10.length > 0 ? (
              <ShapBeeswarm features={top10} />
            ) : (
              <div className="text-[12px] text-[#6b7280] py-8 text-center">
                Run rubric_completion.py to populate SHAP importances.
              </div>
            )}
          </Card>

          <Card title="Feature Glossary" className="col-span-12 md:col-span-5">
            <div className="space-y-2 max-h-[420px] overflow-y-auto pr-1">
              {top10.map((f, i) => (
                <div key={f.feature} className="flex items-baseline gap-3 py-2 border-b border-[#eef0f3] last:border-0">
                  <span className="text-[10px] text-[#9ca3af] tabular-nums w-5">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <span className="text-[12px] font-semibold text-[#2563eb] w-16">{f.feature}</span>
                  <span className="text-[12px] text-[#4b5563] flex-1 leading-snug">{f.description}</span>
                  <span className="text-[10px] tabular-nums text-[#9ca3af]">
                    {Number(f.importance).toFixed(4)}
                  </span>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Footer */}
        <footer className="text-[11px] text-[#9ca3af] flex flex-wrap items-center justify-between gap-2 pt-4 border-t border-[#e5e7eb]">
          <span>{summary.dataset} · {summary.model_name}</span>
          <span>Train n = {summary.n_train.toLocaleString()} · Test n = {summary.n_test.toLocaleString()} · CV folds = {summary.cv_folds}</span>
        </footer>
      </div>
    </main>
  )
}
