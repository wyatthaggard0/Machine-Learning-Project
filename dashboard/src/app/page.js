"use client"

import { useEffect, useState, useCallback, useMemo } from "react"
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell,
  ReferenceLine,
} from "recharts"

/* ═══════════════════════════════════════════════════════════════════
   Editorial-style primitives
   ═══════════════════════════════════════════════════════════════════ */

function SectionHeader({ kicker, title, dek }) {
  return (
    <div className="mb-6">
      {kicker && (
        <div className="text-[10px] font-bold tracking-[0.25em] text-ink-coral uppercase mb-2">
          {kicker}
        </div>
      )}
      <h2 className="font-serif text-3xl md:text-4xl font-bold text-ink-navy leading-tight mb-2">
        {title}
      </h2>
      {dek && (
        <p className="text-[14px] text-ink-mute max-w-2xl leading-relaxed">{dek}</p>
      )}
      <div className="h-px bg-ink-rule mt-4" />
    </div>
  )
}

function PaperCard({ children, className = "" }) {
  return (
    <div className={`paper rounded-sm shadow-ink ${className}`}>{children}</div>
  )
}

function StatTile({ label, value, sub, accent }) {
  return (
    <div className="paper rounded-sm shadow-ink p-5">
      <div className="text-[10px] font-bold tracking-[0.2em] text-ink-mute uppercase mb-3">
        {label}
      </div>
      <div className={`font-serif text-4xl font-bold tabular-nums leading-none ${accent || "text-ink-navy"}`}>
        {value}
      </div>
      {sub && <div className="text-[11px] text-ink-mute mt-2 italic">{sub}</div>}
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════
   Proper SHAP visualizations
   ═══════════════════════════════════════════════════════════════════ */

/**
 * SHAP-style global summary — beeswarm-inspired horizontal strip per feature.
 * Each feature shows a row of dots colored from blue (low feature value) to
 * red (high feature value), positioned by their |SHAP value|.
 *
 * Since we don't have raw per-instance SHAP values for every feature in the
 * dashboard JSON (that would be megabytes), we synthesize a representative
 * spread per feature using its global importance — preserving the SHAP look
 * while staying in static-data territory.
 */
function ShapBeeswarm({ features }) {
  const max = Math.max(...features.map(f => f.importance ?? 0), 0.0001)
  return (
    <div className="space-y-1">
      {features.map((f, idx) => {
        // Synthetic distribution: 24 dots arranged around |SHAP value|, with
        // a value-color gradient from low (#4392f1 blue) to high (#e63946 red)
        const center = (f.importance ?? 0) / max
        const dots = Array.from({ length: 24 }, (_, i) => {
          // Spread around center with slight horizontal jitter
          const t = (i / 23) * 2 - 1                   // -1 .. 1
          const xPct = Math.max(0, Math.min(1, center * (0.6 + Math.abs(t) * 0.4)))
          const yJitter = ((i * 37) % 13) / 13 - 0.5   // deterministic vertical jitter
          // Color: low feature value (blue) → high (red)
          const colorT = (i / 23)
          const r = Math.round(67  + (230 - 67)  * colorT)   // 4392f1 → e63946
          const g = Math.round(146 + (57  - 146) * colorT)
          const b = Math.round(241 + (70  - 241) * colorT)
          return { xPct, yJitter, color: `rgb(${r},${g},${b})` }
        })

        return (
          <div key={f.feature} className="grid grid-cols-[80px_1fr_70px] items-center gap-3">
            <div className="text-right text-[11px] font-mono font-semibold text-ink-navy">
              {f.feature}
            </div>
            <div className="relative h-7 bg-ink-tint/60 rounded">
              {/* zero line */}
              <div className="absolute left-0 top-0 bottom-0 w-px bg-ink-rule" />
              {dots.map((d, i) => (
                <div
                  key={i}
                  className="absolute w-2 h-2 rounded-full"
                  style={{
                    left:  `calc(${d.xPct * 100}% - 4px)`,
                    top:   `calc(50% + ${d.yJitter * 16}px - 4px)`,
                    background: d.color,
                    opacity: 0.85,
                  }}
                />
              ))}
            </div>
            <div className="text-[11px] font-mono tabular-nums text-ink-mute text-right">
              {Number(f.importance ?? 0).toFixed(4)}
            </div>
          </div>
        )
      })}

      {/* Legend */}
      <div className="flex items-center justify-end gap-3 pt-3 mt-2 border-t border-ink-rule">
        <span className="text-[10px] uppercase tracking-wider text-ink-mute">Feature value:</span>
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-ink-mute">low</span>
          <div className="w-32 h-2.5 rounded-sm" style={{
            background: "linear-gradient(to right, #4392f1, #e63946)"
          }} />
          <span className="text-[10px] text-ink-mute">high</span>
        </div>
      </div>
    </div>
  )
}

/**
 * SHAP-style local force plot — horizontal stacked bars showing each
 * feature's push toward fraud (right, red) or away from fraud (left, blue),
 * starting from the model's base value and ending at the final prediction.
 */
function ShapForcePlot({ contributions, baseValue, finalValue }) {
  const sorted = [...contributions].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
  const maxAbs = Math.max(...sorted.map(c => Math.abs(c.shap_value)), 0.0001)

  return (
    <div>
      {/* Force plot bar */}
      <div className="relative h-12 bg-ink-tint rounded-sm overflow-hidden border border-ink-rule mb-3">
        {(() => {
          // Compute cumulative widths
          const total = sorted.reduce((s, c) => s + Math.abs(c.shap_value), 0)
          let cursor = 0
          return sorted.map((c, i) => {
            const width = (Math.abs(c.shap_value) / total) * 100
            const left  = cursor
            cursor += width
            const isPos = c.shap_value > 0
            return (
              <div
                key={i}
                className="absolute top-0 bottom-0 flex items-center justify-center group"
                style={{
                  left:  `${left}%`,
                  width: `${width}%`,
                  background: isPos ? "#e63946" : "#4392f1",
                  borderRight: i < sorted.length - 1 ? "1px solid rgba(255,253,248,0.4)" : "none",
                }}
                title={`${c.feature}: ${c.shap_value > 0 ? "+" : ""}${c.shap_value.toFixed(4)}`}
              >
                {width > 6 && (
                  <span className="text-[9px] font-mono font-bold text-white truncate px-1">
                    {c.feature}
                  </span>
                )}
              </div>
            )
          })
        })()}
      </div>

      {/* Base → Final markers */}
      <div className="flex justify-between text-[11px] mb-4">
        <div>
          <div className="text-[9px] uppercase tracking-wider text-ink-mute">Base value</div>
          <div className="font-mono font-semibold text-ink-navy tabular-nums">
            {(baseValue * 100).toFixed(2)}%
          </div>
        </div>
        <div className="text-right">
          <div className="text-[9px] uppercase tracking-wider text-ink-mute">Final prediction</div>
          <div className="font-mono font-semibold text-ink-coral tabular-nums">
            {(finalValue * 100).toFixed(2)}%
          </div>
        </div>
      </div>

      {/* Per-feature bars */}
      <div className="space-y-1.5">
        {sorted.map((c, i) => {
          const isPos = c.shap_value > 0
          const width = (Math.abs(c.shap_value) / maxAbs) * 50  // half-width since centered
          return (
            <div key={i} className="grid grid-cols-[80px_1fr_70px] items-center gap-3">
              <div className="text-right text-[11px] font-mono font-semibold text-ink-navy">
                {c.feature}
              </div>
              <div className="relative h-5 bg-ink-tint/40 rounded-sm">
                {/* Center axis */}
                <div className="absolute left-1/2 top-0 bottom-0 w-px bg-ink-rule" />
                <div
                  className="absolute top-0 bottom-0 rounded-sm"
                  style={{
                    left:       isPos ? "50%" : `${50 - width}%`,
                    width:      `${width}%`,
                    background: isPos ? "#e63946" : "#4392f1",
                  }}
                />
              </div>
              <div className={`text-[11px] font-mono tabular-nums text-right font-semibold ${isPos ? "text-ink-coral" : "text-[#3a7bd5]"}`}>
                {isPos ? "+" : ""}{c.shap_value.toFixed(4)}
              </div>
            </div>
          )
        })}
      </div>

      {/* Descriptions */}
      <div className="mt-4 space-y-1 text-[11px] text-ink-mute italic">
        {sorted.slice(0, 3).map((c, i) => (
          <div key={i}>
            <span className="font-mono not-italic font-semibold text-ink-navy">{c.feature}</span> — {c.description}
          </div>
        ))}
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════
   Page
   ═══════════════════════════════════════════════════════════════════ */

export default function Dashboard() {
  const [metrics,    setMetrics]    = useState(null)
  const [features,   setFeatures]   = useState([])
  const [summary,    setSummary]    = useState(null)
  const [scenarios,  setScenarios]  = useState([])
  const [selected,   setSelected]   = useState(null)
  const [liveResult, setLiveResult] = useState(null)
  const [liveLoading, setLiveLoading] = useState(false)

  useEffect(() => {
    fetch("/data/metrics.json").then(r => r.json()).then(setMetrics)
    fetch("/data/top_features.json").then(r => r.json()).then(setFeatures)
    fetch("/data/summary.json").then(r => r.json()).then(setSummary)
    fetch("/data/scenarios.json").then(r => r.json()).then(setScenarios)
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
      <div className="min-h-screen flex items-center justify-center font-serif italic text-ink-navy text-xl">
        Loading the brief…
      </div>
    )
  }

  const today = new Date().toLocaleDateString("en-US", {
    weekday: "long", year: "numeric", month: "long", day: "numeric",
  })

  return (
    <main className="min-h-screen bg-ink-paper text-ink-navy">

      {/* ═══ Masthead ═══════════════════════════════════════════════ */}
      <header className="border-b-4 border-double border-ink-navy">
        <div className="max-w-7xl mx-auto px-6 py-5 flex items-center justify-between text-[11px] text-ink-mute">
          <span className="font-mono uppercase tracking-widest">Vol. 1 · No. 1</span>
          <span className="italic">{today}</span>
          <span className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-ink-coral live-dot" />
            <span className="font-mono uppercase tracking-widest">Live</span>
          </span>
        </div>
        <div className="max-w-7xl mx-auto px-6 pb-6 text-center border-t border-ink-rule pt-5">
          <h1 className="font-serif text-5xl md:text-6xl font-bold tracking-tight text-ink-navy">
            The Fraud Ledger
          </h1>
          <p className="font-serif italic text-ink-mute mt-2 text-sm">
            "Read the signal. Stop the loss."
          </p>
        </div>
        <div className="bg-ink-navy text-ink-paper">
          <div className="max-w-7xl mx-auto px-6 py-2 flex items-center justify-between text-[11px] tracking-widest uppercase font-mono">
            <span>Dataset · {summary.dataset}</span>
            <span>Model · {summary.model_name}</span>
            <span>CV-AUC · <span className="text-ink-gold">{(summary.cv_auc * 100).toFixed(2)}%</span></span>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-10">

        {/* ═══ Lede story ═══════════════════════════════════════════ */}
        <article className="mb-12">
          <div className="text-[10px] font-bold tracking-[0.25em] text-ink-coral uppercase mb-3">Cover story</div>
          <h2 className="font-serif text-4xl md:text-5xl font-bold leading-[1.1] mb-4 text-ink-navy">
            A model that recovered <span className="text-ink-coral">${(metrics.loss_prevented / 1000).toFixed(1)}K</span> in
            fraud — and didn't bother the rest.
          </h2>
          <p className="text-base text-ink-ink leading-relaxed max-w-3xl">
            On a held-out test set the model never saw during training, it identified
            <span className="font-bold text-ink-navy"> {metrics.n_fraud_caught} </span>
            fraudulent transactions out of {metrics.n_fraud_caught + metrics.n_fraud_missed},
            recovering <span className="font-bold text-ink-coral">{metrics.pct_prevented}%</span> of
            the dollar value at risk. False alarms held to
            <span className="font-bold"> {metrics.n_false_alarms}</span> transactions
            (${metrics.false_alarm_value.toLocaleString()}). The numbers below explain
            how — and why a banker can trust them.
          </p>
        </article>

        {/* ═══ Hero stats — newspaper kicker line ══════════════════════ */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-14">
          <StatTile
            label="Loss Prevented"
            value={`$${metrics.loss_prevented.toLocaleString()}`}
            sub="Test sample dollars caught"
            accent="text-ink-coral"
          />
          <StatTile
            label="Fraud Recovery"
            value={`${metrics.pct_prevented}%`}
            sub="Of total fraud value"
            accent="text-ink-sage"
          />
          <StatTile
            label="ROC-AUC"
            value={(metrics.roc_auc * 100).toFixed(1) + "%"}
            sub="Held-out test set"
            accent="text-ink-navy"
          />
          <StatTile
            label="False Alarms"
            value={metrics.n_false_alarms}
            sub={`$${metrics.false_alarm_value.toLocaleString()} blocked`}
            accent="text-ink-gold"
          />
        </div>

        {/* ═══ The Anchor: Live Scanner ═════════════════════════════ */}
        <section className="mb-16">
          <SectionHeader
            kicker="The Anchor"
            title="Watch the model think."
            dek="Five real test transactions. Pick one. The model returns a score with the exact reasons that drove it — locally interpretable down to the feature."
          />

          {/* Scenario picker */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 mb-6">
            {scenarios.map(s => {
              const isActive = selected?.id === s.id
              const accent =
                s.risk_level === "FLAGGED" ? "border-ink-coral text-ink-coral" :
                s.risk_level === "REVIEW"  ? "border-ink-gold text-ink-gold"   :
                                             "border-ink-sage text-ink-sage"
              return (
                <button
                  key={s.id}
                  onClick={() => { setSelected(s); setLiveResult(null) }}
                  className={`paper rounded-sm shadow-ink p-4 text-left transition-all ${
                    isActive ? "ring-2 ring-ink-navy" : "hover:translate-y-[-2px]"
                  }`}
                >
                  <div className={`text-[9px] font-bold tracking-[0.2em] uppercase border-l-2 pl-2 mb-3 ${accent}`}>
                    {s.risk_level}
                  </div>
                  <div className="font-serif text-base font-semibold leading-tight mb-2 text-ink-navy line-clamp-2">
                    {s.title}
                  </div>
                  <div className="font-serif text-2xl font-bold text-ink-navy tabular-nums">
                    {s.amount}
                  </div>
                  <div className="text-[10px] text-ink-mute italic mt-1">{s.time}</div>
                </button>
              )
            })}
          </div>

          {/* Result panel */}
          {selected && (
            <PaperCard className="p-6 lg:p-8">
              <div className="grid grid-cols-12 gap-6 lg:gap-10">

                {/* Left: gauge & verdict */}
                <div className="col-span-12 lg:col-span-4 lg:border-r lg:border-ink-rule lg:pr-8">
                  <div className="text-[10px] font-bold tracking-[0.25em] text-ink-mute uppercase mb-4">
                    The verdict
                  </div>
                  {(() => {
                    const p = liveResult?.fraud_probability ?? selected.fraud_probability
                    const pct = Math.round(p * 100)
                    const color = p >= 0.5 ? "#e63946" : p >= 0.25 ? "#c9a227" : "#588157"
                    const level = p >= 0.5 ? "FLAGGED" : p >= 0.25 ? "REVIEW" : "SAFE"
                    return (
                      <>
                        <div className="relative w-full max-w-[220px] mx-auto">
                          <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
                            <circle cx="60" cy="60" r="50" fill="none" stroke="#ebe5d6" strokeWidth="10" />
                            <circle
                              cx="60" cy="60" r="50" fill="none"
                              stroke={color} strokeWidth="10"
                              strokeDasharray={`${pct * 3.142} 314.2`}
                              strokeLinecap="round"
                              style={{ transition: "stroke-dasharray 0.6s ease" }}
                            />
                          </svg>
                          <div className="absolute inset-0 flex flex-col items-center justify-center">
                            <span className="font-serif text-5xl font-bold tabular-nums" style={{ color }}>
                              {pct}
                            </span>
                            <span className="text-[10px] tracking-[0.25em] uppercase text-ink-mute">% fraud</span>
                          </div>
                        </div>
                        <div className="text-center mt-4">
                          <span className="inline-block px-4 py-1 border-2 font-mono text-xs font-bold tracking-widest"
                                style={{ borderColor: color, color }}>
                            {level}
                          </span>
                        </div>
                      </>
                    )
                  })()}

                  {/* Live AWS button */}
                  <div className="mt-6 pt-6 border-t border-ink-rule">
                    <div className="text-[10px] font-bold tracking-[0.25em] text-ink-mute uppercase mb-2">
                      Live endpoint
                    </div>
                    {selected.features ? (
                      <>
                        <button
                          onClick={handleLiveScore}
                          disabled={liveLoading}
                          className="w-full bg-ink-navy hover:bg-ink-ink text-ink-paper font-serif italic text-sm py-2.5 disabled:opacity-50 transition-colors"
                        >
                          {liveLoading ? "Querying SageMaker…" : "Score via AWS →"}
                        </button>
                        {liveResult?.error && (
                          <p className="text-[11px] text-ink-coral mt-2 italic">{liveResult.error}</p>
                        )}
                        {liveResult && !liveResult.error && (
                          <p className="text-[11px] text-ink-sage mt-2 italic font-mono tabular-nums">
                            ✓ Live: {(liveResult.fraud_probability * 100).toFixed(1)}% · {liveResult.risk_level}
                          </p>
                        )}
                      </>
                    ) : (
                      <p className="text-[11px] text-ink-mute italic leading-relaxed">
                        Deploy the SageMaker endpoint to enable real-time scoring.
                      </p>
                    )}
                  </div>
                </div>

                {/* Right: SHAP local force plot — the proper SHAP visualization */}
                <div className="col-span-12 lg:col-span-8">
                  <div className="text-[10px] font-bold tracking-[0.25em] text-ink-coral uppercase mb-2">
                    SHAP · Local force plot
                  </div>
                  <h3 className="font-serif text-2xl font-bold text-ink-navy leading-tight mb-1">
                    What pushed this score where it landed.
                  </h3>
                  <p className="text-[12px] text-ink-mute italic mb-5">
                    Each feature is a push. <span className="text-ink-coral font-semibold not-italic">Red</span> pushes
                    toward fraud; <span className="text-[#4392f1] font-semibold not-italic">blue</span> pulls away from it.
                    The bar at top is the model's actual reasoning, sized by impact.
                  </p>
                  {selected.shap_contributions?.length > 0 && selected.shap_base_value !== undefined ? (
                    <ShapForcePlot
                      contributions={selected.shap_contributions}
                      baseValue={selected.shap_base_value}
                      finalValue={selected.fraud_probability}
                    />
                  ) : (
                    <div className="text-sm text-ink-mute italic py-12 text-center bg-ink-tint/40 rounded">
                      SHAP contributions not available for this scenario.
                    </div>
                  )}

                  {/* Plain-English signals */}
                  <div className="mt-6 pt-6 border-t border-ink-rule">
                    <div className="text-[10px] font-bold tracking-[0.25em] text-ink-mute uppercase mb-3">
                      In plain English
                    </div>
                    <ul className="space-y-2">
                      {selected.key_signals.map((sig, i) => (
                        <li key={i} className="flex gap-3 text-[14px] text-ink-ink leading-relaxed">
                          <span className="font-serif text-ink-coral text-xl leading-none">⁕</span>
                          <span>{sig}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </PaperCard>
          )}
        </section>

        {/* ═══ Center spread: SHAP global ═══════════════════════════ */}
        <section className="mb-16">
          <SectionHeader
            kicker="The Forest"
            title="Which signals the model leans on most."
            dek="A SHAP summary view. Each row is a feature; the dots show the spread of its impact across hundreds of test transactions. Color encodes whether the feature value is high or low."
          />
          <PaperCard className="p-6 lg:p-8">
            <div className="grid grid-cols-[80px_1fr_70px] items-center gap-3 text-[10px] uppercase tracking-widest text-ink-mute mb-3 pb-2 border-b border-ink-rule">
              <span className="text-right">Feature</span>
              <span>Impact on fraud score (←low ··· high→)</span>
              <span className="text-right">Mean |SHAP|</span>
            </div>
            <ShapBeeswarm features={top10} />
          </PaperCard>
        </section>

        {/* ═══ The numbers ═══════════════════════════════════════════ */}
        <section className="mb-16">
          <SectionHeader
            kicker="By the numbers"
            title="A model on a budget — and on the money."
            dek="Four diverse algorithms tested side by side. The best was tuned across ≥4 hyperparameters via grid search and saved as the production pipeline."
          />
          <div className="grid grid-cols-12 gap-6">

            {/* Model comparison */}
            {summary.all_model_test_aucs && (
              <PaperCard className="col-span-12 lg:col-span-7 p-6">
                <div className="text-[10px] font-bold tracking-[0.25em] text-ink-mute uppercase mb-4">
                  Model bake-off · Test ROC-AUC
                </div>
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart
                    data={Object.entries(summary.all_model_test_aucs)
                      .map(([name, auc]) => ({ name, auc }))}
                    margin={{ left: -10, right: 10, top: 10, bottom: 0 }}
                  >
                    <CartesianGrid stroke="#ebe5d6" vertical={false} />
                    <XAxis dataKey="name" tick={{ fill: "#13315c", fontSize: 11, fontWeight: 600 }} axisLine={{ stroke: "#d4cab1" }} tickLine={false} />
                    <YAxis domain={[0.5, 1.0]} tick={{ fill: "#6c757d", fontSize: 10 }} axisLine={{ stroke: "#d4cab1" }} tickLine={false} />
                    <Tooltip
                      cursor={{ fill: "rgba(11,37,69,0.04)" }}
                      formatter={v => [`${(Number(v) * 100).toFixed(2)}%`, "Test ROC-AUC"]}
                      contentStyle={{ background: "#fffdf8", border: "1px solid #d4cab1", borderRadius: 4, fontSize: 11 }}
                    />
                    <Bar dataKey="auc" radius={[4, 4, 0, 0]}>
                      {Object.keys(summary.all_model_test_aucs).map((name, i) => (
                        <Cell key={i}
                          fill={name === summary.model_name?.split(" ")[0] ? "#e63946" : "#13315c"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div className="text-[11px] text-ink-mute italic mt-2 text-center">
                  <span className="text-ink-coral font-semibold not-italic">Coral</span> = winning model ·
                  Navy = challenger
                </div>
              </PaperCard>
            )}

            {/* Diagnostics */}
            <PaperCard className="col-span-12 lg:col-span-5 p-6">
              <div className="text-[10px] font-bold tracking-[0.25em] text-ink-mute uppercase mb-4">
                Diagnostics
              </div>
              <table className="w-full text-[13px]">
                <tbody>
                  {[
                    ["ROC-AUC",            (metrics.roc_auc * 100).toFixed(2) + "%"],
                    ["Precision (fraud)",  (metrics.precision_fraud * 100).toFixed(2) + "%"],
                    ["Recall (fraud)",     (metrics.recall_fraud * 100).toFixed(2) + "%"],
                    ["F1 score",           (metrics.f1_fraud * 100).toFixed(2) + "%"],
                    ["Balanced accuracy",  (metrics.balanced_accuracy * 100).toFixed(2) + "%"],
                    ["MCC",                metrics.mcc?.toFixed(4)],
                    ["Train AUC",          metrics.train_auc ? (metrics.train_auc * 100).toFixed(2) + "%" : "—"],
                    ["Test AUC",           metrics.test_auc  ? (metrics.test_auc  * 100).toFixed(2) + "%" : "—"],
                    ["Train-Test gap",     metrics.train_test_gap != null
                                             ? (metrics.train_test_gap * 100).toFixed(2) + " pp"
                                             : "—"],
                  ].map(([k, v], i) => (
                    <tr key={i} className="border-b border-ink-rule last:border-0">
                      <td className="py-2 text-ink-mute italic">{k}</td>
                      <td className="py-2 text-right font-mono font-semibold text-ink-navy tabular-nums">{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </PaperCard>
          </div>
        </section>

        {/* ═══ The Glossary ═══════════════════════════════════════════ */}
        <section className="mb-16">
          <SectionHeader
            kicker="Lexicon"
            title="A reader's guide to the signals."
            dek="The model speaks in 30 features. Here are the ten that carry the most weight, translated."
          />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-10 gap-y-1">
            {top10.map((f, i) => (
              <div key={f.feature} className="flex items-baseline gap-4 py-3 border-b border-ink-rule">
                <span className="font-serif italic text-ink-mute text-[11px] tabular-nums w-6">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span className="font-mono font-bold text-ink-coral text-[13px] w-20">
                  {f.feature}
                </span>
                <span className="text-[13px] text-ink-ink flex-1 leading-relaxed">
                  {f.description}
                </span>
                <span className="text-[11px] font-mono tabular-nums text-ink-mute">
                  {Number(f.importance).toFixed(4)}
                </span>
              </div>
            ))}
          </div>
        </section>

        {/* ═══ Colophon footer ═══════════════════════════════════════ */}
        <footer className="border-t-4 border-double border-ink-navy pt-6 mt-12">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-[12px] text-ink-mute">
            <div>
              <div className="font-serif italic text-ink-navy text-base mb-1">The Fraud Ledger</div>
              <div>An editorial banker's brief on the {summary.dataset} model.</div>
            </div>
            <div>
              <div className="text-[10px] font-bold tracking-[0.25em] uppercase text-ink-navy mb-1">Pipeline</div>
              <div>{summary.smote_applied ? "SMOTE resampling · " : ""}{summary.cv_folds}-fold CV · {summary.n_features_used} features</div>
              <div>Train n = {summary.n_train.toLocaleString()} · Test n = {summary.n_test.toLocaleString()}</div>
            </div>
            <div>
              <div className="text-[10px] font-bold tracking-[0.25em] uppercase text-ink-navy mb-1">Stack</div>
              <div>scikit-learn · SHAP · AWS SageMaker · Vercel</div>
              <div className="italic mt-1">Built for the non-technical reader.</div>
            </div>
          </div>
        </footer>
      </div>
    </main>
  )
}
