"use client"

import { useEffect, useState, useCallback } from "react"
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell, LineChart, Line, CartesianGrid,
} from "recharts"

/* ──────────────────────────────────────────────────────────────────
   Bloomberg-style primitives
   ────────────────────────────────────────────────────────────────── */

function Panel({ title, code, children, className = "" }) {
  return (
    <div className={`border border-[#1f1f1f] bg-[#111111] ${className}`}>
      {(title || code) && (
        <div className="flex items-center justify-between border-b border-[#1f1f1f] bg-[#0a0a0a] px-3 py-1.5">
          <span className="text-[11px] font-bold tracking-widest text-[#ffaa00] uppercase">
            {title}
          </span>
          {code && (
            <span className="text-[10px] text-[#555555] font-mono">{code}</span>
          )}
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  )
}

function Stat({ label, value, sub, color = "text-[#ffaa00]", trend }) {
  return (
    <div className="border border-[#1f1f1f] bg-[#111111] px-4 py-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-widest text-[#888888]">{label}</span>
        {trend && (
          <span className={`text-[10px] font-bold ${trend === "up" ? "text-[#00d97e]" : "text-[#ff3344]"}`}>
            {trend === "up" ? "▲" : "▼"}
          </span>
        )}
      </div>
      <div className={`text-2xl font-bold ${color} mt-1 tabular-nums`}>{value}</div>
      {sub && <div className="text-[10px] text-[#555555] mt-0.5 uppercase tracking-wide">{sub}</div>}
    </div>
  )
}

function FraudGauge({ value }) {
  const pct = Math.round(value * 100)
  const level = value >= 0.5 ? "FLAGGED" : value >= 0.25 ? "REVIEW" : "SAFE"
  const color = value >= 0.5 ? "#ff3344" : value >= 0.25 ? "#ffaa00" : "#00d97e"

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-32 h-32">
        <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
          <circle cx="60" cy="60" r="48" fill="none" stroke="#1f1f1f" strokeWidth="8" />
          <circle
            cx="60" cy="60" r="48" fill="none"
            stroke={color}
            strokeWidth="8"
            strokeDasharray={`${pct * 3.016} 301.6`}
            strokeLinecap="butt"
            style={{ transition: "stroke-dasharray 0.6s ease" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl font-bold tabular-nums" style={{ color }}>{pct}</span>
          <span className="text-[9px] tracking-widest text-[#555555] uppercase">RISK SCORE</span>
        </div>
      </div>
      <div className="mt-2 px-3 py-1 border" style={{ borderColor: color, color }}>
        <span className="text-[11px] font-bold tracking-widest">{level}</span>
      </div>
    </div>
  )
}

/* ──────────────────────────────────────────────────────────────────
   Main page
   ────────────────────────────────────────────────────────────────── */

export default function Dashboard() {
  const [metrics,   setMetrics]   = useState(null)
  const [features,  setFeatures]  = useState([])
  const [summary,   setSummary]   = useState(null)
  const [scenarios, setScenarios] = useState([])
  const [selected,  setSelected]  = useState(null)
  const [liveResult, setLiveResult] = useState(null)
  const [liveLoading, setLiveLoading] = useState(false)
  const [time, setTime] = useState("")

  useEffect(() => {
    fetch("/data/metrics.json").then(r => r.json()).then(setMetrics)
    fetch("/data/top_features.json").then(r => r.json()).then(setFeatures)
    fetch("/data/summary.json").then(r => r.json()).then(setSummary)
    fetch("/data/scenarios.json").then(r => r.json()).then(setScenarios)
  }, [])

  // Terminal clock
  useEffect(() => {
    const tick = () => {
      const d = new Date()
      const hh = String(d.getUTCHours()).padStart(2, "0")
      const mm = String(d.getUTCMinutes()).padStart(2, "0")
      const ss = String(d.getUTCSeconds()).padStart(2, "0")
      setTime(`${hh}:${mm}:${ss} UTC`)
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  const handleSelectScenario = useCallback((s) => {
    setSelected(s)
    setLiveResult(null)
  }, [])

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

  if (!metrics || !summary) {
    return (
      <div className="min-h-screen flex items-center justify-center text-[#ffaa00] font-mono">
        <span className="caret">LOADING</span>
      </div>
    )
  }

  const top10 = features.slice(0, 10).map(f => ({
    ...f,
    importance: f.shap_importance ?? Math.abs(f.coefficient ?? 0),
  }))

  return (
    <main className="min-h-screen bg-[#0a0a0a] text-[#e8e8e8] relative" style={{ zIndex: 2 }}>

      {/* ── Top status bar ───────────────────────────────────────── */}
      <div className="border-b border-[#1f1f1f] bg-[#000000] flex items-center text-[11px] tabular-nums">
        <div className="bg-[#ffaa00] text-black font-bold px-3 py-1 tracking-widest">FRAUD DETECTION TERMINAL</div>
        <div className="px-3 text-[#888888]">IEEE-CIS / VESTA</div>
        <div className="px-3 text-[#555555]">|</div>
        <div className="px-3 text-[#888888]">MODEL: <span className="text-[#ffaa00]">{summary.model_name}</span></div>
        <div className="px-3 text-[#555555]">|</div>
        <div className="px-3 text-[#888888]">CV-AUC: <span className="text-[#00d97e]">{(summary.cv_auc * 100).toFixed(2)}%</span></div>
        <div className="ml-auto px-3 text-[#888888]">{time}</div>
      </div>

      {/* ── Ticker bar (running stats) ────────────────────────────── */}
      <div className="border-b border-[#1f1f1f] bg-[#0a0a0a] overflow-hidden">
        <div className="ticker-track text-[11px] py-1 tabular-nums">
          {[...Array(2)].map((_, k) => (
            <span key={k} className="flex">
              {[
                ["ROC-AUC",  `${(metrics.roc_auc * 100).toFixed(2)}%`,  "green"],
                ["RECALL",   `${(metrics.recall_fraud * 100).toFixed(1)}%`, "green"],
                ["PRECISION",`${(metrics.precision_fraud * 100).toFixed(1)}%`,"green"],
                ["LOSS PREV",`$${metrics.loss_prevented.toLocaleString()}`,"green"],
                ["LOSS MISS",`$${metrics.missed_loss.toLocaleString()}`,"red"],
                ["FALSE+",   `${metrics.n_false_alarms}`, "red"],
                ["FRAUD+",   `${metrics.n_fraud_caught}`, "green"],
                ["F1",       `${(metrics.f1_fraud * 100).toFixed(1)}%`, "green"],
                ["MCC",      `${(metrics.mcc).toFixed(3)}`, "green"],
                ["BAL-ACC",  `${(metrics.balanced_accuracy * 100).toFixed(1)}%`, "green"],
              ].map(([label, val, dir], i) => (
                <span key={`${k}-${i}`} className="px-6 flex items-center gap-2">
                  <span className="text-[#888888]">{label}</span>
                  <span className={dir === "green" ? "text-[#00d97e]" : "text-[#ff3344]"}>
                    {dir === "green" ? "▲" : "▼"} {val}
                  </span>
                  <span className="text-[#2a2a2a]">|</span>
                </span>
              ))}
            </span>
          ))}
        </div>
      </div>

      {/* ── Main grid ─────────────────────────────────────────────── */}
      <div className="p-3 grid grid-cols-12 gap-3">

        {/* Header / command line */}
        <div className="col-span-12 flex items-end justify-between border-b border-[#1f1f1f] pb-2 mb-1">
          <div>
            <div className="text-[10px] tracking-widest text-[#555555] uppercase">{"<FRAUD>"} {"<EQUITY>"} {"<RISK>"} GO</div>
            <h1 className="text-2xl font-bold text-[#ffaa00] tracking-wide caret">FRAUD.MODEL</h1>
            <div className="text-[11px] text-[#888888]">CREDIT CARD TRANSACTION FRAUD CLASSIFIER · {summary.dataset.toUpperCase()}</div>
          </div>
          <div className="text-right text-[10px] text-[#555555] tracking-wider">
            <div>FEATURES: <span className="text-[#e8e8e8]">{summary.n_features_used}</span></div>
            <div>TRAIN N: <span className="text-[#e8e8e8]">{summary.n_train.toLocaleString()}</span></div>
            <div>TEST N: <span className="text-[#e8e8e8]">{summary.n_test.toLocaleString()}</span></div>
          </div>
        </div>

        {/* KEY STATS — 4 across */}
        <div className="col-span-12 grid grid-cols-2 lg:grid-cols-4 gap-3">
          <Stat
            label="LOSS PREVENTED (TP)"
            value={`$${metrics.loss_prevented.toLocaleString()}`}
            sub="TEST SET"
            color="text-[#00d97e]"
            trend="up"
          />
          <Stat
            label="% FRAUD VALUE CAUGHT"
            value={`${metrics.pct_prevented}%`}
            sub="OF TOTAL FRAUD $"
            color="text-[#00d97e]"
            trend="up"
          />
          <Stat
            label="ROC-AUC SCORE"
            value={(metrics.roc_auc * 100).toFixed(2) + "%"}
            sub="HELD-OUT TEST"
            color="text-[#ffaa00]"
          />
          <Stat
            label="FALSE ALARMS"
            value={metrics.n_false_alarms}
            sub={`$${metrics.false_alarm_value.toLocaleString()}`}
            color="text-[#ff3344]"
            trend="down"
          />
        </div>

        {/* DOLLAR IMPACT TABLE + MODEL COMPARISON */}
        <div className="col-span-12 lg:col-span-7">
          <Panel title="P&L IMPACT" code="DLR>">
            <table className="w-full text-[12px] tabular-nums">
              <tbody>
                {[
                  ["TOTAL FRAUD VALUE",      `$${metrics.total_fraud_value.toLocaleString()}`, "text-[#888888]"],
                  ["▲ LOSS PREVENTED  (TP)", `$${metrics.loss_prevented.toLocaleString()}`,    "text-[#00d97e]"],
                  ["▼ LOSS MISSED     (FN)", `$${metrics.missed_loss.toLocaleString()}`,       "text-[#ff3344]"],
                  ["▼ FALSE ALARM VAL (FP)", `$${metrics.false_alarm_value.toLocaleString()}`, "text-[#ffaa00]"],
                  ["FRAUD CAUGHT       (#)", metrics.n_fraud_caught,  "text-[#00d97e]"],
                  ["FRAUD MISSED       (#)", metrics.n_fraud_missed,  "text-[#ff3344]"],
                  ["FRAUD RATE       (%)",  `${metrics.fraud_rate_pct}%`, "text-[#888888]"],
                ].map(([label, val, color], i) => (
                  <tr key={i} className="border-b border-[#1f1f1f] last:border-0">
                    <td className="py-1.5 text-[#888888] uppercase tracking-wider">{label}</td>
                    <td className={`py-1.5 text-right font-bold ${color}`}>{val}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
        </div>

        <div className="col-span-12 lg:col-span-5">
          {summary.all_model_test_aucs && (
            <Panel title="MODEL COMPARISON" code="MDL>">
              <ResponsiveContainer width="100%" height={210}>
                <BarChart
                  data={Object.entries(summary.all_model_test_aucs).map(([name, auc]) => ({ name, auc }))}
                  margin={{ left: -10, right: 10, top: 10, bottom: 0 }}
                >
                  <CartesianGrid stroke="#1f1f1f" vertical={false} />
                  <XAxis dataKey="name" tick={{ fill: "#888888", fontSize: 10 }} axisLine={{ stroke: "#2a2a2a" }} />
                  <YAxis domain={[0.5, 1.0]} tick={{ fill: "#888888", fontSize: 10 }} axisLine={{ stroke: "#2a2a2a" }} />
                  <Tooltip
                    cursor={{ fill: "#1f1f1f" }}
                    formatter={v => [`${(Number(v) * 100).toFixed(2)}%`, "AUC"]}
                    contentStyle={{ background: "#000", border: "1px solid #ffaa00", fontSize: 11, fontFamily: "JetBrains Mono" }}
                  />
                  <Bar dataKey="auc">
                    {Object.keys(summary.all_model_test_aucs).map((name, i) => (
                      <Cell key={i} fill={name === summary.model_name?.split(" ")[0] ? "#ffaa00" : "#00b8d4"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="text-[10px] text-[#555555] mt-1 tracking-wider uppercase text-center">
                ▲ AMBER = SELECTED MODEL · CYAN = CHALLENGERS
              </div>
            </Panel>
          )}
        </div>

        {/* TRANSACTION SCANNER */}
        <div className="col-span-12">
          <Panel title="LIVE TRANSACTION SCANNER" code="TXN>">
            <div className="text-[11px] text-[#888888] mb-3 uppercase tracking-wider">
              {">"} SELECT TRANSACTION TO ANALYZE
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-2 mb-4">
              {scenarios.map(s => {
                const isActive = selected?.id === s.id
                const riskColor =
                  s.risk_level === "FLAGGED" ? "#ff3344" :
                  s.risk_level === "REVIEW"  ? "#ffaa00" : "#00d97e"
                return (
                  <button
                    key={s.id}
                    onClick={() => handleSelectScenario(s)}
                    className={`text-left border bg-[#0a0a0a] hover:bg-[#161616] transition-colors p-2.5 ${
                      isActive ? "border-[#ffaa00]" : "border-[#1f1f1f]"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[10px] font-bold tracking-widest" style={{ color: riskColor }}>
                        ● {s.risk_level}
                      </span>
                      <span className="text-[10px] text-[#555555] tabular-nums">{s.id.toUpperCase()}</span>
                    </div>
                    <div className="text-[12px] text-[#e8e8e8] leading-tight mb-1.5 truncate">{s.title}</div>
                    <div className="text-base text-[#ffaa00] font-bold tabular-nums">{s.amount}</div>
                    <div className="text-[10px] text-[#555555] mt-0.5">{s.time}</div>
                  </button>
                )
              })}
            </div>

            {selected && (
              <div className="border-t border-[#1f1f1f] pt-4">
                <div className="grid grid-cols-12 gap-4">

                  {/* Gauge */}
                  <div className="col-span-12 md:col-span-3 flex flex-col items-center justify-center border-r border-[#1f1f1f] pr-4">
                    <FraudGauge value={liveResult?.fraud_probability ?? selected.fraud_probability} />
                    {liveResult && !liveResult.error && (
                      <div className="text-[10px] text-[#00d97e] mt-2 tracking-widest">● LIVE AWS</div>
                    )}
                  </div>

                  {/* Signals */}
                  <div className="col-span-12 md:col-span-5">
                    <div className="text-[10px] text-[#888888] uppercase tracking-widest mb-2">► RISK SIGNALS</div>
                    <ul className="space-y-1.5 text-[12px]">
                      {selected.key_signals.map((sig, i) => (
                        <li key={i} className="flex gap-2 text-[#e8e8e8]">
                          <span className="text-[#ffaa00]">›</span>
                          <span>{sig}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Live AWS */}
                  <div className="col-span-12 md:col-span-4 flex flex-col gap-2">
                    <div className="text-[10px] text-[#888888] uppercase tracking-widest">► LIVE INFERENCE</div>
                    {selected.features ? (
                      <>
                        <div className="text-[11px] text-[#555555] leading-relaxed">
                          POST features → SAGEMAKER ENDPOINT
                        </div>
                        <button
                          onClick={handleLiveScore}
                          disabled={liveLoading}
                          className="border border-[#ffaa00] bg-[#ffaa00]/10 hover:bg-[#ffaa00] hover:text-black text-[#ffaa00] py-1.5 px-3 text-[11px] font-bold tracking-widest disabled:opacity-50 transition-colors"
                        >
                          {liveLoading ? "▶ SCORING..." : "▶ EXECUTE [ENTER]"}
                        </button>
                        {liveResult?.error && (
                          <div className="text-[10px] text-[#ff3344] tracking-wider">ERR: {liveResult.error}</div>
                        )}
                        {liveResult && !liveResult.error && (
                          <div className="text-[10px] text-[#00d97e] tracking-wider tabular-nums">
                            ► RESPONSE: {(liveResult.fraud_probability * 100).toFixed(2)}% [{liveResult.risk_level}]
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="text-[10px] text-[#555555] leading-relaxed">
                        DEPLOY ENDPOINT VIA sagemaker_deploy.py TO ENABLE
                      </div>
                    )}
                  </div>
                </div>

                {/* SHAP local contributions */}
                {selected.shap_contributions?.length > 0 && (
                  <div className="mt-4 pt-3 border-t border-[#1f1f1f]">
                    <div className="text-[10px] text-[#888888] uppercase tracking-widest mb-2">
                      ► SHAP DECOMPOSITION · PER-FEATURE CONTRIBUTION
                    </div>
                    <ResponsiveContainer width="100%" height={Math.max(160, selected.shap_contributions.length * 28)}>
                      <BarChart
                        data={[...selected.shap_contributions].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))}
                        layout="vertical"
                        margin={{ left: 0, right: 16, top: 4, bottom: 4 }}
                      >
                        <CartesianGrid stroke="#1f1f1f" horizontal={false} />
                        <XAxis type="number" tick={{ fill: "#888888", fontSize: 10 }} axisLine={{ stroke: "#2a2a2a" }} />
                        <YAxis type="category" dataKey="feature" tick={{ fill: "#e8e8e8", fontSize: 10 }} width={64} axisLine={{ stroke: "#2a2a2a" }} />
                        <Tooltip
                          cursor={{ fill: "#1f1f1f" }}
                          formatter={(v, _n, { payload }) => [Number(v).toFixed(4), payload.description]}
                          contentStyle={{ background: "#000", border: "1px solid #ffaa00", fontSize: 11, fontFamily: "JetBrains Mono" }}
                        />
                        <Bar dataKey="shap_value">
                          {selected.shap_contributions.map((entry, i) => (
                            <Cell key={i} fill={entry.shap_value > 0 ? "#ff3344" : "#00b8d4"} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                    {selected.shap_base_value !== undefined && (
                      <div className="text-[10px] text-[#555555] mt-1 tabular-nums tracking-wider">
                        BASE: {(selected.shap_base_value * 100).toFixed(2)}% → FINAL: {(selected.fraud_probability * 100).toFixed(2)}%
                        {" · "}RED = FRAUD↑ · CYAN = LEGIT↑
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </Panel>
        </div>

        {/* SHAP GLOBAL FEATURE IMPORTANCE */}
        <div className="col-span-12 lg:col-span-7">
          <Panel title="SHAP FEATURE IMPORTANCE" code="SHP>">
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={top10} layout="vertical" margin={{ left: 0, right: 16, top: 4, bottom: 4 }}>
                <CartesianGrid stroke="#1f1f1f" horizontal={false} />
                <XAxis type="number" tick={{ fill: "#888888", fontSize: 10 }} axisLine={{ stroke: "#2a2a2a" }} />
                <YAxis type="category" dataKey="feature" tick={{ fill: "#e8e8e8", fontSize: 10 }} width={64} axisLine={{ stroke: "#2a2a2a" }} />
                <Tooltip
                  cursor={{ fill: "#1f1f1f" }}
                  formatter={(v, _n, { payload }) => [Number(v).toFixed(4), payload.description]}
                  contentStyle={{ background: "#000", border: "1px solid #ffaa00", fontSize: 11, fontFamily: "JetBrains Mono" }}
                />
                <Bar dataKey="importance" fill="#ffaa00" />
              </BarChart>
            </ResponsiveContainer>
          </Panel>
        </div>

        {/* MODEL DIAGNOSTICS */}
        <div className="col-span-12 lg:col-span-5">
          <Panel title="MODEL DIAGNOSTICS" code="DGN>">
            <table className="w-full text-[11px] tabular-nums">
              <thead>
                <tr className="border-b border-[#1f1f1f] text-[#555555] uppercase">
                  <th className="text-left py-1.5 font-normal tracking-wider">METRIC</th>
                  <th className="text-right py-1.5 font-normal tracking-wider">VALUE</th>
                </tr>
              </thead>
              <tbody className="text-[#e8e8e8]">
                {[
                  ["ROC-AUC",            (metrics.roc_auc * 100).toFixed(2) + "%",            "text-[#ffaa00]"],
                  ["PRECISION (FRAUD)",  (metrics.precision_fraud * 100).toFixed(2) + "%",    ""],
                  ["RECALL (FRAUD)",     (metrics.recall_fraud * 100).toFixed(2) + "%",       "text-[#00d97e]"],
                  ["F1 (FRAUD)",         (metrics.f1_fraud * 100).toFixed(2) + "%",           ""],
                  ["BALANCED ACCURACY",  (metrics.balanced_accuracy * 100).toFixed(2) + "%",  ""],
                  ["MCC",                metrics.mcc?.toFixed(4),                              ""],
                  ["TRAIN AUC",          metrics.train_auc ? (metrics.train_auc * 100).toFixed(2) + "%" : "—",  "text-[#888888]"],
                  ["TEST AUC",           metrics.test_auc  ? (metrics.test_auc  * 100).toFixed(2) + "%" : "—",  ""],
                  ["TRAIN-TEST GAP",     metrics.train_test_gap ? (metrics.train_test_gap * 100).toFixed(2) + " bps" : "—", "text-[#888888]"],
                  ["DECISION THRESHOLD", metrics.best_threshold,                                "text-[#888888]"],
                ].map(([k, v, color], i) => (
                  <tr key={i} className="border-b border-[#1f1f1f] last:border-0">
                    <td className="py-1.5 text-[#888888] tracking-wider uppercase">{k}</td>
                    <td className={`py-1.5 text-right font-bold ${color}`}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
        </div>

        {/* FEATURE LEGEND */}
        <div className="col-span-12">
          <Panel title="FEATURE GLOSSARY" code="GLS>">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-1 text-[11px]">
              {top10.map(f => (
                <div key={f.feature} className="flex items-baseline gap-3 border-b border-[#1f1f1f] py-1">
                  <span className="text-[#ffaa00] font-bold w-16 tabular-nums">{f.feature}</span>
                  <span className="text-[#888888] flex-1">{f.description}</span>
                  <span className="text-[#555555] tabular-nums text-[10px]">{Number(f.importance).toFixed(4)}</span>
                </div>
              ))}
            </div>
          </Panel>
        </div>

        {/* FOOTER */}
        <div className="col-span-12 border-t border-[#1f1f1f] pt-2 mt-2 flex items-center justify-between text-[10px] text-[#555555] tracking-wider uppercase">
          <span>{">"} {summary.dataset} · {summary.model_name}</span>
          <span>SMOTE: {summary.smote_applied ? "ON" : "OFF"} · CV-FOLDS: {summary.cv_folds} · FEATURES: {summary.n_features_used}</span>
          <span className="text-[#ffaa00]">[F1] HELP · [F2] EXPORT · [ESC] EXIT</span>
        </div>
      </div>
    </main>
  )
}
