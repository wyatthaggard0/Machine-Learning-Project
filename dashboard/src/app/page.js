"use client"

import { useEffect, useState, useCallback } from "react"
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, Cell, PieChart, Pie,
} from "recharts"

/* ──────────────────────────────────────────────────────────────────
   Databox-style primitives
   ────────────────────────────────────────────────────────────────── */

function Card({ title, subtitle, children, className = "" }) {
  return (
    <div className={`bg-[#1d5742] rounded-xl px-6 py-5 ${className}`}>
      {(title || subtitle) && (
        <div className="text-center mb-4">
          {title && (
            <h3 className="text-[11px] font-bold tracking-[0.18em] text-white uppercase">
              {title}
            </h3>
          )}
          {subtitle && (
            <p className="text-[11px] text-[#80a194] mt-0.5">{subtitle}</p>
          )}
        </div>
      )}
      {children}
    </div>
  )
}

function Delta({ value, suffix = "%" }) {
  if (value == null) return null
  const positive = value >= 0
  const color = positive ? "text-[#22c773] bg-[#22c773]/15" : "text-[#ff5d5d] bg-[#ff5d5d]/15"
  return (
    <span className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] font-bold ${color}`}>
      {positive ? "▲" : "▼"} {Math.abs(value).toFixed(1)}{suffix}
    </span>
  )
}

function FraudGauge({ value }) {
  const pct = Math.round(value * 100)
  const level = value >= 0.5 ? "FLAGGED" : value >= 0.25 ? "REVIEW" : "SAFE"
  const color = value >= 0.5 ? "#ff5d5d" : value >= 0.25 ? "#ffa84a" : "#22c773"

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-44 h-44">
        <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
          <circle cx="60" cy="60" r="50" fill="none" stroke="#164534" strokeWidth="14" />
          <circle
            cx="60" cy="60" r="50" fill="none"
            stroke={color}
            strokeWidth="14"
            strokeDasharray={`${pct * 3.142} 314.2`}
            strokeLinecap="round"
            style={{ transition: "stroke-dasharray 0.6s ease" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-4xl font-bold text-white tabular-nums">{pct}%</span>
          <span className="text-[10px] tracking-widest text-[#80a194] uppercase mt-0.5">Fraud Risk</span>
        </div>
      </div>
      <div
        className="mt-3 px-3 py-1 rounded-full text-[11px] font-bold tracking-wider"
        style={{ background: `${color}26`, color }}
      >
        {level}
      </div>
    </div>
  )
}

/* ──────────────────────────────────────────────────────────────────
   Page
   ────────────────────────────────────────────────────────────────── */

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

  const handleSelect = useCallback((s) => {
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
      <div className="min-h-screen flex items-center justify-center text-white">
        Loading dashboard…
      </div>
    )
  }

  // Donut for fraud caught vs missed
  const fraudPieData = [
    { name: "Caught",  value: metrics.loss_prevented, fill: "#ff6a13" },
    { name: "Missed",  value: metrics.missed_loss,    fill: "#164534" },
  ]

  // Top features for SHAP global chart
  const top10 = features.slice(0, 10).map(f => ({
    ...f,
    importance: f.shap_importance ?? Math.abs(f.coefficient ?? 0),
  }))

  return (
    <main className="min-h-screen bg-[#0e3a2c] text-white p-4 lg:p-6">
      {/* ── Browser-chrome top bar ────────────────────────────────── */}
      <div className="bg-[#f1f4f1] rounded-t-xl px-4 py-2.5 flex items-center gap-2 mb-0">
        <span className="w-3 h-3 rounded-full bg-[#ff5f56]" />
        <span className="w-3 h-3 rounded-full bg-[#ffbd2e]" />
        <span className="w-3 h-3 rounded-full bg-[#27c93f]" />
        <span className="ml-3 text-[12px] text-[#666]">Fraud Detection · Banker Dashboard</span>
      </div>

      {/* ── Dashboard frame ───────────────────────────────────────── */}
      <div className="bg-[#0e3a2c] rounded-b-xl p-4 lg:p-5 border border-t-0 border-[#1d5742]">

        {/* Header strip */}
        <div className="flex flex-col md:flex-row md:items-center justify-between mb-5 gap-3">
          <div>
            <h1 className="text-2xl font-bold text-white">Fraud Detection Model</h1>
            <p className="text-[12px] text-[#bfd6cb]">
              IEEE-CIS Credit Card Transactions · Model: <span className="text-[#ff6a13] font-semibold">{summary.model_name}</span>
            </p>
          </div>
          <div className="flex items-center gap-2 text-[11px] text-[#bfd6cb]">
            <span className="w-2 h-2 rounded-full bg-[#22c773] animate-pulse" />
            <span>Live · Model deployed on AWS SageMaker</span>
          </div>
        </div>

        {/* ── Row 1: 3 stat tiles + donut ─────────────────────────── */}
        <div className="grid grid-cols-12 gap-4 mb-4">

          <Card title="Loss Prevented" subtitle="Test set · cumulative" className="col-span-12 md:col-span-3">
            <div className="text-center">
              <div className="text-4xl font-bold text-white tabular-nums mb-2">
                ${(metrics.loss_prevented / 1000).toFixed(2)}<span className="text-2xl text-[#bfd6cb]">k</span>
              </div>
              <div className="flex items-center justify-center gap-2">
                <Delta value={metrics.pct_prevented} suffix="%" />
                <span className="text-[10px] text-[#80a194] uppercase tracking-wide">caught</span>
              </div>
              {/* Mini bar chart of caught vs missed */}
              <div className="mt-4 flex items-end justify-center gap-2 h-14">
                {[
                  { label: "Caught",  val: metrics.loss_prevented, color: "#ff6a13" },
                  { label: "Missed",  val: metrics.missed_loss,    color: "#164534" },
                  { label: "False+",  val: metrics.false_alarm_value, color: "#ffa84a" },
                ].map(b => {
                  const max = Math.max(metrics.loss_prevented, metrics.missed_loss, metrics.false_alarm_value, 1)
                  return (
                    <div key={b.label} className="flex flex-col items-center">
                      <div className="w-6 rounded-t" style={{ height: `${(b.val / max) * 56}px`, background: b.color, minHeight: "4px" }} />
                      <span className="text-[9px] text-[#80a194] mt-1">{b.label}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          </Card>

          <Card title="Model Performance" subtitle="On held-out test set" className="col-span-12 md:col-span-4">
            <table className="w-full text-[12px]">
              <tbody>
                {[
                  ["ROC-AUC",          (metrics.roc_auc * 100).toFixed(1) + "%",          "▲", "up"],
                  ["Recall (fraud)",   (metrics.recall_fraud * 100).toFixed(1) + "%",     "▲", "up"],
                  ["Precision (fraud)",(metrics.precision_fraud * 100).toFixed(1) + "%",  "▲", "up"],
                  ["F1 Score",         (metrics.f1_fraud * 100).toFixed(1) + "%",         "▲", "up"],
                  ["Balanced Accuracy",(metrics.balanced_accuracy * 100).toFixed(1) + "%","▲", "up"],
                ].map(([k, v, icon, dir], i) => (
                  <tr key={i} className="border-b border-[#164534] last:border-0">
                    <td className="py-1.5 text-[#bfd6cb]">{k}</td>
                    <td className="py-1.5 text-right font-semibold text-white tabular-nums">{v}</td>
                    <td className="py-1.5 pl-2 text-right">
                      <span className={`text-[10px] ${dir === "up" ? "text-[#22c773]" : "text-[#ff5d5d]"}`}>{icon}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Bank Loss Coverage" subtitle="Fraud value caught vs missed" className="col-span-12 md:col-span-5">
            <div className="grid grid-cols-2 gap-4 items-center">
              <div className="relative h-44">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={fraudPieData}
                      cx="50%" cy="50%"
                      innerRadius={50} outerRadius={75}
                      dataKey="value"
                      stroke="none"
                    >
                      {fraudPieData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                    </Pie>
                    <Tooltip
                      formatter={v => [`$${Number(v).toLocaleString()}`, ""]}
                      contentStyle={{ background: "#0e3a2c", border: "1px solid #2a7257", borderRadius: 8, fontSize: 11 }}
                    />
                  </PieChart>
                </ResponsiveContainer>
                <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                  <span className="text-2xl font-bold text-white tabular-nums">{metrics.pct_prevented}%</span>
                  <span className="text-[10px] text-[#80a194] uppercase tracking-wider">recovered</span>
                </div>
              </div>
              <div className="space-y-2 text-[12px]">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-[#ff6a13]" />
                    <span className="text-[#bfd6cb]">Caught</span>
                  </div>
                  <div className="text-base font-bold text-white pl-4 tabular-nums">
                    ${metrics.loss_prevented.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-[#164534] border border-[#2a7257]" />
                    <span className="text-[#bfd6cb]">Missed</span>
                  </div>
                  <div className="text-base font-bold text-white pl-4 tabular-nums">
                    ${metrics.missed_loss.toLocaleString()}
                  </div>
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-[#ffa84a]" />
                    <span className="text-[#bfd6cb]">False alarm $</span>
                  </div>
                  <div className="text-base font-bold text-white pl-4 tabular-nums">
                    ${metrics.false_alarm_value.toLocaleString()}
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </div>

        {/* ── Row 2: model comparison + diagnostics ───────────────── */}
        <div className="grid grid-cols-12 gap-4 mb-4">

          {summary.all_model_test_aucs && (
            <Card title="Model Comparison" subtitle="ROC-AUC across 4 algorithms" className="col-span-12 md:col-span-7">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart
                  data={Object.entries(summary.all_model_test_aucs).map(([name, auc]) => ({ name, auc }))}
                  margin={{ left: -10, right: 10, top: 10, bottom: 0 }}
                >
                  <CartesianGrid stroke="#164534" vertical={false} />
                  <XAxis dataKey="name" tick={{ fill: "#bfd6cb", fontSize: 11 }} axisLine={{ stroke: "#2a7257" }} tickLine={false} />
                  <YAxis domain={[0.5, 1.0]} tick={{ fill: "#80a194", fontSize: 10 }} axisLine={{ stroke: "#2a7257" }} tickLine={false} />
                  <Tooltip
                    cursor={{ fill: "rgba(255,106,19,0.08)" }}
                    formatter={v => [`${(Number(v) * 100).toFixed(2)}%`, "Test ROC-AUC"]}
                    contentStyle={{ background: "#0e3a2c", border: "1px solid #2a7257", borderRadius: 8, fontSize: 11 }}
                  />
                  <Bar dataKey="auc" radius={[6, 6, 0, 0]}>
                    {Object.keys(summary.all_model_test_aucs).map((name, i) => (
                      <Cell key={i} fill={name === summary.model_name?.split(" ")[0] ? "#ff6a13" : "#2a7257"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="text-[10px] text-[#80a194] mt-1 text-center uppercase tracking-wide">
                Orange = winning model · Green = challengers
              </div>
            </Card>
          )}

          <Card title="Training Overview" subtitle="Pipeline & diagnostics" className="col-span-12 md:col-span-5">
            <div className="grid grid-cols-2 gap-3 text-[12px]">
              {[
                ["Algorithm",        summary.model_name?.split(" ")[0]],
                ["Features used",    summary.n_features_used],
                ["Training rows",    summary.n_train.toLocaleString()],
                ["Test rows",        summary.n_test.toLocaleString()],
                ["CV folds",         summary.cv_folds],
                ["CV ROC-AUC",       (summary.cv_auc * 100).toFixed(2) + "%"],
                ["Resampling",       summary.smote_applied ? "SMOTE" : "None"],
                ["Train AUC",        metrics.train_auc ? (metrics.train_auc * 100).toFixed(2) + "%" : "—"],
                ["Test AUC",         metrics.test_auc  ? (metrics.test_auc  * 100).toFixed(2) + "%" : "—"],
                ["Train-Test Gap",   metrics.train_test_gap != null
                                       ? (metrics.train_test_gap * 100).toFixed(2) + " pp"
                                       : "—"],
              ].map(([k, v], i) => (
                <div key={i} className="bg-[#164534] rounded-lg px-3 py-2">
                  <div className="text-[10px] text-[#80a194] uppercase tracking-wider">{k}</div>
                  <div className="text-sm font-bold text-white tabular-nums">{v ?? "—"}</div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* ── Row 3: live transaction scanner ─────────────────────── */}
        <Card title="Live Transaction Scanner" subtitle="Click any transaction to see how the model scores it" className="mb-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 mb-5">
            {scenarios.map(s => {
              const isActive = selected?.id === s.id
              const riskColor =
                s.risk_level === "FLAGGED" ? "#ff5d5d" :
                s.risk_level === "REVIEW"  ? "#ffa84a" : "#22c773"
              return (
                <button
                  key={s.id}
                  onClick={() => handleSelect(s)}
                  className={`text-left rounded-xl p-3.5 transition-all ${
                    isActive
                      ? "bg-[#ff6a13]/15 ring-2 ring-[#ff6a13]"
                      : "bg-[#164534] hover:bg-[#1c5740]"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="inline-flex items-center gap-1 text-[10px] font-bold tracking-wider"
                          style={{ color: riskColor }}>
                      ● {s.risk_level}
                    </span>
                  </div>
                  <div className="text-[12px] text-white leading-tight mb-2 line-clamp-2">{s.title}</div>
                  <div className="text-xl font-bold text-[#ff6a13] tabular-nums">{s.amount}</div>
                  <div className="text-[10px] text-[#80a194] mt-0.5">{s.time}</div>
                </button>
              )
            })}
          </div>

          {selected && (
            <div className="bg-[#164534] rounded-xl p-5">
              <div className="grid grid-cols-12 gap-6">
                {/* Gauge */}
                <div className="col-span-12 md:col-span-3 flex justify-center md:border-r md:border-[#2a7257]">
                  <div className="flex flex-col items-center">
                    <FraudGauge value={liveResult?.fraud_probability ?? selected.fraud_probability} />
                    {liveResult && !liveResult.error && (
                      <div className="text-[10px] text-[#22c773] mt-2 tracking-wider">● Live AWS result</div>
                    )}
                  </div>
                </div>

                {/* Risk signals */}
                <div className="col-span-12 md:col-span-5">
                  <div className="text-[11px] font-semibold text-[#bfd6cb] uppercase tracking-wider mb-2">
                    Why the model flagged this
                  </div>
                  <ul className="space-y-2">
                    {selected.key_signals.map((sig, i) => (
                      <li key={i} className="flex gap-2 text-[13px] text-white leading-relaxed">
                        <span className="text-[#ff6a13] font-bold">›</span>
                        <span>{sig}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Live AWS */}
                <div className="col-span-12 md:col-span-4">
                  <div className="text-[11px] font-semibold text-[#bfd6cb] uppercase tracking-wider mb-2">
                    Live AWS Endpoint
                  </div>
                  {selected.features ? (
                    <>
                      <p className="text-[12px] text-[#80a194] leading-relaxed mb-3">
                        Send this transaction to the deployed SageMaker model for a real-time score.
                      </p>
                      <button
                        onClick={handleLiveScore}
                        disabled={liveLoading}
                        className="w-full bg-[#ff6a13] hover:bg-[#e85b08] text-white font-semibold py-2 px-4 rounded-lg text-[13px] disabled:opacity-50 transition-colors"
                      >
                        {liveLoading ? "Scoring…" : "Score via AWS →"}
                      </button>
                      {liveResult?.error && (
                        <p className="text-[11px] text-[#ff5d5d] mt-2">{liveResult.error}</p>
                      )}
                      {liveResult && !liveResult.error && (
                        <p className="text-[11px] text-[#22c773] mt-2 tabular-nums">
                          Live score: <span className="font-bold">{(liveResult.fraud_probability * 100).toFixed(1)}%</span> · {liveResult.risk_level}
                        </p>
                      )}
                    </>
                  ) : (
                    <p className="text-[12px] text-[#80a194] leading-relaxed">
                      Run <code className="bg-[#0e3a2c] px-1.5 py-0.5 rounded text-[#ff6a13]">sagemaker_deploy.py</code> to enable live scoring.
                    </p>
                  )}
                </div>
              </div>

              {/* SHAP local contribution chart */}
              {selected.shap_contributions?.length > 0 && (
                <div className="mt-6 pt-5 border-t border-[#2a7257]">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <div className="text-[11px] font-bold text-white uppercase tracking-wider">
                        SHAP Decomposition · Per-Feature Contribution
                      </div>
                      <div className="text-[11px] text-[#80a194] mt-0.5">
                        Orange bars push toward fraud · Green bars push toward legitimate
                      </div>
                    </div>
                    {selected.shap_base_value !== undefined && (
                      <div className="text-right">
                        <div className="text-[10px] text-[#80a194] uppercase tracking-wider">Base → Final</div>
                        <div className="text-[12px] text-white font-semibold tabular-nums">
                          {(selected.shap_base_value * 100).toFixed(1)}% → {(selected.fraud_probability * 100).toFixed(1)}%
                        </div>
                      </div>
                    )}
                  </div>
                  <ResponsiveContainer
                    width="100%"
                    height={Math.max(180, selected.shap_contributions.length * 36)}
                  >
                    <BarChart
                      data={[...selected.shap_contributions].sort(
                        (a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value)
                      )}
                      layout="vertical"
                      margin={{ left: 0, right: 24, top: 4, bottom: 4 }}
                    >
                      <CartesianGrid stroke="#0e3a2c" horizontal={false} />
                      <XAxis type="number" tick={{ fill: "#80a194", fontSize: 11 }} axisLine={{ stroke: "#2a7257" }} tickLine={false} />
                      <YAxis type="category" dataKey="feature" tick={{ fill: "#ffffff", fontSize: 12, fontWeight: 600 }} width={72} axisLine={{ stroke: "#2a7257" }} tickLine={false} />
                      <Tooltip
                        cursor={{ fill: "rgba(255,106,19,0.06)" }}
                        formatter={(v, _n, { payload }) => [Number(v).toFixed(4), payload.description]}
                        contentStyle={{ background: "#0e3a2c", border: "1px solid #2a7257", borderRadius: 8, fontSize: 11 }}
                      />
                      <Bar dataKey="shap_value" radius={[0, 4, 4, 0]}>
                        {selected.shap_contributions.map((entry, i) => (
                          <Cell key={i} fill={entry.shap_value > 0 ? "#ff6a13" : "#22c773"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}
        </Card>

        {/* ── Row 4: SHAP global + glossary ───────────────────────── */}
        <div className="grid grid-cols-12 gap-4 mb-4">
          <Card title="SHAP Feature Importance" subtitle="Global · mean |SHAP value|" className="col-span-12 md:col-span-7">
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={top10} layout="vertical" margin={{ left: 0, right: 16, top: 4, bottom: 4 }}>
                <CartesianGrid stroke="#164534" horizontal={false} />
                <XAxis type="number" tick={{ fill: "#80a194", fontSize: 11 }} axisLine={{ stroke: "#2a7257" }} tickLine={false} />
                <YAxis type="category" dataKey="feature" tick={{ fill: "#ffffff", fontSize: 11, fontWeight: 600 }} width={72} axisLine={{ stroke: "#2a7257" }} tickLine={false} />
                <Tooltip
                  cursor={{ fill: "rgba(255,106,19,0.08)" }}
                  formatter={(v, _n, { payload }) => [Number(v).toFixed(4), payload.description]}
                  contentStyle={{ background: "#0e3a2c", border: "1px solid #2a7257", borderRadius: 8, fontSize: 11 }}
                />
                <Bar dataKey="importance" fill="#ff6a13" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Feature Glossary" subtitle="Plain-English signal descriptions" className="col-span-12 md:col-span-5">
            <div className="space-y-1.5 max-h-[340px] overflow-y-auto pr-1">
              {top10.map(f => (
                <div key={f.feature} className="bg-[#164534] rounded-lg px-3 py-2">
                  <div className="flex items-baseline justify-between mb-0.5">
                    <span className="text-[12px] font-bold text-[#ff6a13]">{f.feature}</span>
                    <span className="text-[10px] text-[#80a194] tabular-nums">
                      {Number(f.importance).toFixed(4)}
                    </span>
                  </div>
                  <div className="text-[11px] text-[#bfd6cb] leading-snug">{f.description}</div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* ── Footer ──────────────────────────────────────────────── */}
        <div className="flex flex-col md:flex-row items-center justify-between text-[11px] text-[#80a194] pt-3 border-t border-[#1d5742] gap-2">
          <span>{summary.dataset} · {summary.model_name}</span>
          <span className="flex items-center gap-2">
            <span className="bg-[#ff6a13] text-white text-[10px] font-bold px-2 py-0.5 rounded">LIVE DEMO</span>
            Banker dashboard · Built on AWS SageMaker + Vercel
          </span>
        </div>
      </div>
    </main>
  )
}
