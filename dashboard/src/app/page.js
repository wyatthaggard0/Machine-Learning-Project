"use client"

import { useEffect, useState, useCallback } from "react"
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell, PieChart, Pie, Legend,
} from "recharts"

function StatCard({ label, value, sub, color }) {
  return (
    <div className="bg-[#16213e] rounded-2xl p-6 flex flex-col gap-1 border border-[#0f3460]">
      <span className="text-sm text-gray-400 uppercase tracking-wide">{label}</span>
      <span className={`text-3xl font-bold ${color ?? "text-white"}`}>{value}</span>
      {sub && <span className="text-xs text-gray-500">{sub}</span>}
    </div>
  )
}

function Section({ title, children }) {
  return (
    <section className="mb-12">
      <h2 className="text-xl font-semibold text-gray-200 mb-5 border-l-4 border-[#3498db] pl-3">
        {title}
      </h2>
      {children}
    </section>
  )
}

const COLORS = { positive: "#e74c3c", negative: "#3498db" }

const RISK_CONFIG = {
  SAFE:    { color: "text-[#2ecc71]", bg: "bg-[#2ecc71]/10 border-[#2ecc71]/30", dot: "#2ecc71", label: "SAFE — Approve" },
  REVIEW:  { color: "text-[#f39c12]", bg: "bg-[#f39c12]/10 border-[#f39c12]/30", dot: "#f39c12", label: "REVIEW — Flag for Analyst" },
  FLAGGED: { color: "text-[#e74c3c]", bg: "bg-[#e74c3c]/10 border-[#e74c3c]/30", dot: "#e74c3c", label: "FLAGGED — Decline" },
}

function FraudGauge({ value }) {
  const pct = Math.round(value * 100)
  const config = value >= 0.5 ? RISK_CONFIG.FLAGGED : value >= 0.25 ? RISK_CONFIG.REVIEW : RISK_CONFIG.SAFE
  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative w-36 h-36">
        <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
          <circle cx="60" cy="60" r="48" fill="none" stroke="#1a2a4a" strokeWidth="12" />
          <circle
            cx="60" cy="60" r="48" fill="none"
            stroke={config.dot}
            strokeWidth="12"
            strokeDasharray={`${pct * 3.016} 301.6`}
            strokeLinecap="round"
            style={{ transition: "stroke-dasharray 0.8s ease" }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-3xl font-bold ${config.color}`}>{pct}%</span>
          <span className="text-xs text-gray-500 mt-0.5">fraud risk</span>
        </div>
      </div>
      <span className={`text-sm font-bold px-3 py-1 rounded-full border ${config.bg} ${config.color}`}>
        {config.label}
      </span>
    </div>
  )
}

export default function Dashboard() {
  const [metrics, setMetrics]     = useState(null)
  const [features, setFeatures]   = useState([])
  const [summary, setSummary]     = useState(null)
  const [scenarios, setScenarios] = useState([])
  const [selected, setSelected]   = useState(null)
  const [liveResult, setLiveResult] = useState(null)
  const [liveLoading, setLiveLoading] = useState(false)

  useEffect(() => {
    fetch("/data/metrics.json").then(r => r.json()).then(setMetrics)
    fetch("/data/top_features.json").then(r => r.json()).then(setFeatures)
    fetch("/data/summary.json").then(r => r.json()).then(setSummary)
    fetch("/data/scenarios.json").then(r => r.json()).then(setScenarios)
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
      const data = await res.json()
      setLiveResult(data)
    } catch {
      setLiveResult({ error: "Could not reach live endpoint" })
    } finally {
      setLiveLoading(false)
    }
  }, [selected])

  if (!metrics || !summary) {
    return (
      <div className="min-h-screen flex items-center justify-center text-gray-400">
        Loading dashboard…
      </div>
    )
  }

  const pieData = [
    { name: `Fraud Caught — $${metrics.loss_prevented.toLocaleString()}`,   value: metrics.loss_prevented },
    { name: `Fraud Missed — $${metrics.missed_loss.toLocaleString()}`,      value: metrics.missed_loss },
  ]

  const top10 = features.slice(0, 10).map(f => ({
    ...f,
    shortName: f.feature.length > 10 ? f.feature : f.feature,
  }))

  return (
    <main className="min-h-screen px-6 py-10 max-w-6xl mx-auto">

      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className="mb-12 text-center">
        <p className="text-[#3498db] text-sm font-semibold uppercase tracking-widest mb-2">
          IEEE-CIS Credit Card Fraud Detection
        </p>
        <h1 className="text-4xl md:text-5xl font-bold text-white mb-3">
          Fraud Detection Model
        </h1>
        <p className="text-gray-400 max-w-xl mx-auto text-base">
          A machine learning model trained on real transaction data to identify
          fraudulent credit card purchases before money leaves the account.
        </p>
      </header>

      {/* ── Hero numbers ───────────────────────────────────────────── */}
      <Section title="At a Glance">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard
            label="Loss Prevented"
            value={`$${metrics.loss_prevented.toLocaleString()}`}
            sub="fraud dollars caught in test"
            color="text-[#2ecc71]"
          />
          <StatCard
            label="Fraud Caught"
            value={`${metrics.pct_prevented}%`}
            sub="of total fraud value stopped"
            color="text-[#2ecc71]"
          />
          <StatCard
            label="Model Accuracy Score"
            value={(metrics.roc_auc * 100).toFixed(1) + "%"}
            sub="ROC-AUC (100% = perfect)"
            color="text-[#3498db]"
          />
          <StatCard
            label="False Alarms"
            value={metrics.n_false_alarms}
            sub={`$${metrics.false_alarm_value.toLocaleString()} in legit blocks`}
            color="text-[#f39c12]"
          />
        </div>
      </Section>

      {/* ── How it works ───────────────────────────────────────────── */}
      <Section title="How It Works">
        <div className="grid md:grid-cols-3 gap-4">
          {[
            {
              step: "1",
              title: "Transaction Arrives",
              body: "When a credit card purchase is made, the system instantly reads details like the amount, card type, time of day, and spending history.",
            },
            {
              step: "2",
              title: "Model Scores It",
              body: "The model calculates a fraud probability score from 0–100%. Anything above a tuned threshold is flagged as suspicious.",
            },
            {
              step: "3",
              title: "Action Taken",
              body: "Flagged transactions are automatically declined or sent to a human analyst for review — stopping fraud before money leaves.",
            },
          ].map(({ step, title, body }) => (
            <div key={step} className="bg-[#16213e] rounded-2xl p-6 border border-[#0f3460]">
              <div className="w-8 h-8 rounded-full bg-[#3498db] flex items-center justify-center text-white font-bold text-sm mb-3">
                {step}
              </div>
              <h3 className="font-semibold text-white mb-2">{title}</h3>
              <p className="text-gray-400 text-sm leading-relaxed">{body}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* ── Live demo ─────────────────────────────────────────────── */}
      {scenarios.length > 0 && (
        <Section title="Live Deployments">
          <p className="text-gray-400 text-sm mb-5 leading-relaxed max-w-2xl">
            Select any transaction below to see how the model scores it. These
            are representative examples from the test dataset — transactions the
            model had never seen during training.
          </p>

          {/* Scenario cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 mb-8">
            {scenarios.map(s => {
              const cfg = RISK_CONFIG[s.risk_level] ?? RISK_CONFIG.REVIEW
              const isActive = selected?.id === s.id
              return (
                <button
                  key={s.id}
                  onClick={() => handleSelectScenario(s)}
                  className={`text-left rounded-2xl p-4 border transition-all duration-150 ${
                    isActive
                      ? `${cfg.bg} border-current`
                      : "bg-[#16213e] border-[#0f3460] hover:border-gray-500"
                  }`}
                >
                  <div className={`text-xs font-bold mb-2 ${cfg.color}`}>
                    {s.risk_level}
                  </div>
                  <div className="text-white text-sm font-semibold leading-snug mb-1">
                    {s.title}
                  </div>
                  <div className="text-[#3498db] text-lg font-bold">{s.amount}</div>
                  <div className="text-gray-500 text-xs mt-1">{s.time}</div>
                </button>
              )
            })}
          </div>

          {/* Result panel */}
          {selected && (
            <div className="bg-[#16213e] rounded-2xl p-6 border border-[#0f3460] animate-fade-in">
              <div className="grid md:grid-cols-3 gap-8 items-start">

                {/* Gauge */}
                <div className="flex justify-center">
                  <FraudGauge value={liveResult?.fraud_probability ?? selected.fraud_probability} />
                </div>

                {/* Key signals */}
                <div>
                  <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-3">
                    Why the model flagged this
                  </h3>
                  <ul className="space-y-2">
                    {selected.key_signals.map((sig, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-gray-300">
                        <span className="mt-1 text-[#3498db]">›</span>
                        {sig}
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Live score CTA */}
                <div className="flex flex-col gap-3">
                  <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
                    Live AWS Endpoint
                  </h3>
                  {selected.features ? (
                    <>
                      <p className="text-xs text-gray-500 leading-relaxed">
                        Send this transaction to the deployed SageMaker model for a
                        real-time score.
                      </p>
                      <button
                        onClick={handleLiveScore}
                        disabled={liveLoading}
                        className="px-4 py-2 rounded-xl bg-[#3498db] hover:bg-[#2980b9] text-white text-sm font-semibold disabled:opacity-50 transition-colors"
                      >
                        {liveLoading ? "Scoring…" : "Score via AWS →"}
                      </button>
                      {liveResult?.error && (
                        <p className="text-xs text-[#e74c3c]">{liveResult.error}</p>
                      )}
                      {liveResult && !liveResult.error && (
                        <p className="text-xs text-[#2ecc71]">
                          Live score: {(liveResult.fraud_probability * 100).toFixed(1)}%
                          {" "}({liveResult.risk_level})
                        </p>
                      )}
                    </>
                  ) : (
                    <p className="text-xs text-gray-500 leading-relaxed">
                      Run <code className="text-gray-300">sagemaker_deploy.py</code> in
                      JupyterLab to enable live scoring from this demo.
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}
        </Section>
      )}

      {/* ── Dollar impact ──────────────────────────────────────────── */}
      <Section title="Dollar Impact — Test Results">
        <div className="grid md:grid-cols-2 gap-8 items-center">
          <div>
            <p className="text-gray-400 text-sm mb-4 leading-relaxed">
              The chart shows how much fraud money the model caught versus missed
              on the held-out test sample — transactions it had never seen before.
            </p>
            <div className="space-y-3">
              {[
                { label: "Total fraud in test",    val: `$${metrics.total_fraud_value.toLocaleString()}`,  color: "bg-gray-500" },
                { label: "Loss prevented (caught)", val: `$${metrics.loss_prevented.toLocaleString()}`,     color: "bg-[#2ecc71]" },
                { label: "Loss missed (uncaught)",  val: `$${metrics.missed_loss.toLocaleString()}`,        color: "bg-[#e74c3c]" },
                { label: "False alarm value",       val: `$${metrics.false_alarm_value.toLocaleString()}`,  color: "bg-[#f39c12]" },
              ].map(({ label, val, color }) => (
                <div key={label} className="flex items-center justify-between bg-[#16213e] rounded-xl px-4 py-3 border border-[#0f3460]">
                  <div className="flex items-center gap-3">
                    <div className={`w-3 h-3 rounded-full ${color}`} />
                    <span className="text-sm text-gray-300">{label}</span>
                  </div>
                  <span className="font-semibold text-white text-sm">{val}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%" cy="50%"
                  innerRadius={60} outerRadius={100}
                  dataKey="value"
                  label={({ name, percent }) =>
                    `${(percent * 100).toFixed(0)}%`
                  }
                  labelLine={false}
                >
                  <Cell fill="#2ecc71" />
                  <Cell fill="#e74c3c" />
                </Pie>
                <Legend
                  formatter={v => (
                    <span className="text-xs text-gray-300">{v}</span>
                  )}
                />
                <Tooltip
                  formatter={v => [`$${Number(v).toLocaleString()}`, ""]}
                  contentStyle={{ background: "#16213e", border: "1px solid #0f3460", borderRadius: 8 }}
                  labelStyle={{ color: "#e0e0e0" }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </Section>

      {/* ── Top features ───────────────────────────────────────────── */}
      <Section title="What the Model Looks At — Top 10 Signals">
        <p className="text-gray-400 text-sm mb-5 leading-relaxed max-w-2xl">
          These are the transaction signals that most strongly influence the fraud
          score. Red bars push the score up (more suspicious); blue bars push it
          down (looks legitimate).
        </p>
        <div className="bg-[#16213e] rounded-2xl p-6 border border-[#0f3460]">
          <ResponsiveContainer width="100%" height={340}>
            <BarChart
              data={top10}
              layout="vertical"
              margin={{ left: 8, right: 30, top: 4, bottom: 4 }}
            >
              <XAxis type="number" tick={{ fill: "#9ca3af", fontSize: 11 }} />
              <YAxis
                type="category" dataKey="feature"
                tick={{ fill: "#d1d5db", fontSize: 11 }}
                width={72}
              />
              <Tooltip
                formatter={(v, _n, { payload }) => [
                  `Coefficient: ${v}`,
                  payload.description,
                ]}
                contentStyle={{ background: "#0f3460", border: "none", borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: "#e0e0e0", fontWeight: 600 }}
              />
              <Bar dataKey="coefficient" radius={[0, 4, 4, 0]}>
                {top10.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={entry.coefficient > 0 ? "#e74c3c" : "#3498db"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        {/* Feature legend table */}
        <div className="mt-5 space-y-2">
          {top10.map(f => (
            <div key={f.feature} className="flex items-start gap-3 text-sm">
              <span
                className="mt-0.5 w-2.5 h-2.5 rounded-full flex-shrink-0"
                style={{ background: f.coefficient > 0 ? "#e74c3c" : "#3498db", marginTop: 5 }}
              />
              <span className="text-gray-200 font-mono w-24 flex-shrink-0">{f.feature}</span>
              <span className="text-gray-400">{f.description}</span>
            </div>
          ))}
        </div>
      </Section>

      {/* ── Model details ──────────────────────────────────────────── */}
      <Section title="Model Details">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Algorithm"      value={summary.model_name.split(" ")[0]}      sub={summary.model_name} />
          <StatCard label="Features Used"  value={summary.n_features_used}               sub="out of 400+ raw columns" />
          <StatCard label="Training Size"  value={summary.n_train.toLocaleString()}      sub="transactions (after SMOTE)" />
          <StatCard label="CV Folds"       value={summary.cv_folds}                      sub={`CV AUC = ${(summary.cv_auc * 100).toFixed(1)}%`} color="text-[#3498db]" />
        </div>
      </Section>

      {/* ── Performance metrics ────────────────────────────────────── */}
      <Section title="Performance Metrics Explained">
        <div className="grid md:grid-cols-2 gap-4">
          {[
            {
              metric: "ROC-AUC",
              value: (metrics.roc_auc * 100).toFixed(1) + "%",
              explain: "How well the model separates fraud from legitimate transactions. 100% = perfect, 50% = random guessing.",
              color: "text-[#3498db]",
            },
            {
              metric: "Recall (Fraud Caught Rate)",
              value: (metrics.recall_fraud * 100).toFixed(1) + "%",
              explain: "Of all actual fraud transactions, this is the share the model successfully flagged.",
              color: "text-[#2ecc71]",
            },
            {
              metric: "Precision",
              value: (metrics.precision_fraud * 100).toFixed(1) + "%",
              explain: "Of all transactions the model flagged as fraud, this is the share that were actually fraud.",
              color: "text-[#f39c12]",
            },
            {
              metric: "Balanced Accuracy",
              value: (metrics.balanced_accuracy * 100).toFixed(1) + "%",
              explain: "Average accuracy across both fraud and legitimate classes — fair even with unequal class sizes.",
              color: "text-[#9b59b6]",
            },
          ].map(({ metric, value, explain, color }) => (
            <div key={metric} className="bg-[#16213e] rounded-2xl p-5 border border-[#0f3460]">
              <div className="flex justify-between items-start mb-2">
                <span className="text-sm text-gray-400">{metric}</span>
                <span className={`text-2xl font-bold ${color}`}>{value}</span>
              </div>
              <p className="text-xs text-gray-500 leading-relaxed">{explain}</p>
            </div>
          ))}
        </div>
      </Section>

      <footer className="text-center text-xs text-gray-600 pt-4 border-t border-[#0f3460]">
        IEEE-CIS Fraud Detection · {summary.dataset} · Model: {summary.model_name}
      </footer>
    </main>
  )
}
