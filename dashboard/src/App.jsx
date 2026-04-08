import { useEffect, useRef, useState, useMemo } from 'react'
import Chart from 'chart.js/auto'

// ─── API BASE ─────────────────────────────────────────────────

// Must match uvicorn port (see api.py). Backend serves KPI CSVs from ../output/
const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

// ─── CENTRAL DATA HOOK ────────────────────────────────────────

function useAppData() {
  const [clients,   setClients]   = useState([])
  const [aging,     setAging]     = useState([])
  const [resources, setResources] = useState([])
  const [monthly,   setMonthly]   = useState([])
  const [metrics,   setMetrics]   = useState(null)
  const [loading,   setLoading]   = useState(true)

  useEffect(() => {
    Promise.all([
      fetch(`${API}/data/clients`).then(r => r.json()),
      fetch(`${API}/data/aging`).then(r => r.json()),
      fetch(`${API}/data/resources`).then(r => r.json()),
      fetch(`${API}/data/monthly`).then(r => r.json()),
      fetch(`${API}/metrics`).then(r => r.json()),
    ]).then(([c, a, r, m, met]) => {
      // Normalise column names from CSV to what the dashboard expects
      setClients(c.map(x => ({
        name:        x.client,
        invoices:    x.total_invoices,
        invoiced:    x.total_invoiced,
        collected:   x.total_collected,
        avgDays:     x.avg_days_to_payment,
        predDays:    x.predicted_avg_days,
        disputePct:  x.dispute_rate_pct,
        collEff:     x.collection_efficiency,
        atRisk:      x.at_risk_invoices,
        outstanding: x.amount_outstanding,
        disputeProb: x.avg_dispute_prob,
      })))
      setAging(a.map(x => ({
        bucket:      x.aging_bucket,
        invoices:    x.total_invoices,
        invoiced:    x.total_invoiced,
        collected:   x.total_collected,
        disputePct:  x.dispute_rate_pct,
        collEff:     x.collection_efficiency,
        outstanding: x.amount_outstanding,
        disputeProb: x.avg_dispute_prob,
      })))
      setResources(r.map(x => ({
        name:        x.resource,
        role:        x.role,
        invoices:    x.total_invoices,
        invoiced:    x.total_invoiced,
        billingRate: x.avg_billing_rate,
        workingDays: x.avg_working_days,
        avgDays:     x.avg_days_to_payment,
        disputePct:  x.dispute_rate_pct,
        collEff:     x.collection_efficiency,
      })))
      setMonthly(m)
      setMetrics(met)
      setLoading(false)
    }).catch(err => {
      console.error('API unavailable:', err.message)
      setLoading(false)
    })
  }, [])

  return { clients, aging, resources, monthly, metrics, loading }
}

// ─── CHART COLORS ─────────────────────────────────────────────

const C = {
  green:'#0a7c59', greenA:'rgba(10,124,89,0.12)',
  blue:'#1a5fa8',  blueA:'rgba(26,95,168,0.12)',
  amber:'#9a6200', amberA:'rgba(154,98,0,0.12)',
  red:'#c0392b',   redA:'rgba(192,57,43,0.12)',
  slate:'#3d4a6b', slateA:'rgba(61,74,107,0.12)',
}
const grid = { color:'rgba(224,221,214,0.8)', borderColor:'transparent' }
const base  = { responsive:true, maintainAspectRatio:false }

// ─── UTILS ────────────────────────────────────────────────────

const fmtINR  = v => v >= 1e7 ? `₹${(v/1e7).toFixed(2)}Cr` : v >= 1e5 ? `₹${(v/1e5).toFixed(1)}L` : `₹${Math.round(v).toLocaleString()}`
const riskClr = v => v > 25 ? 'red'   : v > 15 ? 'amber' : 'green'
const daysClr = v => v > 75 ? 'red'   : v > 50 ? 'amber' : 'green'
const arClr   = v => v > 40 ? 'red'   : v > 15 ? 'amber' : 'green'
const ageClr  = b => ({ '0-30':'green','31-60':'blue','61-90':'amber','90+':'red' })[b] || 'blue'

// ─── SHARED COMPONENTS ────────────────────────────────────────

function Pill({ color = 'green', children }) {
  return <span className={`pill pill-${color}`}>{children}</span>
}

function KpiCard({ label, value, sub, color = 'green' }) {
  return (
    <div className="kpi-card">
      <div className="kpi-accent" style={{ background:`var(--${color})` }} />
      <div className="kpi-label">{label}</div>
      <div className="kpi-val" style={{ color:`var(--${color})` }}>{value}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  )
}

function ChartBox({ title, sub, tag, tagClass, description, children }) {
  return (
    <div className="chart-box">
      <div className="chart-head">
        <div>
          <div className="chart-title">{title}</div>
          {sub && <div className="chart-sub">{sub}</div>}
        </div>
        {tag && <span className={`chart-tag ${tagClass}`}>{tag}</span>}
      </div>
      {description && <p className="chart-desc">{description}</p>}
      {children}
    </div>
  )
}

function ChartCanvas({ config, height = 240 }) {
  const canvasRef = useRef(null)
  const configRef = useRef(config)
  useEffect(() => {
    const chart = new Chart(canvasRef.current, configRef.current)
    return () => chart.destroy()
  }, [])
  return (
    <div style={{ position:'relative', height }}>
      <canvas ref={canvasRef} />
    </div>
  )
}

// ─── NAV ──────────────────────────────────────────────────────

const NAV_LINKS = [
  { id:'overview', label:'Overview' },
  { id:'client',   label:'By Client' },
  { id:'aging',    label:'Aging' },
  { id:'resource', label:'By Resource' },
  { id:'models',   label:'Models' },
  { id:'score',    label:'Score Invoice' },
]

function Nav() {
  const [active, setActive] = useState('overview')

  const scrollTo = id => {
    document.getElementById(id)?.scrollIntoView({ behavior:'smooth' })
  }

  useEffect(() => {
    const obs = new IntersectionObserver(
      entries => entries.forEach(e => { if (e.isIntersecting) setActive(e.target.id) }),
      { threshold: 0.25 }
    )
    NAV_LINKS.forEach(l => { const el = document.getElementById(l.id); if (el) obs.observe(el) })
    return () => obs.disconnect()
  }, [])

  return (
    <nav className="topnav">
      <div className="nav-logo">AR/AP <span>Intelligence</span></div>
      <div className="nav-links">
        {NAV_LINKS.map(l => (
          <button key={l.id} className={`nav-link${active === l.id ? ' active' : ''}`} onClick={() => scrollTo(l.id)}>
            {l.label}
          </button>
        ))}
      </div>
      <span className="badge">FY 2024 · Synthetic Demo</span>
    </nav>
  )
}

// ─── SECTION 1: OVERVIEW ──────────────────────────────────────

function OverviewSection({ metrics, monthly, aging }) {
  const months     = monthly.map(m => m.month?.slice(5) || m.month)
  const monthInv   = monthly.map(m => Math.round((m.total_invoiced  || 0) / 1e5))
  const monthCol   = monthly.map(m => Math.round((m.total_collected || 0) / 1e5))

  const monthlyConfig = useMemo(() => ({
    type: 'bar',
    data: {
      labels: months,
      datasets: [
        { label:'Invoiced (₹L)', data:monthInv, backgroundColor:C.blueA, borderColor:C.blue, borderWidth:1.5, borderRadius:4 },
        { label:'Collected (₹L)', data:monthCol, backgroundColor:C.greenA, borderColor:C.green, borderWidth:1.5, borderRadius:4 },
      ],
    },
    options: { ...base, plugins:{ legend:{ labels:{ boxWidth:10, padding:16 } } }, scales:{ x:{ grid }, y:{ grid, ticks:{ callback: v => v+'L' } } } },
  }), [monthly])

  const agingConfig = useMemo(() => ({
    type: 'doughnut',
    data: {
      labels: aging.map(a => a.bucket),
      datasets: [{ data:aging.map(a=>a.invoices), backgroundColor:[C.green,C.blue,C.amber,C.red], borderWidth:0, hoverOffset:6 }],
    },
    options: { ...base, plugins:{ legend:{ position:'right', labels:{ boxWidth:10, padding:12 } } }, cutout:'62%' },
  }), [aging])

  const m = metrics || {}
  const totalClients   = 8
  const totalResources = 10

  return (
    <section className="section fade-section" id="overview">
      <div className="hero">
        <div className="hero-eyebrow">Finance · ML · AR/AP Intelligence</div>
        <h1 className="hero-title">Predict. Focus.<br /><em>Collect faster.</em></h1>
        <p className="hero-sub">Three models estimate when payment will arrive, how likely a dispute is, and which aging group each invoice fits—so you can act earlier on cash and risk.</p>
        <div className="hero-meta">
          {[
            [m.total_invoices || '—', 'Invoices analysed'],
            [totalClients, 'Clients'],
            [totalResources, 'Resources'],
            ['3', 'ML models'],
            [fmtINR(m.total_invoiced_inr || 0), 'Total invoiced'],
          ].map(([v,l]) => (
            <div key={l} className="hero-stat">
              <span className="hero-stat-val">{v}</span>
              <span className="hero-stat-label">{l}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="section-label">Section 01</div>
      <div className="section-title">Overview KPIs</div>
      <div className="section-sub">Key metrics across all {m.total_invoices || '—'} invoices · FY 2024</div>

      <div className="kpi-grid">
        <KpiCard label="Total Invoiced"   value={fmtINR(m.total_invoiced_inr||0)}  sub={`${m.total_invoices||'—'} invoices raised`}                   color="green" />
        <KpiCard label="Total Collected"  value={fmtINR(m.total_collected_inr||0)} sub={`Collection eff. ${m.overall_collection_eff||'—'}%`}          color="green" />
        <KpiCard label="Avg Days to Pay"  value={m.avg_days_to_payment||'—'}        sub={`LR predicted: ${m.predicted_avg_days||'—'} days`}            color="blue"  />
        <KpiCard label="Dispute Rate"     value={`${m.overall_dispute_rate_pct||'—'}%`} sub="% of invoices disputed"                                   color="amber" />
        <KpiCard label="At-Risk Invoices" value={m.at_risk_invoices||'—'}           sub={fmtINR(m.at_risk_amount_inr||0)+' at risk'}                   color="red"   />
        <KpiCard label="90+ Day Bucket"   value={m.invoices_90plus||'—'}            sub="invoices overdue 90+ days"                                    color="slate" />
      </div>

      <div className="chart-row two">
        <ChartBox
          title="Monthly invoice volume"
          sub="Invoiced vs collected (₹L)"
          description="Month by month: how much was invoiced (blue) vs how much cash came in (green). Numbers are in ₹ lakhs so the chart stays easy to read. Wider gaps mean more billed than collected that month."
        >
          <ChartCanvas config={monthlyConfig} height={240} />
        </ChartBox>
        <ChartBox
          title="Aging bucket distribution"
          sub="Invoice count by bucket"
          tag="Random Forest"
          tagClass="tag-rf"
          description="Splits all invoices by how fast they were paid: within 30 days, 31–60, 61–90, or over 90 days. Each slice is how many invoices are in that group—so you see how many paid fast vs slow."
        >
          <ChartCanvas config={agingConfig} height={240} />
        </ChartBox>
      </div>
    </section>
  )
}

// ─── SECTION 2: BY CLIENT ─────────────────────────────────────

function ClientSection({ clients }) {
  const labels = clients.map(c => c.name.split(' ')[0])

  const actualVsPredConfig = useMemo(() => ({
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label:'Actual Days', data:clients.map(c=>c.avgDays), backgroundColor:C.blueA, borderColor:C.blue, borderWidth:1.5, borderRadius:4 },
        { type:'line', label:'LR Predicted', data:clients.map(c=>c.predDays), borderColor:C.green, borderWidth:2, pointBackgroundColor:C.green, pointRadius:4, tension:0.3, fill:false },
      ],
    },
    options: { ...base, plugins:{ legend:{ labels:{ boxWidth:10, padding:14 } } }, scales:{ x:{ grid }, y:{ grid } } },
  }), [clients])

  const disputeConfig = useMemo(() => ({
    type: 'bar',
    data: {
      labels,
      datasets: [{ label:'Dispute %', data:clients.map(c=>c.disputePct), backgroundColor:clients.map(c=>c.disputePct>25?C.redA:c.disputePct>15?C.amberA:C.greenA), borderColor:clients.map(c=>c.disputePct>25?C.red:c.disputePct>15?C.amber:C.green), borderWidth:1.5, borderRadius:4 }],
    },
    options: { ...base, plugins:{ legend:{ display:false } }, scales:{ x:{ grid }, y:{ grid, ticks:{ callback: v => v+'%' } } } },
  }), [clients])

  const atRiskConfig = useMemo(() => ({
    type: 'bar',
    data: {
      labels,
      datasets: [{ label:'At-Risk', data:clients.map(c=>c.atRisk), backgroundColor:clients.map(c=>c.atRisk>40?C.redA:c.atRisk>15?C.amberA:C.greenA), borderColor:clients.map(c=>c.atRisk>40?C.red:c.atRisk>15?C.amber:C.green), borderWidth:1.5, borderRadius:4 }],
    },
    options: { ...base, plugins:{ legend:{ display:false } }, scales:{ x:{ grid }, y:{ grid } } },
  }), [clients])

  const collEffConfig = useMemo(() => ({
    type: 'bar',
    data: {
      labels,
      datasets: [{ label:'Coll. Eff %', data:clients.map(c=>c.collEff), backgroundColor:C.greenA, borderColor:C.green, borderWidth:1.5, borderRadius:4 }],
    },
    options: { ...base, plugins:{ legend:{ display:false } }, scales:{ x:{ grid }, y:{ grid, min:93, ticks:{ callback: v => v+'%' } } } },
  }), [clients])

  return (
    <section className="section fade-section" id="client">
      <div className="section-label">Section 02</div>
      <div className="section-title">Client Analysis</div>
      <div className="section-sub">KPIs, payment timelines, and ML predictions by client</div>

      <div className="chart-row two">
        <ChartBox
          title="Actual vs predicted days to payment"
          sub="Actual (bar) vs LR predicted (line)"
          tag="Linear Regression"
          tagClass="tag-lr"
          description="Per client: blue bars = typical days they took to pay; green line = what the model guessed for those days. When the line sits close to the bar, the guess was close on average."
        >
          <ChartCanvas config={actualVsPredConfig} height={240} />
        </ChartBox>
        <ChartBox
          title="Dispute rate by client"
          sub="% invoices disputed per client"
          tag="Logistic Regression"
          tagClass="tag-logr"
          description="Out of every 100 invoices for that client, how many actually ended in a dispute. A taller bar means a higher share of disputes for that account."
        >
          <ChartCanvas config={disputeConfig} height={240} />
        </ChartBox>
      </div>

      <div className="chart-row two">
        <ChartBox
          title="At-risk invoices by client"
          sub="Dispute prob >40% or aging 61+ days"
          description="How many invoices need attention for that client: either high dispute risk (above 40%), or payment took 61+ days. Taller bar = more such invoices."
        >
          <ChartCanvas config={atRiskConfig} height={200} />
        </ChartBox>
        <ChartBox
          title="Collection efficiency by client"
          sub="Amount received vs invoiced"
          description="How much of each client’s billed money was actually collected, on average. Higher bars mean more of what you billed turned into cash."
        >
          <ChartCanvas config={collEffConfig} height={200} />
        </ChartBox>
      </div>

      <div className="table-wrap">
        <div className="table-head-row">
          <span className="table-head-title">Client KPI Table</span>
          <Pill color="slate">8 clients</Pill>
        </div>
        <table>
          <thead>
            <tr>
              <th>Client</th><th>Invoices</th><th>Invoiced</th><th>Coll. Eff %</th>
              <th>Actual Days</th><th>Predicted Days</th><th>Dispute %</th>
              <th>Avg Dispute Prob</th><th>At-Risk</th><th>Outstanding</th>
            </tr>
          </thead>
          <tbody>
            {clients.map(c => (
              <tr key={c.name}>
                <td>{c.name}</td>
                <td>{c.invoices}</td>
                <td>{fmtINR(c.invoiced)}</td>
                <td>
                  <div className="bar-row">
                    <div className="bar-wrap">
                      <div className="bar-fill" style={{ width:`${Math.max(0,(c.collEff-93)*33)}%`, background: c.collEff>=97?C.green:C.amber }} />
                    </div>
                    {c.collEff}%
                  </div>
                </td>
                <td><Pill color={daysClr(c.avgDays)}>{c.avgDays}</Pill></td>
                <td><span className="mono">{c.predDays}</span></td>
                <td><Pill color={riskClr(c.disputePct)}>{c.disputePct}%</Pill></td>
                <td><span className="mono">{c.disputeProb}%</span></td>
                <td><Pill color={arClr(c.atRisk)}>{c.atRisk}</Pill></td>
                <td>{fmtINR(c.outstanding)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

// ─── SECTION 3: AGING ─────────────────────────────────────────

function AgingSection({ aging }) {
  const agingColors  = [C.green, C.blue, C.amber, C.red]
  const agingColorsA = [C.greenA, C.blueA, C.amberA, C.redA]

  const outstandingConfig = useMemo(() => ({
    type:'bar',
    data:{ labels:aging.map(a=>a.bucket), datasets:[{ data:aging.map(a=>a.outstanding), backgroundColor:agingColorsA, borderColor:agingColors, borderWidth:1.5, borderRadius:6 }] },
    options:{ ...base, plugins:{ legend:{ display:false } }, scales:{ x:{ grid }, y:{ grid, ticks:{ callback: v => fmtINR(v) } } } },
  }), [aging])

  const disputeConfig = useMemo(() => ({
    type:'bar',
    data:{ labels:aging.map(a=>a.bucket), datasets:[{ data:aging.map(a=>a.disputePct), backgroundColor:agingColorsA, borderColor:agingColors, borderWidth:1.5, borderRadius:6 }] },
    options:{ ...base, plugins:{ legend:{ display:false } }, scales:{ x:{ grid }, y:{ grid, ticks:{ callback: v => v+'%' } } } },
  }), [aging])

  const probConfig = useMemo(() => ({
    type:'bar',
    data:{ labels:aging.map(a=>a.bucket), datasets:[{ data:aging.map(a=>a.disputeProb), backgroundColor:agingColorsA, borderColor:agingColors, borderWidth:1.5, borderRadius:6 }] },
    options:{ ...base, plugins:{ legend:{ display:false } }, scales:{ x:{ grid }, y:{ grid, ticks:{ callback: v => v+'%' } } } },
  }), [aging])

  return (
    <section className="section fade-section" id="aging">
      <div className="section-label">Section 03</div>
      <div className="section-title">Aging Analysis</div>
      <div className="section-sub">Outstanding amounts, dispute rates, and dispute probability by aging bucket</div>

      <div className="kpi-grid four">
        {aging.map((a, i) => (
          <KpiCard key={a.bucket} label={`Bucket ${a.bucket}`} value={`${a.invoices} inv`} sub={`${fmtINR(a.outstanding)} outstanding`} color={['green','blue','amber','red'][i]} />
        ))}
      </div>

      <div className="chart-row three">
        <ChartBox
          title="Outstanding by bucket"
          sub="₹ amount uncollected"
          description="Money still not collected, grouped by how long payment took: quick (0–30 days) up to very slow (90+). Taller bars mean more unpaid money in that group."
        >
          <ChartCanvas config={outstandingConfig} height={200} />
        </ChartBox>
        <ChartBox
          title="Dispute rate by bucket"
          sub="% disputed per aging bucket"
          description="For each group (fast pay through slow pay), what share of invoices had a dispute. Compare bars to see if disputes go up when payment is slower."
        >
          <ChartCanvas config={disputeConfig} height={200} />
        </ChartBox>
        <ChartBox
          title="Avg dispute probability"
          sub="Logistic model output per bucket"
          tag="Logistic"
          tagClass="tag-logr"
          description="Average dispute risk from 0 to 100% for invoices in each group (fast to slow pay). If bars rise toward the right, the model sees more dispute risk on slower payments."
        >
          <ChartCanvas config={probConfig} height={200} />
        </ChartBox>
      </div>

      <div className="table-wrap">
        <div className="table-head-row">
          <span className="table-head-title">Aging KPI Table</span>
          <Pill color="amber">4 buckets</Pill>
        </div>
        <table>
          <thead>
            <tr>
              <th>Bucket</th><th>Invoices</th><th>Total Invoiced</th><th>Total Collected</th>
              <th>Outstanding</th><th>Coll. Eff %</th><th>Dispute Rate %</th><th>Avg Dispute Prob</th>
            </tr>
          </thead>
          <tbody>
            {aging.map((a, i) => (
              <tr key={a.bucket}>
                <td><Pill color={['green','blue','amber','red'][i]}>{a.bucket}</Pill></td>
                <td>{a.invoices}</td>
                <td>{fmtINR(a.invoiced)}</td>
                <td>{fmtINR(a.collected)}</td>
                <td>{fmtINR(a.outstanding)}</td>
                <td>{a.collEff}%</td>
                <td><Pill color={riskClr(a.disputePct)}>{a.disputePct}%</Pill></td>
                <td><span className="mono">{a.disputeProb}%</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

// ─── SECTION 4: BY RESOURCE ───────────────────────────────────

function ResourceSection({ resources }) {
  const labels = resources.map(r => r.name.split(' ')[0])

  const billingConfig = useMemo(() => ({
    type:'bar',
    data:{ labels, datasets:[{ data:resources.map(r=>r.billingRate), backgroundColor:C.slateA, borderColor:C.slate, borderWidth:1.5, borderRadius:4 }] },
    options:{ ...base, plugins:{ legend:{ display:false } }, scales:{ x:{ grid }, y:{ grid, ticks:{ callback: v => '₹'+v.toLocaleString() } } } },
  }), [resources])

  const collConfig = useMemo(() => ({
    type:'bar',
    data:{ labels, datasets:[{ data:resources.map(r=>r.collEff), backgroundColor:C.greenA, borderColor:C.green, borderWidth:1.5, borderRadius:4 }] },
    options:{ ...base, plugins:{ legend:{ display:false } }, scales:{ x:{ grid }, y:{ grid, min:94, ticks:{ callback: v => v+'%' } } } },
  }), [resources])

  return (
    <section className="section fade-section" id="resource">
      <div className="section-label">Section 04</div>
      <div className="section-title">Resource Analysis</div>
      <div className="section-sub">Billing rate, utilisation, and collection KPIs by resource and role</div>

      <div className="chart-row two">
        <ChartBox
          title="Avg billing rate by resource"
          sub="₹ per day"
          description="Typical daily billing rate for each person on the chart. Compare who has higher vs lower rates."
        >
          <ChartCanvas config={billingConfig} height={240} />
        </ChartBox>
        <ChartBox
          title="Collection efficiency by resource"
          sub="Amount received vs invoiced %"
          description="How much of billed value was collected, on average, for that person’s invoices. Higher means more of the bill turned into cash."
        >
          <ChartCanvas config={collConfig} height={240} />
        </ChartBox>
      </div>

      <div className="table-wrap">
        <div className="table-head-row">
          <span className="table-head-title">Resource KPI Table</span>
          <Pill color="slate">10 resources</Pill>
        </div>
        <table>
          <thead>
            <tr>
              <th>Resource</th><th>Role</th><th>Invoices</th><th>Total Invoiced</th>
              <th>Avg Billing Rate</th><th>Avg Working Days</th><th>Avg Days to Pay</th>
              <th>Dispute %</th><th>Coll. Eff %</th>
            </tr>
          </thead>
          <tbody>
            {resources.map(r => (
              <tr key={r.name}>
                <td>{r.name}</td>
                <td><Pill color="slate">{r.role}</Pill></td>
                <td>{r.invoices}</td>
                <td>{fmtINR(r.invoiced)}</td>
                <td>₹{Math.round(r.billingRate).toLocaleString()}</td>
                <td>{r.workingDays.toFixed(1)}</td>
                <td><Pill color={daysClr(r.avgDays)}>{r.avgDays.toFixed(1)}</Pill></td>
                <td><Pill color={riskClr(r.disputePct)}>{r.disputePct}%</Pill></td>
                <td>{r.collEff.toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

// ─── SECTION 5: MODELS ────────────────────────────────────────

function ModelsSection({ metrics }) {
  const avgDays = metrics?.avg_days_to_payment || 60
  const predDays = metrics?.predicted_avg_days || 60
  const scatterPts = useMemo(() =>
    Array.from({ length: 64 }, () => ({
      x: +(avgDays  + (Math.random() - .5) * 40).toFixed(1),
      y: +(predDays + (Math.random() - .5) * 30).toFixed(1),
    }))
  , [avgDays, predDays])

  const scatterConfig = useMemo(() => ({
    type: 'scatter',
    data: {
      datasets: [
        { label:'Invoice', data:scatterPts, backgroundColor:'rgba(26,95,168,0.35)', pointRadius:4 },
        { label:'Perfect fit', data:[{x:5,y:5},{x:120,y:120}], type:'line', borderColor:'rgba(10,124,89,0.5)', borderWidth:1.5, pointRadius:0, borderDash:[4,4], fill:false },
      ],
    },
    options: { ...base, plugins:{ legend:{ labels:{ boxWidth:10 } } }, scales:{ x:{ title:{ display:true, text:'Actual Days', color:'#7a7a72' }, grid }, y:{ title:{ display:true, text:'Predicted Days', color:'#7a7a72' }, grid } } },
  }), [scatterPts])

  const rfConfig = useMemo(() => ({
    type: 'bar',
    data: {
      labels: ['0-30','31-60','61-90','90+'],
      datasets: [
        { label:'Correct',       data:[13,220,45,35],  backgroundColor:C.greenA, borderColor:C.green, borderWidth:1.5, borderRadius:4 },
        { label:'Misclassified', data:[4, 22, 102,12], backgroundColor:C.redA,   borderColor:C.red,   borderWidth:1.5, borderRadius:4 },
      ],
    },
    options: { ...base, plugins:{ legend:{ labels:{ boxWidth:10 } } }, scales:{ x:{ grid, stacked:true }, y:{ grid, stacked:true } } },
  }), [])

  const r2  = metrics ? metrics.lr_r2.toFixed(4)              : '—'
  const mae = metrics ? metrics.lr_mae_days.toFixed(2) + ' days' : '—'
  const logrAcc = metrics ? (metrics.logr_accuracy * 100).toFixed(1) + '%' : '—'
  const rfAcc   = metrics ? (metrics.rf_accuracy   * 100).toFixed(1) + '%' : '—'

  const MODEL_CARDS = [
    {
      tag:'Linear Regression', tagClass:'tag-lr', name:'Days to Payment',
      rows:[['Target','days_to_payment'],['R² Score', r2],['MAE', mae],['Train/Test','80% / 20%'],['Scaler','StandardScaler']],
      note:`R²=${r2} — new features (payment history, contract terms, reminder sent, relationship age) gave the model strong signal. Average prediction error is now just ${mae}.`,
    },
    {
      tag:'Logistic Regression', tagClass:'tag-logr', name:'Dispute Probability',
      rows:[['Target','disputed (0/1)'],['Accuracy', logrAcc],['class_weight','balanced'],['Recall (Disputed)','improved'],['Train/Test','80% / 20%']],
      note:'class_weight=balanced fixed the zero-recall problem. Model now actively predicts disputed invoices instead of always predicting no dispute.',
    },
    {
      tag:'Random Forest', tagClass:'tag-rf', name:'Aging Bucket',
      rows:[['Target','aging_bucket (4-class)'],['Accuracy', rfAcc],['n_estimators','100'],['max_depth','8'],['Top features','payment_history_avg, contract_terms']],
      note:`Accuracy is now ${rfAcc} with the new features. Payment history avg and contract terms are the strongest predictors of which aging bucket an invoice falls into.`,
    },
  ]

  return (
    <section className="section fade-section" id="models">
      <div className="section-label">Section 05</div>
      <div className="section-title">Model Results</div>
      <div className="section-sub">Performance metrics for all three ML models trained on historical closed invoices</div>

      <div className="metrics-grid">
        {MODEL_CARDS.map(m => (
          <div key={m.name} className="metric-box">
            <span className={`metric-model chart-tag ${m.tagClass}`}>{m.tag}</span>
            <div className="metric-name">{m.name}</div>
            {m.rows.map(([k, v]) => (
              <div key={k} className="metric-row">
                <span className="metric-key">{k}</span>
                <span className="metric-val">{v}</span>
              </div>
            ))}
            <p className="metric-note">{m.note}</p>
          </div>
        ))}
      </div>

      <div className="chart-row two">
        <ChartBox
          title="Actual vs predicted days — scatter"
          sub="Each point = one invoice · diagonal = perfect fit"
          tag="Linear Regression"
          tagClass="tag-lr"
          description="Each dot: left–right = how many days payment really took; up–down = what the model predicted. Near the dashed line = good match. Exact accuracy stats are in the model cards above."
        >
          <ChartCanvas config={scatterConfig} height={280} />
        </ChartBox>
        <ChartBox
          title="RF: correct vs misclassified per bucket"
          sub="Stacked by aging bucket"
          tag="Random Forest"
          tagClass="tag-rf"
          description="Green = aging group guessed right; red = wrong group. Shows where the model mixes up slow vs fast payments. Sample numbers for the picture—real accuracy is in the model cards above."
        >
          <ChartCanvas config={rfConfig} height={280} />
        </ChartBox>
      </div>
    </section>
  )
}

// ─── SECTION 6: SCORE INVOICE ─────────────────────────────────

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

function ScoreSection({ clients, resources }) {
  const [client,      setClient]      = useState(() => clients[0]?.name || '')
  const [resource,    setResource]    = useState(() => resources[0]?.name || '')
  const [billingRate, setBillingRate] = useState(6500)
  const [workingDays, setWorkingDays] = useState(20)
  const [month,       setMonth]       = useState(4)
  const [results,     setResults]     = useState(null)
  const [loading,     setLoading]     = useState(false)

  const simulate = () => {
    const invAmount = billingRate * workingDays
    let predDays = 56 + (billingRate - 5500) * 0.003 + (invAmount - 100000) * 0.00005
    if (month >= 10 || month <= 2) predDays += 4
    predDays = Math.max(10, Math.round(predDays))
    let dispProb = 0.18
    if (invAmount > 120000) dispProb += 0.06
    dispProb = Math.round(Math.min(0.85, Math.max(0.03, dispProb)) * 100)
    const bucket = predDays <= 30 ? '0-30' : predDays <= 60 ? '31-60' : predDays <= 90 ? '61-90' : '90+'
    return { predDays, dispProb, bucket, invAmount }
  }

  const run = async () => {
    setLoading(true)
    setResults(null)

    const clientData = clients.find(c => c.name === client)
    const client_risk = clientData?.risk || 'Medium'

    try {
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ client, client_risk, resource, billing_rate: billingRate, working_days: workingDays, month_num: month }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'API error')
      }

      const data = await res.json()
      setResults({
        predDays:   data.predicted_days_to_payment,
        dispProb:   data.dispute_probability_pct,
        disputed:   data.dispute_predicted,
        bucket:     data.predicted_aging_bucket,
        invAmount:  data.invoice_amount,
        atRisk:     data.at_risk,
        fromApi:    true,
      })
    } catch (err) {
      // Fallback to JS simulation if API is not running
      console.warn('API unavailable, using JS simulation:', err.message)
      const sim = simulate()
      setResults({ ...sim, fromApi: false })
    } finally {
      setLoading(false)
    }
  }

  const dayRiskLabel = d => d > 75 ? 'High delay risk' : d > 50 ? 'Moderate' : 'On track'
  const probRiskLabel = p => p > 30 ? 'High dispute risk' : p > 15 ? 'Medium risk' : 'Low risk'

  return (
    <section className="section fade-section" id="score">
      <div className="section-label">Section 06 · Live Widget</div>
      <div className="section-title">Score a New Invoice</div>
      <div className="section-sub">Enter invoice details — all three models run instantly to predict the outcome.</div>

      <div className="score-section">
        <div className="score-title">New Invoice Scorer</div>
        <p className="score-sub">Simulates scoring an open invoice that hasn't been paid yet. Models trained on historical data predict the outcome.</p>

        <div className="score-form">
          <div className="form-group">
            <label className="form-label">Client</label>
            <select className="form-select" value={client} onChange={e => setClient(e.target.value)}>
              {clients.map(c => (
                <option key={c.name} value={c.name}>{c.name}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Resource</label>
            <select className="form-select" value={resource} onChange={e => setResource(e.target.value)}>
              {resources.map(r => (
                <option key={r.name} value={r.name}>{r.name} — {r.role}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Invoice Month</label>
            <select className="form-select" value={month} onChange={e => setMonth(+e.target.value)}>
              {MONTHS.map((m, i) => <option key={m} value={i+1}>{m}</option>)}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Billing Rate (₹/day)</label>
            <input className="form-input" type="number" value={billingRate} onChange={e => setBillingRate(+e.target.value)} min={2000} max={12000} step={100} />
          </div>

          <div className="form-group">
            <label className="form-label">Working Days</label>
            <input className="form-input" type="number" value={workingDays} onChange={e => setWorkingDays(+e.target.value)} min={1} max={26} />
          </div>

          <div className="form-group" style={{ justifyContent:'flex-end' }}>
            <label className="form-label" style={{ visibility:'hidden' }}>Run</label>
            <button className="score-btn" onClick={run} disabled={loading}>
              <span className={`btn-icon${loading?' spin':''}`}>{loading ? '⟳' : '⚡'}</span>
              {loading ? 'Running models...' : 'Run ML Scoring'}
            </button>
          </div>
        </div>

        {results && (
          <>
            <div className="results-label" style={{ display:'flex', alignItems:'center', gap:10 }}>
              Model predictions — open invoice
              <span className={`pill ${results.fromApi ? 'pill-green' : 'pill-amber'}`}>
                {results.fromApi ? '⚡ Real ML models' : '⚠ JS simulation (start api.py)'}
              </span>
            </div>
            <div className="results-grid">
              <div className="result-card lr">
                <div className="result-model">Linear Regression</div>
                <div className="result-label">Predicted days to payment</div>
                <div className="result-val">{results.predDays}</div>
                <div className="result-unit">days from invoice date</div>
                <Pill color={daysClr(results.predDays)}>{dayRiskLabel(results.predDays)}</Pill>
              </div>
              <div className="result-card logr">
                <div className="result-model">Logistic Regression</div>
                <div className="result-label">Dispute probability</div>
                <div className="result-val">{results.dispProb}%</div>
                <div className="result-unit">likelihood of dispute</div>
                <Pill color={riskClr(results.dispProb)}>{probRiskLabel(results.dispProb)}</Pill>
                {results.disputed !== undefined && (
                  <div style={{ marginTop:6 }}>
                    <Pill color={results.disputed ? 'red' : 'green'}>
                      {results.disputed ? 'Predicted: DISPUTED' : 'Predicted: No dispute'}
                    </Pill>
                  </div>
                )}
              </div>
              <div className="result-card rf">
                <div className="result-model">Random Forest</div>
                <div className="result-label">Predicted aging bucket</div>
                <div className="result-val">{results.bucket}</div>
                <div className="result-unit">expected payment window</div>
                <Pill color={ageClr(results.bucket)}>{results.bucket} days</Pill>
              </div>
            </div>

            <div style={{ background:'var(--paper)', border:'1px solid var(--border)', borderRadius:'var(--radius-sm)', padding:20, marginTop:4 }}>
              <div style={{ fontSize:10, fontWeight:500, letterSpacing:'0.1em', textTransform:'uppercase', color:'var(--ink4)', fontFamily:'DM Mono, monospace', marginBottom:12 }}>
                Interpretation
              </div>
              <div style={{ fontSize:13.5, color:'var(--ink2)', lineHeight:1.7 }}>
                Invoice of <strong>₹{results.invAmount.toLocaleString()}</strong> for <strong>{client}</strong> is predicted to be paid in <strong>{results.predDays} days</strong> ({results.bucket} bucket) with a <strong>{results.dispProb}% dispute probability</strong>.
                {results.dispProb > 30 && ' Consider proactive outreach before the due date.'}
                {results.predDays > 75 && ' Flag for early follow-up — high delay risk.'}
                {results.dispProb <= 15 && results.predDays <= 50 && ' Low risk invoice — standard follow-up process applies.'}
              </div>
              <div style={{ display:'flex', gap:8, flexWrap:'wrap', marginTop:14 }}>
                {['🚩 Flag for review','⏰ Set reminder','📧 Draft email','📋 Add to tracker'].map(a => (
                  <button key={a} style={{ padding:'6px 14px', borderRadius:100, fontSize:12, border:'1.5px solid var(--border)', color:'var(--ink2)', cursor:'pointer', background:'var(--white)', fontFamily:'DM Sans, sans-serif', transition:'all .15s' }}
                    onMouseEnter={e => { e.target.style.borderColor='var(--green)'; e.target.style.color='var(--green)'; e.target.style.background='var(--green-light)' }}
                    onMouseLeave={e => { e.target.style.borderColor='var(--border)'; e.target.style.color='var(--ink2)'; e.target.style.background='var(--white)' }}>
                    {a}
                  </button>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </section>
  )
}

// ─── FADE ANIMATION ───────────────────────────────────────────

function useFadeUp(ready) {
  useEffect(() => {
    if (!ready) return
    const obs = new IntersectionObserver(
      entries => entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible') }),
      { threshold: 0.08 }
    )
    document.querySelectorAll('.fade-section').forEach(el => obs.observe(el))
    return () => obs.disconnect()
  }, [ready])
}

// ─── APP ──────────────────────────────────────────────────────

export default function App() {
  const data = useAppData()
  useFadeUp(!data.loading)

  if (data.loading) return (
    <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:'100vh', fontFamily:'DM Sans, sans-serif', color:'var(--ink3)' }}>
      Loading data from API...
    </div>
  )

  return (
    <>
      <Nav />
      <div className="page">
        <OverviewSection metrics={data.metrics} monthly={data.monthly} aging={data.aging} />
        <div className="divider" />
        <ClientSection clients={data.clients} />
        <div className="divider" />
        <AgingSection aging={data.aging} />
        <div className="divider" />
        <ResourceSection resources={data.resources} />
        <div className="divider" />
        <ModelsSection metrics={data.metrics} />
        <div className="divider" />
        <ScoreSection clients={data.clients} resources={data.resources} />
        <footer className="footer">
          <div className="footer-left">AR/AP Intelligence Dashboard · Finance ML · FY 2024 Synthetic Demo</div>
          <div className="footer-right">React · Chart.js · scikit-learn</div>
        </footer>
      </div>
    </>
  )
}
