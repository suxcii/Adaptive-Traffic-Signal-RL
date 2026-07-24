import { useEffect, useState } from "react";
import { Bar, BarChart, Cell, ResponsiveContainer, XAxis, YAxis } from "recharts";

// ── Backend ─────────────────────────────────────────────────────────────
// Point this at wherever `uvicorn api:app` is running.
const API_BASE = "http://127.0.0.1:8000";

type QValueEntry = { lane: string; movement: string; value: number };

type AgentDecision = {
  type: "rule" | "q-table" | "dqn";
  source: string;
  rule?: string;
  stateBucket?: number[];
  seenBucketBefore?: boolean;
  chosenIndex: number;
  values: QValueEntry[];
};

type LaneDetail = {
  lane: string;
  movement: string;
  before: number;
  arrived: number;
  passed: number;
  blocked: number;
  after: number;
  wasSelected: boolean;
};

type OutgoingDetail = {
  direction: "N" | "S" | "E" | "W";
  before: number;
  in: number;
  out: number;
  after: number;
};

type StepDetails = {
  agentDecision: AgentDecision;
  selectedLaneBreakdown: {
    lane: string;
    before: number;
    capacity: number;
    passed: number;
    blocked: number;
    after: number;
  };
  lanes: LaneDetail[];
  outgoing: OutgoingDetail[];
  blockedArrivals: number;
  carsLeftSystem: number;
};

type Snapshot = {
  episode: number;
  step: number;
  algorithm: string;
  selectedLane: string | null;
  exitDirection: string | null;
  reward: number;
  carsPassed: number;
  totalQueue: number;
  incoming: Record<string, number>;
  outgoing: Record<string, number>;
  done: boolean;
  details: StepDetails | null;
};

const ALGORITHM_OPTIONS = ["Baseline", "Q-learning", "SARSA", "DQN"];

type WaitingComparisonEntry = {
  available: boolean;
  avgWaiting: number | null;
  episodes: number;
};
type WaitingComparison = Record<string, WaitingComparisonEntry>;

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

// ── Palette (Apple Maps-inspired) ──────────────────────────────────────────
const ROAD = "#6A6A6A";
const ROAD_DARK = "#515151";
const GROUND = "#DDD3BE";
const GRASS = "#87AC60";
const WHITE = "#FFFFFF";
const YELLOW = "#F3C028";
const INC_TINT = "rgba(45,105,215,0.20)";
const OUT_TINT = "rgba(230,90,20,0.20)";
const ACTIVE_INC = "rgba(62,203,110,0.65)";
const ACTIVE_OUT = "rgba(62,203,110,0.5)";

// ── SVG geometry ───────────────────────────────────────────────────────────
const SZ = 560;
const CX = 280;
const CY = 280;
const LW = 20; // lane width px
const RW = 4 * LW; // 80px – road width (3 inc + 1 out per arm)
const X1 = CX - RW / 2; // 240
const X2 = CX + RW / 2; // 320
const Y1 = CY - RW / 2; // 240
const Y2 = CY + RW / 2; // 320

// ── Arrow components ─────────────────────────────────────────────────────
// (apex sits on the pointing side for each direction)
function ArrDown({ x, y }: { x: number; y: number }) {
  const s = 5;
  return <polygon points={`${x},${y + s} ${x - s * 0.7},${y - s * 0.5} ${x + s * 0.7},${y - s * 0.5}`} fill={WHITE} opacity={0.8} />;
}
function ArrUp({ x, y }: { x: number; y: number }) {
  const s = 5;
  return <polygon points={`${x},${y - s} ${x - s * 0.7},${y + s * 0.5} ${x + s * 0.7},${y + s * 0.5}`} fill={WHITE} opacity={0.8} />;
}
function ArrLeft({ x, y }: { x: number; y: number }) {
  const s = 5;
  return <polygon points={`${x + s},${y - s * 0.7} ${x - s * 0.5},${y} ${x + s},${y + s * 0.7}`} fill={WHITE} opacity={0.8} />;
}
function ArrRight({ x, y }: { x: number; y: number }) {
  const s = 5;
  return <polygon points={`${x - s},${y - s * 0.7} ${x + s * 0.5},${y} ${x - s},${y + s * 0.7}`} fill={WHITE} opacity={0.8} />;
}

type ArrDir = "down" | "up" | "left" | "right";

function laneArrows(dir: ArrDir, fixed: number, from: number, to: number) {
  const step = 52;
  const start = from + 26;
  const end = to - 10;
  const items: JSX.Element[] = [];
  for (let p = start; p < end; p += step) {
    if (dir === "down") items.push(<ArrDown key={p} x={fixed} y={p} />);
    else if (dir === "up") items.push(<ArrUp key={p} x={fixed} y={p} />);
    else if (dir === "left") items.push(<ArrLeft key={p} x={p} y={fixed} />);
    else items.push(<ArrRight key={p} x={p} y={fixed} />);
  }
  return items;
}

// ── Tree (layered circles, Maps-style) ─────────────────────────────────────
function Tree({ cx, cy, r }: { cx: number; cy: number; r: number }) {
  return (
    <>
      <circle cx={cx} cy={cy} r={r} fill="#6A9D46" />
      <circle cx={cx - r * 0.22} cy={cy - r * 0.22} r={r * 0.62} fill="#82B85A" />
    </>
  );
}

// ── Crosswalk ──────────────────────────────────────────────────────────────
function Crosswalk({ x, y, w, h, vertical }: { x: number; y: number; w: number; h: number; vertical: boolean }) {
  const items: JSX.Element[] = [];
  const sw = 5;
  const gap = 4;
  if (vertical) {
    for (let p = x; p < x + w; p += sw + gap) {
      items.push(<rect key={p} x={p} y={y} width={sw} height={h} fill={WHITE} opacity={0.82} />);
    }
  } else {
    for (let p = y; p < y + h; p += sw + gap) {
      items.push(<rect key={p} x={x} y={p} width={w} height={sw} fill={WHITE} opacity={0.82} />);
    }
  }
  return <>{items}</>;
}

// ── Trees layout ───────────────────────────────────────────────────────────
const TREES: [number, number, number][] = [
  [52, 52, 21], [98, 82, 15], [162, 44, 18], [188, 112, 13],
  [36, 138, 17], [118, 158, 11], [72, 202, 14], [202, 182, 10],
  [SZ - 52, 52, 21], [SZ - 98, 82, 15], [SZ - 162, 44, 18], [SZ - 188, 112, 13],
  [SZ - 36, 138, 17], [SZ - 118, 158, 11], [SZ - 72, 202, 14], [SZ - 202, 182, 10],
  [52, SZ - 52, 21], [98, SZ - 82, 15], [162, SZ - 44, 18], [188, SZ - 112, 13],
  [36, SZ - 138, 17], [118, SZ - 158, 11], [72, SZ - 202, 14], [202, SZ - 182, 10],
  [SZ - 52, SZ - 52, 21], [SZ - 98, SZ - 82, 15], [SZ - 162, SZ - 44, 18], [SZ - 188, SZ - 112, 13],
  [SZ - 36, SZ - 138, 17], [SZ - 118, SZ - 158, 11], [SZ - 72, SZ - 202, 14], [SZ - 202, SZ - 182, 10],
];

// ── Queue layout ─────────────────────────────────────────────────────────
// Incoming lane ids in physical screen order for each arm (matches how the
// backend's MOVEMENT_TO_LANE assigns curb/median/middle lanes).
const NORTH_LANES = ["N1", "N2", "N3"];
const SOUTH_LANES = ["S1", "S2", "S3"];
const WEST_LANES = ["W1", "W2", "W3"];
const EAST_LANES = ["E1", "E2", "E3"];

const OUT_QUEUES: { id: string; dir: "N" | "S" | "E" | "W"; label: string }[] = [
  { id: "N↑", dir: "N", label: "North exit" },
  { id: "S↓", dir: "S", label: "South exit" },
  { id: "E→", dir: "E", label: "East exit" },
  { id: "W←", dir: "W", label: "West exit" },
];

// ── Details panel ───────────────────────────────────────────────────────────
// Explains one step: why that lane got the green light, and exactly how
// every queue moved (arrivals in, cars passed, cars blocked).
function DetailsPanel({ details, algorithm }: { details: StepDetails; algorithm: string }) {
  const { agentDecision, selectedLaneBreakdown: sel, lanes, outgoing } = details;

  const chartData = agentDecision.values.map((v, i) => ({
    ...v,
    chosen: i === agentDecision.chosenIndex,
  }));

  return (
    <div
      className="rounded-xl p-3 flex flex-col gap-3"
      style={{ background: "#141922", border: "1px solid rgba(255,255,255,0.08)", width: "100%", maxWidth: 748 }}
    >
      {/* Agent decision */}
      <div>
        <div className="flex items-center gap-2 mb-1.5">
          <span className="w-2 h-2 rounded-full inline-block" style={{ background: "#7F77DD" }} />
          <span className="text-[11px] font-bold text-white tracking-[0.12em] uppercase">Agent decision</span>
          <span className="text-[9px]" style={{ color: "rgba(255,255,255,0.4)", fontFamily: "'JetBrains Mono', monospace" }}>
            {agentDecision.source}
          </span>
        </div>

        {agentDecision.type === "rule" ? (
          <p className="text-[10px] mb-2" style={{ color: "rgba(255,255,255,0.6)" }}>
            Rule: {agentDecision.rule} — chose{" "}
            <span style={{ color: "#B9FFD1", fontFamily: "'JetBrains Mono', monospace" }}>
              {agentDecision.values[agentDecision.chosenIndex]?.lane}
            </span>{" "}
            ({agentDecision.values[agentDecision.chosenIndex]?.value} cars waiting, the largest queue)
          </p>
        ) : (
          <p className="text-[10px] mb-2" style={{ color: "rgba(255,255,255,0.6)" }}>
            {agentDecision.type === "q-table" && agentDecision.stateBucket && (
              <>
                State bucket: <span style={{ fontFamily: "'JetBrains Mono', monospace" }}>[{agentDecision.stateBucket.join(", ")}]</span>
                {agentDecision.seenBucketBefore === false && (
                  <span style={{ color: "#FFA76B" }}> — unseen during training, falling back to zeros</span>
                )}
                {" — "}
              </>
            )}
            Highest value:{" "}
            <span style={{ color: "#B9FFD1", fontFamily: "'JetBrains Mono', monospace" }}>
              {agentDecision.values[agentDecision.chosenIndex]?.lane}
            </span>{" "}
            ({agentDecision.values[agentDecision.chosenIndex]?.value})
          </p>
        )}

        <div style={{ width: "100%", height: 220 }}>
          <ResponsiveContainer>
            <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 12, bottom: 0, left: 0 }}>
              <XAxis type="number" tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 9 }} axisLine={false} tickLine={false} />
              <YAxis
                type="category"
                dataKey="lane"
                width={30}
                tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 9, fontFamily: "'JetBrains Mono', monospace" }}
                axisLine={false}
                tickLine={false}
              />
              <Bar dataKey="value" radius={2}>
                {chartData.map((d) => (
                  <Cell key={d.lane} fill={d.chosen ? "#3ECB6E" : "rgba(127,119,221,0.55)"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Selected lane breakdown */}
      <div>
        <span className="text-[11px] font-bold text-white tracking-[0.12em] uppercase block mb-1.5">
          Selected lane ({sel.lane})
        </span>
        <div className="grid grid-cols-5 gap-1 text-center">
          {[
            ["Before", sel.before],
            ["Capacity", sel.capacity],
            ["Passed", sel.passed],
            ["Blocked", sel.blocked],
            ["After", sel.after],
          ].map(([k, v]) => (
            <div key={k as string} className="rounded py-1" style={{ background: "rgba(62,203,110,0.10)", border: "1px solid rgba(62,203,110,0.25)" }}>
              <div className="text-[8px] uppercase" style={{ color: "rgba(255,255,255,0.4)" }}>{k}</div>
              <div className="text-[11px] font-bold" style={{ color: "#B9FFD1", fontFamily: "'JetBrains Mono', monospace" }}>{v}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Per-lane arrivals / before-after */}
      <div>
        <span className="text-[11px] font-bold text-white tracking-[0.12em] uppercase block mb-1.5">
          All queues this step
        </span>
        <div className="grid grid-cols-4 gap-1">
          {lanes.map((l) => (
            <div
              key={l.lane}
              className="rounded px-2 py-1 flex items-center justify-between"
              style={{
                background: l.wasSelected ? "rgba(62,203,110,0.18)" : "rgba(74,158,240,0.08)",
                border: `1px solid ${l.wasSelected ? "rgba(62,203,110,0.5)" : "rgba(74,158,240,0.18)"}`,
              }}
            >
              <span className="text-[10px] font-bold" style={{ color: l.wasSelected ? "#B9FFD1" : "#7EB8F7", fontFamily: "'JetBrains Mono', monospace" }}>
                {l.lane}
              </span>
              <span className="text-[9px]" style={{ color: "rgba(255,255,255,0.55)", fontFamily: "'JetBrains Mono', monospace" }}>
                {l.before}
                {l.arrived > 0 && <span style={{ color: "#FFD27E" }}>+{l.arrived}</span>}
                {l.passed > 0 && <span style={{ color: "#7CE7A0" }}>-{l.passed}</span>}
                {" = "}
                {l.after}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Outgoing lanes */}
      <div>
        <span className="text-[11px] font-bold text-white tracking-[0.12em] uppercase block mb-1.5">
          Outgoing lanes
        </span>
        <div className="grid grid-cols-4 gap-1">
          {outgoing.map((o) => (
            <div
              key={o.direction}
              className="rounded px-2 py-1"
              style={{ background: "rgba(240,120,64,0.08)", border: "1px solid rgba(240,120,64,0.18)" }}
            >
              <div className="text-[9px] font-bold mb-0.5" style={{ color: "#FFAD7A" }}>{o.direction} exit</div>
              <div className="text-[9px]" style={{ color: "rgba(255,255,255,0.55)", fontFamily: "'JetBrains Mono', monospace" }}>
                {o.before}
                {o.in > 0 && <span style={{ color: "#7CE7A0" }}> +{o.in}</span>}
                {o.out > 0 && <span style={{ color: "#FF8A8A" }}> -{o.out}</span>}
                {" = "}
                {o.after}
              </div>
            </div>
          ))}
        </div>
        <p className="text-[9px] mt-1.5" style={{ color: "rgba(255,255,255,0.35)" }}>
          {details.carsLeftSystem} car{details.carsLeftSystem === 1 ? "" : "s"} left the system this step
          {details.blockedArrivals > 0 && `, ${details.blockedArrivals} arrival(s) blocked (lane at max queue)`}.
          {algorithm === "Baseline" ? "" : " Values above are the trained agent's own estimates, not ground truth."}
        </p>
      </div>
    </div>
  );
}

// ── Queue-per-step comparison panel ───────────────────────────────────────────
// One clean chart: average queue per step per algorithm, over 100 fresh
// no-exploration evaluation episodes each -- every algorithm, including
// Baseline, is evaluated the same way, so this answers "which controller
// minimizes queue occupancy" fairly. This is a queue-occupancy proxy for
// waiting time, not literal per-car dwell time.
function WaitingComparisonPanel({ data }: { data: WaitingComparison }) {
  const rows = ALGORITHM_OPTIONS
    .map((alg) => ({ algorithm: alg, ...data[alg] }))
    .filter((r) => r.available && r.avgWaiting !== null)
    .sort((a, b) => (a.avgWaiting as number) - (b.avgWaiting as number));

  const bestValue = rows[0]?.avgWaiting ?? null;

  return (
    <div
      className="rounded-xl p-3 flex flex-col gap-2"
      style={{ background: "#141922", border: "1px solid rgba(255,255,255,0.08)", width: "100%", maxWidth: 748 }}
    >
      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-full inline-block" style={{ background: "#3ECB6E" }} />
        <span className="text-[11px] font-bold text-white tracking-[0.12em] uppercase">Queue-per-step comparison</span>
        <span className="text-[9px]" style={{ color: "rgba(255,255,255,0.4)", fontFamily: "'JetBrains Mono', monospace" }}>
          avg queue / step · last 100 episodes
        </span>
      </div>

      {rows.length === 0 ? (
        <p className="text-[10px]" style={{ color: "rgba(255,255,255,0.5)" }}>
          No trained agents available yet -- run `train_all.py` to compare against Baseline.
        </p>
      ) : (
        <div style={{ width: "100%", height: 40 * rows.length + 20 }}>
          <ResponsiveContainer>
            <BarChart data={rows} layout="vertical" margin={{ top: 0, right: 36, bottom: 0, left: 0 }}>
              <XAxis type="number" tick={{ fill: "rgba(255,255,255,0.4)", fontSize: 9 }} axisLine={false} tickLine={false} />
              <YAxis
                type="category"
                dataKey="algorithm"
                width={78}
                tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 9, fontFamily: "'JetBrains Mono', monospace" }}
                axisLine={false}
                tickLine={false}
              />
              <Bar dataKey="avgWaiting" radius={2} label={{ position: "right", fill: "rgba(255,255,255,0.7)", fontSize: 9 }}>
                {rows.map((r) => (
                  <Cell key={r.algorithm} fill={r.avgWaiting === bestValue ? "#3ECB6E" : "rgba(127,119,221,0.55)"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {ALGORITHM_OPTIONS.some((alg) => data[alg] && !data[alg].available) && (
        <p className="text-[9px]" style={{ color: "rgba(255,255,255,0.35)" }}>
          {ALGORITHM_OPTIONS.filter((alg) => data[alg] && !data[alg].available).join(", ")} not yet trained -- excluded above.
        </p>
      )}
    </div>
  );
}

export default function App() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [algorithm, setAlgorithm] = useState("Baseline");
  const [availability, setAvailability] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const [waitingComparison, setWaitingComparison] = useState<WaitingComparison | null>(null);
  const [showWaitingComparison, setShowWaitingComparison] = useState(false);
  const [loadingWaitingComparison, setLoadingWaitingComparison] = useState(false);

  const toggleWaitingComparison = async () => {
    if (showWaitingComparison) {
      setShowWaitingComparison(false);
      return;
    }
    setShowWaitingComparison(true);
    if (waitingComparison) return; // already loaded once this session
    setLoadingWaitingComparison(true);
    setError(null);
    try {
      const data = await api<WaitingComparison>("/waiting-comparison");
      setWaitingComparison(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load queue-per-step comparison");
    } finally {
      setLoadingWaitingComparison(false);
    }
  };

  const startEpisode = async (alg: string) => {
    setLoading(true);
    setError(null);
    try {
      const snap = await api<Snapshot>("/episode/new", {
        method: "POST",
        body: JSON.stringify({ algorithm: alg }),
      });
      setSnapshot(snap);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start episode");
    } finally {
      setLoading(false);
    }
  };

  const stepForward = async () => {
    setLoading(true);
    setError(null);
    try {
      const snap = await api<Snapshot>("/episode/step", { method: "POST" });
      setSnapshot(snap);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to step");
    } finally {
      setLoading(false);
    }
  };

  const stepBack = async () => {
    setLoading(true);
    setError(null);
    try {
      const snap = await api<Snapshot>("/episode/back", { method: "POST" });
      setSnapshot(snap);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to step back");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    api<Record<string, boolean>>("/algorithms")
      .then(setAvailability)
      .catch(() => setError("Can't reach the backend — is `uvicorn api:app` running?"));
    startEpisode("Baseline");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const inc = (id: string) => snapshot?.incoming[id] ?? 0;
  const out = (dir: string) => snapshot?.outgoing[dir] ?? 0;
  const isActiveLane = (id: string) => snapshot?.selectedLane === id;
  const isActiveExit = (dir: string) => snapshot?.exitDirection === dir;

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center gap-5 p-6"
      style={{ background: "#0D111A", fontFamily: "'Inter', sans-serif" }}
    >
      {/* Header */}
      <div className="text-center select-none">
        <h1 className="text-2xl font-bold text-white tracking-tight leading-none">
          Traffic Signal RL Visualizer
        </h1>
        <p className="text-[11px] font-medium tracking-[0.18em] text-gray-400 mt-1 uppercase">
          4-Way · 12 Incoming Queues · 4 Outgoing Queues
        </p>
        {error && (
          <p className="text-[11px] mt-2" style={{ color: "#FF8A8A" }}>
            {error}
          </p>
        )}
      </div>

      <div className="flex items-start gap-5 flex-wrap justify-center">
        {/* ── SVG Diagram ─────────────────────────────────────────── */}
        <div className="flex flex-col gap-3">
          <div
            className="rounded-2xl overflow-hidden shadow-2xl"
            style={{ border: "1px solid rgba(255,255,255,0.08)", opacity: loading ? 0.6 : 1, transition: "opacity 120ms" }}
          >
            <svg width={SZ} height={SZ} viewBox={`0 0 ${SZ} ${SZ}`} style={{ display: "block" }}>
              {/* Ground */}
              <rect width={SZ} height={SZ} fill={GROUND} />

              {/* Corner grass zones */}
              <rect x={0} y={0} width={X1} height={Y1} fill={GRASS} opacity={0.46} />
              <rect x={X2} y={0} width={SZ - X2} height={Y1} fill={GRASS} opacity={0.46} />
              <rect x={0} y={Y2} width={X1} height={SZ - Y2} fill={GRASS} opacity={0.46} />
              <rect x={X2} y={Y2} width={SZ - X2} height={SZ - Y2} fill={GRASS} opacity={0.46} />

              {/* Trees */}
              {TREES.map(([tx, ty, tr], i) => (
                <Tree key={i} cx={tx} cy={ty} r={tr} />
              ))}

              {/* ── Road surfaces ──────────────────────────────── */}
              <rect x={X1} y={0} width={RW} height={Y1} fill={ROAD} />
              <rect x={X1} y={Y2} width={RW} height={SZ - Y2} fill={ROAD} />
              <rect x={0} y={Y1} width={X1} height={RW} fill={ROAD} />
              <rect x={X2} y={Y1} width={SZ - X2} height={RW} fill={ROAD} />
              <rect x={X1} y={Y1} width={RW} height={RW} fill={ROAD_DARK} />

              {/* ── Lane tints (one rect per lane so the RL choice can light up) ── */}
              {/* North incoming */}
              {NORTH_LANES.map((id, i) => (
                <rect key={id} x={X1 + i * LW} y={0} width={LW} height={Y1} fill={isActiveLane(id) ? ACTIVE_INC : INC_TINT} />
              ))}
              <rect x={X1 + 3 * LW} y={0} width={LW} height={Y1} fill={isActiveExit("N") ? ACTIVE_OUT : OUT_TINT} />

              {/* South incoming (outgoing lane is on the left/pos0) */}
              <rect x={X1} y={Y2} width={LW} height={SZ - Y2} fill={isActiveExit("S") ? ACTIVE_OUT : OUT_TINT} />
              {SOUTH_LANES.map((id, i) => (
                <rect key={id} x={X1 + LW + i * LW} y={Y2} width={LW} height={SZ - Y2} fill={isActiveLane(id) ? ACTIVE_INC : INC_TINT} />
              ))}

              {/* West incoming (outgoing lane is on top/pos0) */}
              <rect x={0} y={Y1} width={X1} height={LW} fill={isActiveExit("W") ? ACTIVE_OUT : OUT_TINT} />
              {WEST_LANES.map((id, i) => (
                <rect key={id} x={0} y={Y1 + LW + i * LW} width={X1} height={LW} fill={isActiveLane(id) ? ACTIVE_INC : INC_TINT} />
              ))}

              {/* East incoming */}
              {EAST_LANES.map((id, i) => (
                <rect key={id} x={X2} y={Y1 + i * LW} width={SZ - X2} height={LW} fill={isActiveLane(id) ? ACTIVE_INC : INC_TINT} />
              ))}
              <rect x={X2} y={Y1 + 3 * LW} width={SZ - X2} height={LW} fill={isActiveExit("E") ? ACTIVE_OUT : OUT_TINT} />

              {/* ── Road edge lines ────────────────────────────── */}
              {[
                [X1, 0, X1, Y1], [X2, 0, X2, Y1],
                [X1, Y2, X1, SZ], [X2, Y2, X2, SZ],
                [0, Y1, X1, Y1], [0, Y2, X1, Y2],
                [X2, Y1, SZ, Y1], [X2, Y2, SZ, Y2],
              ].map(([ax1, ay1, ax2, ay2], i) => (
                <line key={i} x1={ax1} y1={ay1} x2={ax2} y2={ay2} stroke={WHITE} strokeWidth={1} opacity={0.35} />
              ))}

              {/* ── Lane dividers ──────────────────────────────── */}
              <line x1={X1 + LW} y1={0} x2={X1 + LW} y2={Y1} stroke={WHITE} strokeWidth={1.5} strokeDasharray="9,9" opacity={0.55} />
              <line x1={X1 + 2 * LW} y1={0} x2={X1 + 2 * LW} y2={Y1} stroke={WHITE} strokeWidth={1.5} strokeDasharray="9,9" opacity={0.55} />
              <line x1={X1 + 3 * LW} y1={0} x2={X1 + 3 * LW} y2={Y1} stroke={YELLOW} strokeWidth={2.5} />

              <line x1={X1 + LW} y1={Y2} x2={X1 + LW} y2={SZ} stroke={YELLOW} strokeWidth={2.5} />
              <line x1={X1 + 2 * LW} y1={Y2} x2={X1 + 2 * LW} y2={SZ} stroke={WHITE} strokeWidth={1.5} strokeDasharray="9,9" opacity={0.55} />
              <line x1={X1 + 3 * LW} y1={Y2} x2={X1 + 3 * LW} y2={SZ} stroke={WHITE} strokeWidth={1.5} strokeDasharray="9,9" opacity={0.55} />

              <line x1={0} y1={Y1 + LW} x2={X1} y2={Y1 + LW} stroke={YELLOW} strokeWidth={2.5} />
              <line x1={0} y1={Y1 + 2 * LW} x2={X1} y2={Y1 + 2 * LW} stroke={WHITE} strokeWidth={1.5} strokeDasharray="9,9" opacity={0.55} />
              <line x1={0} y1={Y1 + 3 * LW} x2={X1} y2={Y1 + 3 * LW} stroke={WHITE} strokeWidth={1.5} strokeDasharray="9,9" opacity={0.55} />

              <line x1={X2} y1={Y1 + LW} x2={SZ} y2={Y1 + LW} stroke={WHITE} strokeWidth={1.5} strokeDasharray="9,9" opacity={0.55} />
              <line x1={X2} y1={Y1 + 2 * LW} x2={SZ} y2={Y1 + 2 * LW} stroke={WHITE} strokeWidth={1.5} strokeDasharray="9,9" opacity={0.55} />
              <line x1={X2} y1={Y1 + 3 * LW} x2={SZ} y2={Y1 + 3 * LW} stroke={YELLOW} strokeWidth={2.5} />

              {/* ── Crosswalks ─────────────────────────────────── */}
              <Crosswalk x={X1} y={Y1 - 13} w={RW} h={13} vertical={true} />
              <Crosswalk x={X1} y={Y2} w={RW} h={13} vertical={true} />
              <Crosswalk x={X1 - 13} y={Y1} w={13} h={RW} vertical={false} />
              <Crosswalk x={X2} y={Y1} w={13} h={RW} vertical={false} />

              {/* ── Direction arrows (incoming point in, outgoing point out) ── */}
              {laneArrows("down", X1 + LW * 0.5, 0, Y1)}
              {laneArrows("down", X1 + LW * 1.5, 0, Y1)}
              {laneArrows("down", X1 + LW * 2.5, 0, Y1)}
              {laneArrows("up", X1 + LW * 3.5, 0, Y1)}
              {laneArrows("down", X1 + LW * 0.5, Y2, SZ)}
              {laneArrows("up", X1 + LW * 1.5, Y2, SZ)}
              {laneArrows("up", X1 + LW * 2.5, Y2, SZ)}
              {laneArrows("up", X1 + LW * 3.5, Y2, SZ)}
              {laneArrows("left", Y1 + LW * 0.5, 0, X1)}
              {laneArrows("right", Y1 + LW * 1.5, 0, X1)}
              {laneArrows("right", Y1 + LW * 2.5, 0, X1)}
              {laneArrows("right", Y1 + LW * 3.5, 0, X1)}
              {laneArrows("left", Y1 + LW * 0.5, X2, SZ)}
              {laneArrows("left", Y1 + LW * 1.5, X2, SZ)}
              {laneArrows("left", Y1 + LW * 2.5, X2, SZ)}
              {laneArrows("right", Y1 + LW * 3.5, X2, SZ)}

              {/* ── Compass labels ─────────────────────── */}
              <text x={CX} y={Y1 - 22} textAnchor="middle" fill={WHITE} fontSize={13} fontWeight="700" fontFamily="Inter, sans-serif" opacity={0.9} letterSpacing={2}>N</text>
              <text x={CX} y={Y2 + 34} textAnchor="middle" fill={WHITE} fontSize={13} fontWeight="700" fontFamily="Inter, sans-serif" opacity={0.9} letterSpacing={2}>S</text>
              <text x={X1 - 22} y={CY + 5} textAnchor="middle" fill={WHITE} fontSize={13} fontWeight="700" fontFamily="Inter, sans-serif" opacity={0.9} letterSpacing={2}>W</text>
              <text x={X2 + 22} y={CY + 5} textAnchor="middle" fill={WHITE} fontSize={13} fontWeight="700" fontFamily="Inter, sans-serif" opacity={0.9} letterSpacing={2}>E</text>

              {/* ── Lane queue labels + live counts ──────────────── */}
              {/* North */}
              {NORTH_LANES.map((id, i) => (
                <g key={id}>
                  <text x={X1 + LW * (i + 0.5)} y={14} textAnchor="middle" fill={isActiveLane(id) ? "#B9FFD1" : "#7EC8FF"} fontSize={8} fontFamily="'JetBrains Mono', monospace" fontWeight="700">{id}</text>
                  <text x={X1 + LW * (i + 0.5)} y={24} textAnchor="middle" fill={isActiveLane(id) ? "#B9FFD1" : "rgba(255,255,255,0.65)"} fontSize={9} fontFamily="'JetBrains Mono', monospace" fontWeight="700">{inc(id)}</text>
                </g>
              ))}
              <text x={X1 + LW * 3.5} y={14} textAnchor="middle" fill={isActiveExit("N") ? "#B9FFD1" : "#FFA76B"} fontSize={8} fontFamily="'JetBrains Mono', monospace" fontWeight="700">N↑</text>
              <text x={X1 + LW * 3.5} y={24} textAnchor="middle" fill={isActiveExit("N") ? "#B9FFD1" : "rgba(255,255,255,0.65)"} fontSize={9} fontFamily="'JetBrains Mono', monospace" fontWeight="700">{out("N")}</text>

              {/* South */}
              <text x={X1 + LW * 0.5} y={SZ - 18} textAnchor="middle" fill={isActiveExit("S") ? "#B9FFD1" : "#FFA76B"} fontSize={8} fontFamily="'JetBrains Mono', monospace" fontWeight="700">S↓</text>
              <text x={X1 + LW * 0.5} y={SZ - 8} textAnchor="middle" fill={isActiveExit("S") ? "#B9FFD1" : "rgba(255,255,255,0.65)"} fontSize={9} fontFamily="'JetBrains Mono', monospace" fontWeight="700">{out("S")}</text>
              {SOUTH_LANES.map((id, i) => (
                <g key={id}>
                  <text x={X1 + LW * (i + 1.5)} y={SZ - 18} textAnchor="middle" fill={isActiveLane(id) ? "#B9FFD1" : "#7EC8FF"} fontSize={8} fontFamily="'JetBrains Mono', monospace" fontWeight="700">{id}</text>
                  <text x={X1 + LW * (i + 1.5)} y={SZ - 8} textAnchor="middle" fill={isActiveLane(id) ? "#B9FFD1" : "rgba(255,255,255,0.65)"} fontSize={9} fontFamily="'JetBrains Mono', monospace" fontWeight="700">{inc(id)}</text>
                </g>
              ))}

              {/* West */}
              <text x={11} y={Y1 + LW * 0.5 - 1} textAnchor="middle" fill={isActiveExit("W") ? "#B9FFD1" : "#FFA76B"} fontSize={8} fontFamily="'JetBrains Mono', monospace" fontWeight="700">W←</text>
              <text x={11} y={Y1 + LW * 0.5 + 8} textAnchor="middle" fill={isActiveExit("W") ? "#B9FFD1" : "rgba(255,255,255,0.65)"} fontSize={9} fontFamily="'JetBrains Mono', monospace" fontWeight="700">{out("W")}</text>
              {WEST_LANES.map((id, i) => (
                <g key={id}>
                  <text x={11} y={Y1 + LW * (i + 1.5) - 1} textAnchor="middle" fill={isActiveLane(id) ? "#B9FFD1" : "#7EC8FF"} fontSize={8} fontFamily="'JetBrains Mono', monospace" fontWeight="700">{id}</text>
                  <text x={11} y={Y1 + LW * (i + 1.5) + 8} textAnchor="middle" fill={isActiveLane(id) ? "#B9FFD1" : "rgba(255,255,255,0.65)"} fontSize={9} fontFamily="'JetBrains Mono', monospace" fontWeight="700">{inc(id)}</text>
                </g>
              ))}

              {/* East */}
              {EAST_LANES.map((id, i) => (
                <g key={id}>
                  <text x={SZ - 11} y={Y1 + LW * (i + 0.5) - 1} textAnchor="middle" fill={isActiveLane(id) ? "#B9FFD1" : "#7EC8FF"} fontSize={8} fontFamily="'JetBrains Mono', monospace" fontWeight="700">{id}</text>
                  <text x={SZ - 11} y={Y1 + LW * (i + 0.5) + 8} textAnchor="middle" fill={isActiveLane(id) ? "#B9FFD1" : "rgba(255,255,255,0.65)"} fontSize={9} fontFamily="'JetBrains Mono', monospace" fontWeight="700">{inc(id)}</text>
                </g>
              ))}
              <text x={SZ - 11} y={Y1 + LW * 3.5 - 1} textAnchor="middle" fill={isActiveExit("E") ? "#B9FFD1" : "#FFA76B"} fontSize={8} fontFamily="'JetBrains Mono', monospace" fontWeight="700">E→</text>
              <text x={SZ - 11} y={Y1 + LW * 3.5 + 8} textAnchor="middle" fill={isActiveExit("E") ? "#B9FFD1" : "rgba(255,255,255,0.65)"} fontSize={9} fontFamily="'JetBrains Mono', monospace" fontWeight="700">{out("E")}</text>
            </svg>
          </div>

          {/* ── Controls ─────────────────────────────────────────── */}
          <div
            className="rounded-xl p-3 flex items-center gap-2 flex-wrap"
            style={{ background: "#141922", border: "1px solid rgba(255,255,255,0.08)" }}
          >
            <select
              value={algorithm}
              onChange={(e) => {
                setAlgorithm(e.target.value);
                startEpisode(e.target.value);
              }}
              disabled={loading}
              className="rounded text-[11px] font-mono py-1.5 px-2"
              style={{ background: "#1B222E", color: "#7EB8F7", border: "1px solid rgba(74,158,240,0.3)" }}
            >
              {ALGORITHM_OPTIONS.map((alg) => (
                <option key={alg} value={alg}>
                  {alg}{availability[alg] === false && alg !== "Baseline" ? " (untrained → baseline)" : ""}
                </option>
              ))}
            </select>

            <button
              onClick={() => startEpisode(algorithm)}
              disabled={loading}
              className="rounded text-[11px] font-mono py-1.5 px-3 font-semibold"
              style={{ background: "rgba(74,158,240,0.15)", color: "#7EB8F7", border: "1px solid rgba(74,158,240,0.3)" }}
            >
              New Episode
            </button>
            <button
              onClick={stepBack}
              disabled={loading || !snapshot || snapshot.step === 0}
              className="rounded text-[11px] font-mono py-1.5 px-3 font-semibold disabled:opacity-40"
              style={{ background: "rgba(255,255,255,0.06)", color: "#E5E7EB", border: "1px solid rgba(255,255,255,0.12)" }}
            >
              ◀ Previous
            </button>
            <button
              onClick={stepForward}
              disabled={loading || snapshot?.done}
              className="rounded text-[11px] font-mono py-1.5 px-3 font-semibold disabled:opacity-40"
              style={{ background: "rgba(62,203,110,0.18)", color: "#7CE7A0", border: "1px solid rgba(62,203,110,0.35)" }}
            >
              Next Step ▶
            </button>
            <button
              onClick={() => setShowDetails((v) => !v)}
              disabled={!snapshot?.details}
              className="rounded text-[11px] font-mono py-1.5 px-3 font-semibold disabled:opacity-40"
              style={{
                background: showDetails ? "rgba(127,119,221,0.22)" : "rgba(127,119,221,0.10)",
                color: "#B7B0F0",
                border: `1px solid ${showDetails ? "rgba(127,119,221,0.55)" : "rgba(127,119,221,0.25)"}`,
              }}
            >
              {showDetails ? "Hide details ▲" : "Show details ▾"}
            </button>
            <button
              onClick={toggleWaitingComparison}
              disabled={loadingWaitingComparison}
              className="rounded text-[11px] font-mono py-1.5 px-3 font-semibold disabled:opacity-40"
              style={{
                background: showWaitingComparison ? "rgba(62,203,110,0.18)" : "rgba(62,203,110,0.08)",
                color: "#7CE7A0",
                border: `1px solid ${showWaitingComparison ? "rgba(62,203,110,0.5)" : "rgba(62,203,110,0.22)"}`,
                marginLeft: "auto",
              }}
            >
              {loadingWaitingComparison
                ? "Running evaluation rollouts…"
                : showWaitingComparison
                ? "Hide queue comparison ▲"
                : "Compare queue per step ▾"}
            </button>
          </div>

          {showDetails && snapshot?.details && (
            <DetailsPanel details={snapshot.details} algorithm={snapshot.algorithm} />
          )}
          {showWaitingComparison && waitingComparison && (
            <WaitingComparisonPanel data={waitingComparison} />
          )}
        </div>

        {/* ── Info Panels ─────────────────────────────────────────── */}
        <div className="flex flex-col gap-3" style={{ minWidth: 188 }}>
          {/* Live stats */}
          <div
            className="rounded-xl p-3"
            style={{ background: "#141922", border: "1px solid rgba(255,255,255,0.08)" }}
          >
            <div className="flex items-center gap-2 mb-2.5">
              <span className="w-2 h-2 rounded-full inline-block" style={{ background: "#3ECB6E" }} />
              <span className="text-[11px] font-bold text-white tracking-[0.12em] uppercase">Live State</span>
            </div>
            {[
              ["Algorithm", snapshot?.algorithm ?? "—"],
              ["Episode", snapshot?.episode ?? "—"],
              ["Step", snapshot?.step ?? "—"],
              ["Active lane", snapshot?.selectedLane ?? "—"],
              ["Reward", snapshot?.reward ?? "—"],
              ["Total queue", snapshot?.totalQueue ?? "—"],
              ["Cars passed", snapshot?.carsPassed ?? "—"],
            ].map(([k, v]) => (
              <div key={k as string} className="flex justify-between items-baseline mb-1 last:mb-0">
                <span className="text-[9px] uppercase tracking-wider" style={{ color: "rgba(255,255,255,0.35)", fontFamily: "Inter, sans-serif" }}>{k}</span>
                <span className="text-[10px] font-semibold" style={{ color: "rgba(255,255,255,0.85)", fontFamily: "'JetBrains Mono', monospace" }}>{String(v)}</span>
              </div>
            ))}
          </div>

          {/* Incoming queues */}
          <div
            className="rounded-xl p-3"
            style={{ background: "#141922", border: "1px solid rgba(255,255,255,0.08)" }}
          >
            <div className="flex items-center gap-2 mb-2.5">
              <span className="w-2 h-2 rounded-full inline-block" style={{ background: "#4A9EF0" }} />
              <span className="text-[11px] font-bold text-white tracking-[0.12em] uppercase">Incoming (12)</span>
            </div>
            <div className="grid grid-cols-3 gap-1">
              {[...NORTH_LANES, ...SOUTH_LANES, ...EAST_LANES, ...WEST_LANES].map((id) => (
                <div
                  key={id}
                  className="rounded text-center text-[10px] font-mono py-1"
                  style={{
                    background: isActiveLane(id) ? "rgba(62,203,110,0.28)" : "rgba(74,158,240,0.10)",
                    border: `1px solid ${isActiveLane(id) ? "rgba(62,203,110,0.7)" : "rgba(74,158,240,0.25)"}`,
                    color: isActiveLane(id) ? "#B9FFD1" : "#7EB8F7",
                    fontFamily: "'JetBrains Mono', monospace",
                  }}
                >
                  {id} <span style={{ opacity: 0.9 }}>{inc(id)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Outgoing queues */}
          <div
            className="rounded-xl p-3"
            style={{ background: "#141922", border: "1px solid rgba(255,255,255,0.08)" }}
          >
            <div className="flex items-center gap-2 mb-2.5">
              <span className="w-2 h-2 rounded-full inline-block" style={{ background: "#F07840" }} />
              <span className="text-[11px] font-bold text-white tracking-[0.12em] uppercase">Outgoing (4)</span>
            </div>
            <div className="flex flex-col gap-1">
              {OUT_QUEUES.map(({ id, dir, label }) => (
                <div
                  key={id}
                  className="rounded py-1.5 px-2 flex items-center justify-between"
                  style={{
                    background: isActiveExit(dir) ? "rgba(62,203,110,0.24)" : "rgba(240,120,64,0.10)",
                    border: `1px solid ${isActiveExit(dir) ? "rgba(62,203,110,0.65)" : "rgba(240,120,64,0.25)"}`,
                    fontFamily: "'JetBrains Mono', monospace",
                  }}
                >
                  <span className="text-[11px] font-bold" style={{ color: isActiveExit(dir) ? "#B9FFD1" : "#FFAD7A" }}>{id}</span>
                  <span className="text-[9px]" style={{ color: "rgba(255,255,255,0.45)" }}>{label}</span>
                  <span className="text-[11px] font-bold" style={{ color: isActiveExit(dir) ? "#B9FFD1" : "#FFAD7A" }}>{out(dir)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Legend */}
          <div
            className="rounded-xl p-3"
            style={{ background: "#141922", border: "1px solid rgba(255,255,255,0.08)" }}
          >
            <span className="text-[11px] font-bold text-white tracking-[0.12em] uppercase block mb-2.5">Legend</span>
            <div className="flex flex-col gap-2">
              {[
                { swatch: { background: "rgba(74,158,240,0.35)", border: "1px solid rgba(74,158,240,0.6)" }, label: "Incoming lane" },
                { swatch: { background: "rgba(240,120,64,0.35)", border: "1px solid rgba(240,120,64,0.6)" }, label: "Outgoing lane" },
                { swatch: { background: "rgba(62,203,110,0.55)", border: "1px solid rgba(62,203,110,0.9)" }, label: "RL's active lane this step" },
                { swatch: { background: YELLOW, height: 3 }, label: "Center divider" },
                { swatch: { background: "repeating-linear-gradient(90deg,#fff 0 5px,transparent 5px 9px)", height: 10, opacity: 0.75 }, label: "Crosswalk" },
              ].map(({ swatch, label }) => (
                <div key={label} className="flex items-center gap-2.5">
                  <div className="rounded-sm flex-shrink-0" style={{ width: 28, height: 10, ...swatch }} />
                  <span className="text-[10px]" style={{ color: "rgba(255,255,255,0.5)", fontFamily: "Inter, sans-serif" }}>{label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}