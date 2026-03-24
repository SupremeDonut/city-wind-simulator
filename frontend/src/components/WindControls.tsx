import { useStore } from "../store/store";

const ROUGHNESS_OPTIONS = [
    { label: "Open sea", value: 0.01 },
    { label: "Farmland", value: 0.03 },
    { label: "Suburban", value: 0.1 },
    { label: "Urban", value: 0.3 },
    { label: "City center", value: 1.0 },
];

export function WindControls() {
    const windParams = useStore(s => s.windParams);
    const dirty = useStore(s => s.dirty);
    const jobState = useStore(s => s.jobState);
    const setWindDirection = useStore(s => s.setWindDirection);
    const setWindSpeed = useStore(s => s.setWindSpeed);
    const setRoughness = useStore(s => s.setRoughness);
    const runPredict = useStore(s => s.runPredict);
    const runSimulation = useStore(s => s.runSimulation);
    const simProgress = useStore(s => s.simProgress);
    const selectedPreset = useStore(s => s.selectedPreset);

    const cx = 60,
        cy = 60,
        r = 45;

    const handleCompassClick = (e: React.MouseEvent<SVGSVGElement>) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const dx = e.clientX - rect.left - cx;
        const dy = e.clientY - rect.top - cy;
        // atan2(dx, -dy) gives meteorological: 0=north, 90=east
        let deg = (Math.atan2(dx, -dy) * 180) / Math.PI;
        if (deg < 0) deg += 360;
        setWindDirection(deg);
    };

    const arrowX = cx + r * Math.sin((windParams.direction * Math.PI) / 180);
    const arrowY = cy - r * Math.cos((windParams.direction * Math.PI) / 180);

    const canRecalc = dirty && jobState !== 'running';

    return (
        <div className="flex flex-col gap-5">
            {/* Compass */}
            <div>
                <div className="text-gray-400 text-xs uppercase tracking-wide mb-2">
                    Wind Direction
                </div>
                <div className="flex flex-col items-center gap-2">
                    <svg
                        width={120}
                        height={120}
                        className="cursor-crosshair"
                        onClick={handleCompassClick}
                    >
                        <circle
                            cx={cx}
                            cy={cy}
                            r={r}
                            fill="none"
                            stroke="#374151"
                            strokeWidth={1.5}
                        />
                        <text
                            x={cx}
                            y={cy - r - 6}
                            textAnchor="middle"
                            fill="#9ca3af"
                            fontSize={11}
                        >
                            N
                        </text>
                        <text
                            x={cx + r + 8}
                            y={cy + 4}
                            textAnchor="middle"
                            fill="#9ca3af"
                            fontSize={11}
                        >
                            E
                        </text>
                        <text
                            x={cx}
                            y={cy + r + 14}
                            textAnchor="middle"
                            fill="#9ca3af"
                            fontSize={11}
                        >
                            S
                        </text>
                        <text
                            x={cx - r - 8}
                            y={cy + 4}
                            textAnchor="middle"
                            fill="#9ca3af"
                            fontSize={11}
                        >
                            W
                        </text>
                        <line
                            x1={cx}
                            y1={cy}
                            x2={arrowX}
                            y2={arrowY}
                            stroke="#60a5fa"
                            strokeWidth={2}
                            strokeLinecap="round"
                        />
                        <circle cx={arrowX} cy={arrowY} r={4} fill="#60a5fa" />
                        <circle cx={cx} cy={cy} r={3} fill="#60a5fa" />
                    </svg>
                    <input
                        type="number"
                        min={0}
                        max={359}
                        value={Math.round(windParams.direction)}
                        onChange={e => {
                            const deg = ((Number(e.target.value) % 360) + 360) % 360;
                            setWindDirection(deg);
                        }}
                        className="w-20 bg-gray-700 text-white text-center text-sm rounded px-2 py-1 border border-gray-600"
                    />
                </div>
            </div>

            {/* Speed slider */}
            <div>
                <div className="text-gray-400 text-xs uppercase tracking-wide mb-2">
                    Wind Speed: {windParams.speed.toFixed(1)} m/s
                </div>
                <input
                    type="range"
                    min={0.5}
                    max={20}
                    step={0.5}
                    value={windParams.speed}
                    onChange={e => setWindSpeed(Number(e.target.value))}
                    className="w-full accent-blue-400"
                />
            </div>

            {/* Roughness dropdown */}
            <div>
                <div className="text-gray-400 text-xs uppercase tracking-wide mb-2">
                    Surface Roughness
                </div>
                <select
                    value={windParams.roughness}
                    onChange={e => setRoughness(Number(e.target.value))}
                    className="w-full bg-gray-700 text-white text-sm rounded px-2 py-1 border border-gray-600"
                >
                    {ROUGHNESS_OPTIONS.map(opt => (
                        <option key={opt.value} value={opt.value}>
                            {opt.label}
                        </option>
                    ))}
                </select>
            </div>

            {/* Recalculate button */}
            <button
                onClick={runPredict}
                disabled={!canRecalc}
                className={`w-full py-2 rounded text-sm font-medium transition-colors ${
                    canRecalc
                        ? 'bg-blue-500 hover:bg-blue-400 text-white cursor-pointer'
                        : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                }`}
            >
                {jobState === 'running' ? 'Computing…' : 'Recalculate'}
            </button>

            {/* Run Simulation button */}
            <button
                onClick={runSimulation}
                disabled={!selectedPreset || jobState === 'running'}
                className={`w-full py-2 rounded text-sm font-medium transition-colors ${
                    selectedPreset && jobState !== 'running'
                        ? 'bg-emerald-600 hover:bg-emerald-500 text-white cursor-pointer'
                        : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                }`}
            >
                {jobState === 'running'
                    ? `Simulating\u2026 ${Math.round(simProgress * 100)}%`
                    : 'Run Simulation'}
            </button>
        </div>
    );
}
