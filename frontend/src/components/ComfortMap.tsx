import { useMemo, useRef } from "react";
import DeckGL from "deck.gl";
import { BitmapLayer } from "@deck.gl/layers";
import { OrthographicView } from "@deck.gl/core";
import { useStore } from "../store/store";
import type { ComfortMap as ComfortMapType } from "../types";

const SPEED_MAX = 10; // m/s — top of colour scale

// Smooth blue → green → yellow → red gradient
function windColor(speed: number): [number, number, number, number] {
    const t = Math.max(0, Math.min(1, speed / SPEED_MAX));
    let r: number, g: number, b: number;
    if (t < 1 / 3) {
        const f = t * 3;
        r = 0;
        g = f;
        b = 1 - f;
    } else if (t < 2 / 3) {
        const f = (t - 1 / 3) * 3;
        r = f;
        g = 1;
        b = 0;
    } else {
        const f = (t - 2 / 3) * 3;
        r = 1;
        g = 1 - f;
        b = 0;
    }
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255), 220];
}

function paintComfortCanvas(
    canvas: HTMLCanvasElement,
    cm: ComfortMapType,
): void {
    const [Nx, Ny] = cm.shape;
    if (canvas.width !== Nx || canvas.height !== Ny) {
        canvas.width = Nx;
        canvas.height = Ny;
    }
    const ctx = canvas.getContext("2d")!;
    const imgData = ctx.createImageData(Nx, Ny);

    for (let y = 0; y < Ny; y++) {
        for (let x = 0; x < Nx; x++) {
            const speed = cm.data[y * Nx + x];
            const [r, g, b, a] = windColor(speed);
            const canvasY = Ny - 1 - y;
            const idx = (canvasY * Nx + x) * 4;
            imgData.data[idx] = r;
            imgData.data[idx + 1] = g;
            imgData.data[idx + 2] = b;
            imgData.data[idx + 3] = a;
        }
    }

    ctx.putImageData(imgData, 0, 0);
}

export function ComfortMap() {
    const result = useStore(s => s.result);
    const jobState = useStore(s => s.jobState);

    // Reuse a single canvas element across snapshots
    const canvasRef = useRef<HTMLCanvasElement>(document.createElement("canvas"));
    // Increment to force DeckGL layer invalidation
    const revRef = useRef(0);

    const layers = useMemo(() => {
        if (!result) return [];
        const cm = result.comfortMap;
        const [Nx, Ny] = cm.shape;
        paintComfortCanvas(canvasRef.current, cm);
        revRef.current++;
        return [
            new BitmapLayer({
                id: `comfort-${revRef.current}`,
                image: canvasRef.current,
                bounds: [0, 0, Nx, Ny],
            }),
        ];
    }, [result]);

    const [Nx, Ny] = result?.comfortMap.shape ?? [128, 128];

    return (
        <div className="relative w-full h-full">
            <DeckGL
                views={new OrthographicView({ id: "ortho" })}
                initialViewState={{ target: [Nx / 2, Ny / 2, 0], zoom: 1 }}
                controller
                layers={layers}
                style={{ position: "absolute", inset: 0 }}
            />

            {jobState === "running" && (
                <div className="absolute inset-0 bg-black/40 flex items-center justify-center pointer-events-none">
                    <span className="text-white text-sm">Computing…</span>
                </div>
            )}

            {/* Wind speed legend */}
            <div
                className="absolute bottom-3 right-3 bg-gray-900/80 rounded-lg p-3 pointer-events-none"
                style={{ width: 140 }}
            >
                <div className="text-gray-400 text-xs uppercase tracking-wide mb-2">
                    Wind Speed
                </div>
                <div
                    className="rounded"
                    style={{
                        height: 10,
                        background:
                            "linear-gradient(to right, #0000ff, #00ff00, #ffff00, #ff0000)",
                    }}
                />
                <div className="flex justify-between mt-1">
                    <span className="text-gray-400 text-xs">0</span>
                    <span className="text-gray-400 text-xs">5</span>
                    <span className="text-gray-400 text-xs">10 m/s</span>
                </div>
            </div>
        </div>
    );
}
