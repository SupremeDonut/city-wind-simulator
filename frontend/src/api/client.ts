import * as LZ4 from "lz4js";
import type {
    VelocityField,
    Preset,
    WindParams,
    SimulationResult,
    ComfortMap,
} from "../types";

const API_BASE = "http://localhost:8000";

function b64ToFloat32(b64: string, encoding?: string): Float32Array {
    const bin = atob(b64);
    let u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);

    // Decompress if lz4-compressed — u8 is now raw float16 bytes
    if (encoding === "lz4+float16") {
        u8 = LZ4.decompress(u8);
    }

    // u8 may be a view with non-zero byteOffset — slice to get aligned buffer
    const buf = u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
    const u16 = new Uint16Array(buf);
    const f32 = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
        const h = u16[i];
        const sign = (h >> 15) & 1;
        const exp  = (h >> 10) & 0x1f;
        const frac = h & 0x3ff;
        if (exp === 0) {
            // subnormal or zero
            f32[i] = (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
        } else if (exp === 31) {
            f32[i] = frac ? NaN : (sign ? -Infinity : Infinity);
        } else {
            f32[i] = (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
        }
    }
    return f32;
}

export async function fetchPresets(): Promise<Preset[]> {
    const res = await fetch(`${API_BASE}/presets`);
    if (!res.ok) throw new Error(`fetchPresets failed: ${res.status}`);
    return res.json();
}

export async function predict(
    preset: Preset,
    params: WindParams,
): Promise<SimulationResult> {
    const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preset_id: preset.id, wind_params: params }),
    });
    if (!res.ok) throw new Error(`predict failed: ${res.status}`);

    const json = await res.json();

    const velocityField: VelocityField = {
        data: b64ToFloat32(json.velocityField.data, json.velocityField.encoding),
        shape: json.velocityField.shape,
        min: json.velocityField.min,
        max: json.velocityField.max,
        domain: json.velocityField.domain,
    };

    const comfortMap: ComfortMap = {
        data: b64ToFloat32(json.comfortMap.data, json.comfortMap.encoding),
        shape: json.comfortMap.shape,
    };

    return { velocityField, comfortMap };
}

export async function simulateLBM(
    preset: Preset,
    params: WindParams,
    callbacks: {
        onProgress: (value: number) => void;
        onSnapshot: (result: SimulationResult) => void;
        onDone: () => void;
        onError: (msg: string) => void;
    },
    options?: { numSteps?: number; snapshotInterval?: number },
): Promise<string> {
    const numSteps = options?.numSteps ?? 3000;
    const snapshotInterval = options?.snapshotInterval ?? 200;

    const res = await fetch(`${API_BASE}/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            preset_id: preset.id,
            wind_params: params,
            resolution: 8.0,
            num_steps: numSteps,
            snapshot_interval: snapshotInterval,
        }),
    });
    if (!res.ok) throw new Error(`simulate failed: ${res.status}`);

    const { job_id } = await res.json();

    const ws = new WebSocket(`ws://localhost:8000/ws/simulation/${job_id}`);
    ws.onmessage = ev => {
        const msg = JSON.parse(ev.data);
        if (msg.type === "ping") {
            return;
        } else if (msg.type === "progress") {
            callbacks.onProgress(msg.value);
        } else if (msg.type === "snapshot") {
            const velocityField: VelocityField = {
                data: b64ToFloat32(msg.velocityField.data, msg.velocityField.encoding),
                shape: msg.velocityField.shape,
                min: msg.velocityField.min,
                max: msg.velocityField.max,
                domain: msg.velocityField.domain,
            };
            const comfortMap: ComfortMap = {
                data: b64ToFloat32(msg.comfortMap.data, msg.comfortMap.encoding),
                shape: msg.comfortMap.shape,
            };
            callbacks.onSnapshot({ velocityField, comfortMap });
        } else if (msg.type === "done") {
            if (msg.velocityField && msg.comfortMap) {
                const velocityField: VelocityField = {
                    data: b64ToFloat32(msg.velocityField.data, msg.velocityField.encoding),
                    shape: msg.velocityField.shape,
                    min: msg.velocityField.min,
                    max: msg.velocityField.max,
                    domain: msg.velocityField.domain,
                };
                const comfortMap: ComfortMap = {
                    data: b64ToFloat32(msg.comfortMap.data, msg.comfortMap.encoding),
                    shape: msg.comfortMap.shape,
                };
                callbacks.onSnapshot({ velocityField, comfortMap });
            }
            callbacks.onDone();
            ws.close();
        } else if (msg.type === "error") {
            callbacks.onError(msg.message);
            ws.close();
        }
    };

    return job_id;
}
