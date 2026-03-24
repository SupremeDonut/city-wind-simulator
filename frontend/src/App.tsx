import { useEffect, useRef, useState, useCallback } from "react";
import { useStore } from "./store/store";
import { PresetSelector } from "./components/PresetSelector";
import { WindControls } from "./components/WindControls";
import { Viewport3D } from "./components/Viewport3D";
import { ComfortMap } from "./components/ComfortMap";

const SIDEBAR_MIN = 200;
const SIDEBAR_MAX = 600;
const SIDEBAR_DEFAULT = 288; // w-72

function JobDot() {
    const jobState = useStore(s => s.jobState);
    const colors: Record<string, string> = {
        idle: "bg-gray-500",
        running: "bg-yellow-400 animate-pulse",
        done: "bg-green-400",
        error: "bg-red-500",
    };
    return (
        <div className="flex items-center gap-2 mt-auto pt-4">
            <span className={`w-2 h-2 rounded-full ${colors[jobState]}`} />
            <span className="text-gray-500 text-xs capitalize">{jobState}</span>
        </div>
    );
}

function App() {
    const presets = useStore(s => s.presets);
    const selectPreset = useStore(s => s.selectPreset);
    const loadPresets = useStore(s => s.loadPresets);

    const [sidebarWidth, setSidebarWidth] = useState(SIDEBAR_DEFAULT);
    const dragging = useRef(false);
    const startX = useRef(0);
    const startWidth = useRef(0);

    useEffect(() => {
        loadPresets();
    }, []);

    useEffect(() => {
        if (presets.length > 0) selectPreset(presets[0]);
    }, [presets]);

    const onMouseDown = useCallback(
        (e: React.MouseEvent) => {
            dragging.current = true;
            startX.current = e.clientX;
            startWidth.current = sidebarWidth;
            e.preventDefault();
        },
        [sidebarWidth],
    );

    useEffect(() => {
        const onMouseMove = (e: MouseEvent) => {
            if (!dragging.current) return;
            const delta = e.clientX - startX.current;
            setSidebarWidth(
                Math.min(
                    SIDEBAR_MAX,
                    Math.max(SIDEBAR_MIN, startWidth.current + delta),
                ),
            );
        };
        const onMouseUp = () => {
            dragging.current = false;
        };
        window.addEventListener("mousemove", onMouseMove);
        window.addEventListener("mouseup", onMouseUp);
        return () => {
            window.removeEventListener("mousemove", onMouseMove);
            window.removeEventListener("mouseup", onMouseUp);
        };
    }, []);

    return (
        <div className="flex h-screen overflow-hidden bg-gray-900 text-white">
            {/* Sidebar */}
            <aside
                className="shrink-0 flex flex-col p-4 border-r border-gray-700 overflow-y-auto"
                style={{ width: sidebarWidth }}
            >
                <h1 className="text-white font-bold text-lg mb-4 tracking-tight">
                    Wind Simulator
                </h1>

                <div className="mb-4">
                    <div className="text-gray-400 text-xs uppercase tracking-wide mb-2">
                        Presets
                    </div>
                    <PresetSelector />
                </div>

                <hr className="border-gray-700 my-2" />

                <div className="mb-4">
                    <div className="text-gray-400 text-xs uppercase tracking-wide mb-3">
                        Wind Parameters
                    </div>
                    <WindControls />
                </div>

                <hr className="border-gray-700 my-2" />

                <div className="flex-1 min-h-0 flex flex-col">
                    <div className="text-gray-400 text-xs uppercase tracking-wide mb-2">
                        Comfort Map
                    </div>
                    <div
                        className="flex-1 min-h-0 rounded overflow-hidden"
                        style={{ minHeight: 200 }}
                    >
                        <ComfortMap />
                    </div>
                </div>

                <JobDot />
            </aside>

            {/* Resize handle */}
            <div
                className="w-1 shrink-0 cursor-col-resize bg-gray-700 hover:bg-blue-500 transition-colors"
                onMouseDown={onMouseDown}
            />

            {/* Main panel */}
            <main className="flex-1 min-h-0 min-w-0">
                <Viewport3D />
            </main>
        </div>
    );
}

export default App;
