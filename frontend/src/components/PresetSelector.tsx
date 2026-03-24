import { useStore } from "../store/store";

export function PresetSelector() {
    const presets = useStore(s => s.presets);
    const selectedPreset = useStore(s => s.selectedPreset);
    const selectPreset = useStore(s => s.selectPreset);

    return (
        <div className="flex gap-3 overflow-x-auto pb-2">
            {presets.map(p => (
                <button
                    key={p.id}
                    onClick={() => selectPreset(p)}
                    className={[
                        "w-48 shrink-0 p-4 rounded-lg border text-left cursor-pointer bg-gray-800",
                        selectedPreset?.id === p.id
                            ? "border-blue-400 ring-2 ring-blue-400"
                            : "border-gray-600 hover:border-gray-400",
                    ].join(" ")}
                >
                    <div className="text-white font-semibold text-sm">
                        {p.name}
                    </div>
                    <div className="text-gray-400 text-sm mt-1">
                        {p.description}
                    </div>
                    <div className="font-mono text-xs text-gray-500 mt-2">
                        {p.domainSize[0]}×{p.domainSize[1]}×{p.domainSize[2]}m @{" "}
                        {p.resolution}m
                    </div>
                </button>
            ))}
        </div>
    );
}
