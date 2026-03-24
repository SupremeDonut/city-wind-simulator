import { create } from 'zustand'
import { fetchPresets, predict, simulateLBM } from '../api/client'
import type { Preset, WindParams, SimulationResult, JobState } from '../types'

interface AppState {
  presets: Preset[]
  selectedPreset: Preset | null
  windParams: WindParams
  result: SimulationResult | null
  jobState: JobState
  dirty: boolean
  simProgress: number
  loadPresets: () => Promise<void>
  selectPreset: (p: Preset) => void
  setWindDirection: (deg: number) => void
  setWindSpeed: (speed: number) => void
  setRoughness: (z0: number) => void
  runPredict: () => Promise<void>
  runSimulation: () => Promise<void>
  releaseVelocityData: () => void
}

export const useStore = create<AppState>((set, get) => ({
  presets: [],
  selectedPreset: null,
  windParams: { direction: 270, speed: 5, roughness: 0.3 },
  result: null,
  jobState: 'idle',
  dirty: false,
  simProgress: 0,

  loadPresets: async () => {
    const presets = await fetchPresets()
    set({ presets })
  },

  selectPreset: (p) => {
    set({ selectedPreset: p, dirty: false })
    get().runPredict()
  },

  setWindDirection: (deg) => {
    set(s => ({ windParams: { ...s.windParams, direction: deg }, dirty: true }))
  },

  setWindSpeed: (speed) => {
    set(s => ({ windParams: { ...s.windParams, speed }, dirty: true }))
  },

  setRoughness: (z0) => {
    set(s => ({ windParams: { ...s.windParams, roughness: z0 }, dirty: true }))
  },

  runPredict: async () => {
    const { selectedPreset, windParams } = get()
    if (!selectedPreset) return
    set({ jobState: 'running', dirty: false })
    try {
      const result = await predict(selectedPreset, windParams)
      set({ result, jobState: 'done' })
    } catch {
      set({ jobState: 'error' })
    }
  },

  runSimulation: async () => {
    const { selectedPreset, windParams } = get()
    if (!selectedPreset) return
    set({ jobState: 'running', simProgress: 0 })
    try {
      await simulateLBM(selectedPreset, windParams, {
        onProgress: (value) => set({ simProgress: value }),
        onSnapshot: (result) => set({ result }),
        onDone: () => set({ jobState: 'done' }),
        onError: () => set({ jobState: 'error' }),
      })
    } catch {
      set({ jobState: 'error' })
    }
  },

  releaseVelocityData: () => {
    const { result } = get()
    if (!result) return
    set({
      result: {
        ...result,
        velocityField: { ...result.velocityField, data: new Float32Array(0) },
      },
    })
  },
}))
