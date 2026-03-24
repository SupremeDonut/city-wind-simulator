export interface Preset {
  id: string
  name: string
  description: string
  glbPath: string
  domainSize: [number, number, number]  // physical size in metres [x, y, z]
  resolution: number                    // metres per grid cell
}

export interface WindParams {
  direction: number   // degrees, 0=north, clockwise
  speed: number       // m/s
  roughness: number   // z0 roughness length in metres
}

export interface VelocityField {
  data: Float32Array
  shape: [number, number, number]
  min: number
  max: number
  domain: [number, number, number]   // physical size [x, y, z] in metres
}

export interface ComfortMap {
  data: Float32Array  // pedestrian-level speed, indexed y*Nx+x
  shape: [number, number]
}

export interface SimulationResult {
  velocityField: VelocityField
  comfortMap: ComfortMap
}

export type JobState = 'idle' | 'running' | 'done' | 'error'
