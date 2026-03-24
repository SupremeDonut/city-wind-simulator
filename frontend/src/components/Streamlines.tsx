import { useEffect, useMemo, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import type { VelocityField } from "../types";
import { useStore } from "../store/store";

const TEX_SIZE = 64;
const MAX_AGE = 6.0; // seconds
const VEL_EXAGGERATION = 8.0; // visual speed-up so particles traverse a good fraction of the domain
const VEL_STRIDE = 4; // subsample the velocity field before uploading to GPU

// ---------------------------------------------------------------------------
// Shaders — GLSL 3.00 es
// ---------------------------------------------------------------------------

// Full-screen quad vertex shader (used only for the compute pass)
const COMPUTE_VERT = /* glsl */ `
void main() {
  gl_Position = vec4(position, 1.0);
}
`;

// Compute fragment shader: reads current particle state from uPosition,
// advances each particle, respawns expired/out-of-bounds ones.
// Output: RGBA = (gridX, gridY, gridZ, age)
const COMPUTE_FRAG = /* glsl */ `
precision highp float;
precision highp sampler2D;
precision highp sampler3D;

uniform sampler2D uPosition; // (gridX, gridY, gridZ, age)
uniform sampler3D uVelocity; // velocity field as a 3-D texture (rgba, .rgb = vel in m/s)
uniform float     uDelta;
uniform float     uNow;
uniform vec3      uGridSize;   // grid dimensions in cells (e.g. 125, 125, 25)
uniform vec3      uDomainSize; // physical domain in metres (e.g. 1000, 1000, 200)

out vec4 fragColor;

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

void main() {
  vec2  uv   = gl_FragCoord.xy / vec2(textureSize(uPosition, 0));
  vec4  data = texture(uPosition, uv);
  vec3  gpos = data.xyz;
  float age  = data.w;

  // Per-particle lifetime derived deterministically from UV so it matches the vertex shader
  float lifetime = ${MAX_AGE.toFixed(1)} * (0.6 + hash(uv * 13.7) * 0.8);

  vec3  normPos = clamp(gpos / uGridSize, 0.0, 1.0);
  vec3  vel_ms  = texture(uVelocity, normPos).rgb;  // m/s
  float speed   = length(vel_ms);

  bool expired = age > lifetime;
  bool oob     = any(lessThan(gpos, vec3(0.0))) || any(greaterThanEqual(gpos, uGridSize));
  bool stalled = speed < 0.01;

  if (expired || oob || stalled) {
    vec2 seed = uv + vec2(uNow * 0.017, uNow * 0.031);
    gpos = vec3(
      hash(seed)                 * uGridSize.x,
      hash(seed + vec2(1.0,0.0)) * uGridSize.y,
      hash(seed + vec2(0.0,1.0)) * uGridSize.z
    );
    age = 0.0;
  } else {
    // Convert m/s → grid-cells/s, then apply visual exaggeration
    vec3 vel_grid = vel_ms * uGridSize / uDomainSize * ${VEL_EXAGGERATION.toFixed(1)};
    gpos += vel_grid * uDelta;
    age  += uDelta;
  }

  fragColor = vec4(gpos, age);
}
`;

// Render vertex shader: reads particle position from the texture,
// converts to world space, colors by speed.
const RENDER_VERT = /* glsl */ `
precision highp float;
precision highp sampler2D;
precision highp sampler3D;

in vec2 aUV;

uniform sampler2D uPosition;
uniform sampler3D uVelocity;
uniform vec3      uGridSize;
uniform vec3      uDomainSize;

out vec3  vColor;
out float vAlpha;

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

vec3 speedToColor(float t) {
  t = clamp(t, 0.0, 1.0);
  if (t < 1.0/3.0) {
    float f = t * 3.0;         return vec3(0.0, f,       1.0 - f);
  } else if (t < 2.0/3.0) {
    float f = (t-1.0/3.0)*3.0; return vec3(f,   1.0,     0.0    );
  } else {
    float f = (t-2.0/3.0)*3.0; return vec3(1.0, 1.0 - f, 0.0    );
  }
}

void main() {
  vec4  data = texture(uPosition, aUV);
  vec3  gpos = data.xyz;
  float age  = data.w;

  float lifetime = ${MAX_AGE.toFixed(1)} * (0.6 + hash(aUV * 13.7) * 0.8);

  // Grid → world (Z-up, centred in XY)
  vec3 wpos = vec3(
    (gpos.x / uGridSize.x - 0.5) * uDomainSize.x,
    (gpos.y / uGridSize.y - 0.5) * uDomainSize.y,
    (gpos.z / uGridSize.z)        * uDomainSize.z
  );

  float speed = length(texture(uVelocity, clamp(gpos / uGridSize, 0.0, 1.0)).rgb);
  vColor = speedToColor(speed / 10.0);

  // Fade in quickly, fade out smoothly near end of life
  float lf = clamp(age / lifetime, 0.0, 1.0);
  vAlpha = smoothstep(0.0, 0.1, lf) * smoothstep(1.0, 0.8, lf);

  gl_Position  = projectionMatrix * modelViewMatrix * vec4(wpos, 1.0);
  gl_PointSize = 5.0;
}
`;

// Render fragment shader: circular soft point sprite
const RENDER_FRAG = /* glsl */ `
precision highp float;

in vec3  vColor;
in float vAlpha;

out vec4 fragColor;

void main() {
  vec2  cxy = 2.0 * gl_PointCoord - 1.0;
  float r   = dot(cxy, cxy);
  if (r > 1.0) discard;
  fragColor = vec4(vColor, vAlpha * (1.0 - r));
}
`;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeVelocityTexture(
    data: Float32Array,
    shape: [number, number, number],
) {
    const [Nx, Ny, Nz] = shape;
    // Only subsample if the grid is large (>128 on any axis).
    // Simulation data at 8m is already ~125×125×25 — no need to subsample.
    // Predict data at 2m can be ~500×500×100 — subsample by stride 4.
    const s = Math.max(Nx, Ny, Nz) > 128 ? VEL_STRIDE : 1;
    const tx = Math.ceil(Nx / s);
    const ty = Math.ceil(Ny / s);
    const tz = Math.ceil(Nz / s);

    // Subsample to RGBA32F — valid in WebGL2, linearly filterable.
    // uGridSize stays as the ORIGINAL (Nx,Ny,Nz) so particle coords are unchanged;
    // normPos = gpos/uGridSize maps [0,orig] → [0,1], sampling the same physical domain.
    const rgba = new Float32Array(tx * ty * tz * 4);
    let i = 0;
    for (let z = 0; z < Nz; z += s) {
        for (let y = 0; y < Ny; y += s) {
            for (let x = 0; x < Nx; x += s) {
                const src = (z * Ny * Nx + y * Nx + x) * 3;
                rgba[i++] = data[src];
                rgba[i++] = data[src + 1];
                rgba[i++] = data[src + 2];
                rgba[i++] = 0;
            }
        }
    }

    const tex = new THREE.Data3DTexture(rgba, tx, ty, tz);
    tex.format = THREE.RGBAFormat;
    tex.type = THREE.FloatType;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    return tex;
}

function updateVelocityTexture(
    tex: THREE.Data3DTexture,
    data: Float32Array,
    shape: [number, number, number],
) {
    const [Nx, Ny, Nz] = shape;
    const s = Math.max(Nx, Ny, Nz) > 128 ? VEL_STRIDE : 1;
    const tx = Math.ceil(Nx / s);
    const ty = Math.ceil(Ny / s);
    const tz = Math.ceil(Nz / s);

    const rgba = tex.image.data as Float32Array;
    let i = 0;
    for (let z = 0; z < Nz; z += s) {
        for (let y = 0; y < Ny; y += s) {
            for (let x = 0; x < Nx; x += s) {
                const src = (z * Ny * Nx + y * Nx + x) * 3;
                rgba[i++] = data[src];
                rgba[i++] = data[src + 1];
                rgba[i++] = data[src + 2];
                rgba[i++] = 0;
            }
        }
    }
    tex.needsUpdate = true;
}

function makeInitTexture(size: number, shape: [number, number, number]) {
    const [Nx, Ny, Nz] = shape;
    const data = new Float32Array(size * size * 4);
    for (let i = 0; i < size * size; i++) {
        data[i * 4] = Math.random() * Nx;
        data[i * 4 + 1] = Math.random() * Ny;
        data[i * 4 + 2] = Math.random() * Nz;
        data[i * 4 + 3] = Math.random() * MAX_AGE; // stagger ages so not all respawn at once
    }
    const tex = new THREE.DataTexture(
        data,
        size,
        size,
        THREE.RGBAFormat,
        THREE.FloatType,
    );
    tex.needsUpdate = true;
    return tex;
}

function makeRT(size: number) {
    return new THREE.WebGLRenderTarget(size, size, {
        format: THREE.RGBAFormat,
        type: THREE.FloatType,
        minFilter: THREE.NearestFilter, // position data — no interpolation
        magFilter: THREE.NearestFilter,
        depthBuffer: false,
    });
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function Streamlines({
    velocityField,
}: {
    velocityField: VelocityField;
}) {
    const { data, shape, domain } = velocityField;
    const { gl } = useThree();

    // Velocity field as a 3-D texture — reused across snapshots, only recreated if shape changes
    const velTexRef = useRef<THREE.Data3DTexture | null>(null);
    const velShapeRef = useRef<string>("");
    const shapeKey = shape.join(",");

    // Dispose on unmount
    useEffect(() => () => { velTexRef.current?.dispose(); }, []);

    const releaseVelocityData = useStore(s => s.releaseVelocityData);

    // Update or recreate the velocity texture when data/shape changes
    useMemo(() => {
        if (data.length === 0) return;
        if (shapeKey !== velShapeRef.current) {
            // Shape changed — need a new texture
            velTexRef.current?.dispose();
            velTexRef.current = makeVelocityTexture(data, shape);
            velShapeRef.current = shapeKey;
        } else if (velTexRef.current) {
            // Same shape — update data in place
            updateVelocityTexture(velTexRef.current, data, shape);
        } else {
            velTexRef.current = makeVelocityTexture(data, shape);
            velShapeRef.current = shapeKey;
        }
    }, [data, shape, shapeKey]);

    // Release source data from store after texture is updated so GC can reclaim it
    useEffect(() => {
        if (data.length > 0) releaseVelocityData();
    }, [data, releaseVelocityData]);

    // Two ping-pong render targets — allocated once for the lifetime of the component
    const rts = useMemo(
        () => [makeRT(TEX_SIZE), makeRT(TEX_SIZE)] as const,
        [],
    );
    useEffect(
        () => () => {
            rts[0].dispose();
            rts[1].dispose();
        },
        [rts],
    );

    // Current "source" texture for the compute pass:
    //   starts as a DataTexture (random initial positions),
    //   then alternates between the two RT textures each frame.
    const srcTexRef = useRef<THREE.Texture | null>(null);
    const writeIdxRef = useRef(0);

    // --- Compute pass objects (screen quad + orthographic camera + shader) ---
    const compute = useMemo(() => {
        const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
        const mat = new THREE.ShaderMaterial({
            glslVersion: THREE.GLSL3,
            vertexShader: COMPUTE_VERT,
            fragmentShader: COMPUTE_FRAG,
            uniforms: {
                uPosition: { value: null },
                uVelocity: { value: null },
                uDelta: { value: 0 },
                uNow: { value: 0 },
                uGridSize: { value: new THREE.Vector3() },
                uDomainSize: { value: new THREE.Vector3() },
            },
        });
        const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat);
        mesh.frustumCulled = false;
        const scene = new THREE.Scene();
        scene.add(mesh);
        return { camera, scene, mat };
    }, []);

    // --- Render material for <points> ---
    const renderMat = useMemo(
        () =>
            new THREE.ShaderMaterial({
                glslVersion: THREE.GLSL3,
                vertexShader: RENDER_VERT,
                fragmentShader: RENDER_FRAG,
                uniforms: {
                    uPosition: { value: null },
                    uVelocity: { value: null },
                    uGridSize: { value: new THREE.Vector3() },
                    uDomainSize: { value: new THREE.Vector3() },
                },
                blending: THREE.AdditiveBlending,
                depthWrite: false,
                transparent: true,
            }),
        [],
    );

    // --- Points geometry: one vertex per particle, each carrying its texel UV ---
    const pointsGeo = useMemo(() => {
        const N = TEX_SIZE * TEX_SIZE;
        const uvs = new Float32Array(N * 2);
        for (let i = 0; i < N; i++) {
            uvs[i * 2] = ((i % TEX_SIZE) + 0.5) / TEX_SIZE;
            uvs[i * 2 + 1] = (Math.floor(i / TEX_SIZE) + 0.5) / TEX_SIZE;
        }
        const g = new THREE.BufferGeometry();
        g.setAttribute("aUV", new THREE.BufferAttribute(uvs, 2));
        // Dummy positions — actual positions are fetched from the texture in the vertex shader.
        // Set a large bounding sphere so Three.js never frustum-culls the object.
        g.setAttribute(
            "position",
            new THREE.BufferAttribute(new Float32Array(N * 3), 3),
        );
        g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6);
        return g;
    }, []);

    // Reset particles only when grid shape changes (new preset / resolution)
    const prevShapeRef = useRef<string>("");
    useEffect(() => {
        const key = shape.join(",");
        if (key !== prevShapeRef.current) {
            // Shape changed — respawn all particles
            if (srcTexRef.current && 'dispose' in srcTexRef.current) {
                srcTexRef.current.dispose();
            }
            srcTexRef.current = makeInitTexture(TEX_SIZE, shape);
            writeIdxRef.current = 0;
            prevShapeRef.current = key;
        }
    }, [shape]);

    // Update uniforms whenever velocity data or domain changes (every snapshot)
    useEffect(() => {
        const velTex = velTexRef.current;
        if (!velTex) return;

        compute.mat.uniforms.uVelocity.value = velTex;
        compute.mat.uniforms.uGridSize.value.set(shape[0], shape[1], shape[2]);
        compute.mat.uniforms.uDomainSize.value.set(domain[0], domain[1], domain[2]);

        renderMat.uniforms.uVelocity.value = velTex;
        renderMat.uniforms.uGridSize.value.set(shape[0], shape[1], shape[2]);
        renderMat.uniforms.uDomainSize.value.set(domain[0], domain[1], domain[2]);
    }, [data]); // eslint-disable-line react-hooks/exhaustive-deps

    // Throttle compute pass to ~30 fps to reduce CPU/GPU load
    const computeAccum = useRef(0);
    const COMPUTE_INTERVAL = 1 / 30;

    useFrame((state, delta) => {
        if (!srcTexRef.current) return;

        computeAccum.current += delta;
        if (computeAccum.current < COMPUTE_INTERVAL) return;
        const dt = computeAccum.current;
        computeAccum.current = 0;

        const writeRT = rts[writeIdxRef.current];

        // 1. Compute pass — advance all particles on the GPU
        const cu = compute.mat.uniforms;
        cu.uPosition.value = srcTexRef.current;
        cu.uDelta.value = Math.min(dt, 0.05); // cap at 50 ms to avoid tunnelling
        cu.uNow.value = state.clock.elapsedTime;

        gl.setRenderTarget(writeRT);
        gl.render(compute.scene, compute.camera);
        gl.setRenderTarget(null);

        // 2. Point the render material at the freshly written texture
        renderMat.uniforms.uPosition.value = writeRT.texture;

        // 3. Ping-pong for next frame
        srcTexRef.current = writeRT.texture;
        writeIdxRef.current = 1 - writeIdxRef.current;
    });

    return (
        <points
            geometry={pointsGeo}
            material={renderMat}
            frustumCulled={false}
        />
    );
}
