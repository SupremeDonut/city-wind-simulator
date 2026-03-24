import { Suspense, useEffect, useMemo, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import {
    OrbitControls,
    PerspectiveCamera,
    useGLTF,
    useTexture,
} from "@react-three/drei";
import * as THREE from "three";
import { useStore } from "../store/store";
import { Streamlines } from "./Streamlines";

const API_BASE = "http://localhost:8000";

const BUILDING_MATERIAL = new THREE.MeshStandardMaterial({ color: "#eee" });

function Buildings({ glbPath }: { glbPath: string }) {
    const { scene } = useGLTF(glbPath);
    const painted = useMemo(() => {
        scene.traverse(obj => {
            if ((obj as THREE.Mesh).isMesh)
                (obj as THREE.Mesh).material = BUILDING_MATERIAL;
        });
        return scene;
    }, [scene]);
    return <primitive object={painted} />;
}

function GroundGrid({ domainX, domainY }: { domainX: number; domainY: number }) {
    const gridRef = useRef<THREE.GridHelper>(null);
    // 2 m cell size; GridHelper is square so use the larger dimension
    const size = Math.max(domainX, domainY);
    const divisions = Math.round(size / 2);

    useEffect(() => {
        const grid = gridRef.current;
        if (!grid) return;
        // GridHelper material is an array [centerLineMat, gridLineMat]
        const mats = Array.isArray(grid.material) ? grid.material : [grid.material];
        mats.forEach(m => {
            (m as THREE.LineBasicMaterial).transparent = true;
            (m as THREE.LineBasicMaterial).opacity = 0.3;
            (m as THREE.LineBasicMaterial).depthWrite = false;
            m.needsUpdate = true;
        });
    }, []);

    // GridHelper is XZ plane (Y-up); rotate -90° around X to lie in XY (Z-up ground)
    // Offset 0.2 m above z=0 to avoid z-fighting with the ground plane
    return (
        <gridHelper
            ref={gridRef}
            args={[size, divisions, "#ffffff", "#ffffff"]}
            rotation={[-Math.PI / 2, 0, 0]}
            position={[0, 0, 0.2]}
        />
    );
}

function GroundMap({
    presetId,
    domainX,
    domainY,
}: {
    presetId: string;
    domainX: number;
    domainY: number;
}) {
    const texture = useTexture(`${API_BASE}/presets/${presetId}/map-texture`);
    return (
        <>
            <mesh position={[0, 0, 0]}>
                <planeGeometry args={[domainX, domainY]} />
                <meshBasicMaterial map={texture} />
            </mesh>
            <GroundGrid domainX={domainX} domainY={domainY} />
        </>
    );
}

export function Viewport3D() {
    const result = useStore(s => s.result);
    const selectedPreset = useStore(s => s.selectedPreset);

    return (
        <Canvas>
            {/* <color attach="background" args={["#a0a0a0"]} /> */}
            <PerspectiveCamera
                makeDefault
                position={[200, -300, 150]}
                up={[0, 0, 1]}

            />
            <OrbitControls target={[0, 0, 20]} />
            <ambientLight intensity={0.5} />
            <directionalLight position={[100, 100, 200]} intensity={1.0} />
            <group>
                {selectedPreset && (
                    <>
                        <Suspense fallback={null}>
                            <GroundMap
                                presetId={selectedPreset.id}
                                domainX={selectedPreset.domainSize[0]}
                                domainY={selectedPreset.domainSize[1]}
                            />
                        </Suspense>
                        <Buildings glbPath={selectedPreset.glbPath} />
                    </>
                )}
                {result && <Streamlines velocityField={result.velocityField} />}
            </group>
        </Canvas>
    );
}
