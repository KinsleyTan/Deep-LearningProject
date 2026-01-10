"use client";

import { useEffect, useRef , useState} from "react";
import * as THREE from "three";
import { FACEMESH_TRIANGLE } from "@/lib/facemesh_triangles";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import Hero2Dto3D from "@/components/Hero";
import { Check, ChevronsUpDown } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { API_BASE_URL } from "@/lib/config";
import { Badge } from "@/components/ui/badge"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
type Landmark = {
  x: number;
  y: number;
  z: number;
};
import Navbar from "@/components/navbar";
// import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

type RenderMode = "mesh" | "wireframe" | "points";
const renderModes = [
  { value: "mesh", label: "Mesh" },
  { value: "wireframe", label: "Wireframe" },
  { value: "points", label: "Points" },
] as const

export default function Home() {

  const mountRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);

  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const meshRef = useRef<THREE.Mesh | null>(null);

  const [isCameraOn, setIsCameraOn] = useState(false);
  const streamRef = useRef<MediaStream | null>(null);
  const [renderMode, setRenderMode] = useState<RenderMode>("mesh");
  const pointsRef = useRef<THREE.Points | null>(null);
  const [open, setOpen] = useState(false)
  const [isopen, setIsopen] = useState(false);


  useEffect(() => {
    if (typeof window === "undefined") return;

    const resize = () => {
      if (!mountRef.current || !cameraRef.current || !rendererRef.current) return;

      const { clientWidth, clientHeight } = mountRef.current;

      rendererRef.current.setSize(clientWidth, clientHeight, false);

      cameraRef.current.aspect = clientWidth / clientHeight;
      cameraRef.current.updateProjectionMatrix();
    };

    resize();
    window.addEventListener("resize", resize);

    return () => window.removeEventListener("resize", resize);
  }, []);

  /* =========================
     1️⃣ INIT THREE.JS
  ========================= */
  useEffect(() => {
    if (!mountRef.current) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.z = 1.5;
    camera.lookAt(0, 0, 0);
    
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.domElement.style.width = "100%";
    renderer.domElement.style.height = "100%";
    renderer.domElement.style.display = "block";

    mountRef.current.appendChild(renderer.domElement);

    sceneRef.current = scene;
    cameraRef.current = camera;
    rendererRef.current = renderer;

    // 🔹 Persistent point cloud
    const geometry = new THREE.BufferGeometry();
    const dummyPositions = new Float32Array(478 * 3);
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(dummyPositions, 3)
    );
    geometry.setIndex(FACEMESH_TRIANGLE);

    // === MESH MATERIAL ===
    const meshMaterial = new THREE.MeshStandardMaterial({
      color: 0xffe0bd,
      roughness: 0.6,
      metalness: 0.0,
      side: THREE.DoubleSide,
    });

    // === POINT MATERIAL ===
    const pointsMaterial = new THREE.PointsMaterial({
      color: 0xffe0bd,
      size: 0.015,            // 👈 CRITICAL
      sizeAttenuation: true,
      depthTest: false,
    });
    
    const material = new THREE.MeshStandardMaterial({
      color: 0x00ffcc,        // skin-like
      roughness: 0.6,         // softer highlights
      metalness: 0.0,         // skin is not metal
      side: THREE.DoubleSide,
      flatShading: false,
    });
    material.wireframe = true;


    const faceMesh = new THREE.Mesh(geometry, meshMaterial)as THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>;
    const facePoints = new THREE.Points(geometry, pointsMaterial);
    scene.add(faceMesh);

    meshRef.current = faceMesh;
    pointsRef.current = facePoints;
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
    keyLight.position.set(1, 1, 2);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
    fillLight.position.set(-1, 0.5, 1);
    scene.add(fillLight);

    const ambient = new THREE.AmbientLight(0xffffff, 0.3);
    scene.add(ambient);

    const rimLight = new THREE.DirectionalLight(0xffffff, 0.6);
    rimLight.position.set(0, 0, -2);
    scene.add(rimLight);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      if (rendererRef.current) {
        rendererRef.current.dispose();
        mountRef.current?.removeChild(rendererRef.current.domElement);
      }
    };
  }, []);

  /* =========================
     2️⃣ INIT WEBCAM
  ========================= */

  async function startCamera() {
    if (isCameraOn) return;

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });

    streamRef.current = stream;

    if (videoRef.current) {
      videoRef.current.srcObject = stream;
    }

    setIsCameraOn(true);
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setIsCameraOn(false);
  }

  /* =========================
     3️⃣ LIVE INFERENCE
  ========================= */
  const captureAndInfer = async () => {
    if (!videoRef.current || !meshRef.current) return;
    if (videoRef.current.videoWidth === 0) return;

    // 🎯 Capture frame
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(videoRef.current, 0, 0);

    const blob = await new Promise<Blob>((resolve) =>
      canvas.toBlob((b) => resolve(b!), "image/jpeg")
    );

    // 📡 Send to backend
    const formData = new FormData();
    formData.append("file", blob);

    const controller = new AbortController();
    setTimeout(() => controller.abort(), 3000);
    if (!API_BASE_URL) {
      console.error("❌ API_BASE_URL is missing at build time");
      return;
    }

    let res;
    try {
      res = await fetch(`${API_BASE_URL}/infer`, {
        method: "POST",
        body: formData,
        signal: controller.signal
      });
    } catch (e) {
      console.warn("Fetch failed:", e);
      return;
    }

    const data = await res.json();
    if (!data.landmarks || data.landmarks.length === 0) return;

    // 🧮 Convert to Three.js coords
    const positions = new Float32Array(
      data.landmarks.flatMap((p: Landmark) => [
        (p.x - 0.5) * 2,
        -(p.y - 0.5) * 2,
        -p.z * 2,
      ])
    );

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(FACEMESH_TRIANGLE);
    geometry.computeVertexNormals();
    
    meshRef.current!.geometry.dispose();
    pointsRef.current!.geometry.dispose();

    meshRef.current!.geometry = geometry;
    pointsRef.current!.geometry = geometry;

    geometry.computeVertexNormals();
    geometry.computeBoundingBox();

    const box = geometry.boundingBox!;
    const center = new THREE.Vector3();
    box.getCenter(center);

    geometry.translate(-center.x, -center.y, -center.z)

    // 2️⃣ Normalize scale
    const size = new THREE.Vector3()
    box.getSize(size)

    const scale = 1 / Math.max(size.x, size.y);
    meshRef.current.scale.setScalar(scale);
    pointsRef.current!.scale.setScalar(scale);

    meshRef.current.rotation.set(0, 0, 0)
    console.log("EXP RAW:", data.expression);
    if (data.expression) {
    // Handle both nested and flat arrays
    const expr = Array.isArray(data.expression[0])
      ? data.expression[0]
      : data.expression;

    // Pick low-MSE parameters
    const lowerLip = expr[3];   // lower lip out
    const upperLipInv = -expr[5]; // invert upper lip inwards

    // Average = raw score
    const mouthRaw = (lowerLip + upperLipInv) / 2;

    // Threshold — tweak as needed
    const open = mouthRaw > 0.00125;

    setIsopen(open);

    console.log("EXP RAW:", expr);
    console.log({
      mouthRaw: mouthRaw.toFixed(5),
      open
    });
  }


  };

  useEffect(() => {
    const id = setInterval(captureAndInfer, 150);
    return () => clearInterval(id);
  }, []);


    useEffect(() => {
    if (!sceneRef.current) return;

    const scene = sceneRef.current;

    if (meshRef.current) scene.remove(meshRef.current);
    if (pointsRef.current) scene.remove(pointsRef.current);

    if (renderMode === "mesh") {
      (
        meshRef.current!.material as THREE.MeshStandardMaterial
      ).wireframe = false;
      scene.add(meshRef.current!);
    }

    if (renderMode === "wireframe") {
      (
        meshRef.current!.material as THREE.MeshStandardMaterial
      ).wireframe = true;
      scene.add(meshRef.current!);
    }

    if (renderMode === "points") {
      scene.add(pointsRef.current!);
    }
  }, [renderMode]);

  return (
    <div>
    <Navbar></Navbar>
    <div className="w-full h-screen overflow-x-hidden" id = "hero">
      <div>
        <Hero2Dto3D></Hero2Dto3D>
      </div>
    </div>
    <div className="w-full h-screen overflow-x-hidden">
    <section id="inference"
  className="w-full h-screen overflow-x-hidden flex items-center justify-center bg-background px-6 pt-10"
>
  <Card className="w-full max-w-7xl h-[80vh] flex flex-col">
    <CardHeader className="flex flex-row items-center justify-between border-b">
      <CardTitle>Live 2D → 3D Inference</CardTitle>
      <div className="flex gap-2">
        {!isopen ? (
          <Badge variant="secondary" >
            close mouth
          </Badge>
        ) : (
          <Badge variant="secondary"
          className="bg-blue-500 text-white dark:bg-blue-600 px-2">
            open mouth
          </Badge>
        )}
        <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
              <Button
                variant="outline"
                role="combobox"
                aria-expanded={open}
                className="w-[180px] justify-between"
              >
                {renderModes.find(m => m.value === renderMode)?.label ?? "Render mode"}
                <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
              </Button>
            </PopoverTrigger>

            <PopoverContent className="w-[180px] p-0">
              <Command>
                <CommandInput placeholder="Search mode..." />
                <CommandList>
                  <CommandEmpty>No mode found.</CommandEmpty>
                  <CommandGroup>
                    {renderModes.map((mode) => (
                      <CommandItem
                        key={mode.value}
                        value={mode.value}
                        onSelect={(value) => {
                          setRenderMode(value as RenderMode)
                          setOpen(false)
                        }}
                      >
                        {mode.label}
                        <Check
                          className={cn(
                            "ml-auto h-4 w-4",
                            renderMode === mode.value ? "opacity-100" : "opacity-0"
                          )}
                        />
                      </CommandItem>
                    ))}
                  </CommandGroup>
                </CommandList>
              </Command>
            </PopoverContent>
          </Popover>
        {!isCameraOn ? (
          <Button size="lg" onClick={startCamera}>
            Turn on Camera
          </Button>
        ) : (
          <Button size="lg" variant="destructive" onClick={stopCamera}>
            Stop Camera
          </Button>
        )}
      </div>
    </CardHeader>

    <CardContent className=" flex flex-1 gap-4 p-4">
      
      <Card className="h-full w-1/2 overflow-hidden flex flex-col">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-muted-foreground">
            Camera Input
          </CardTitle>
        </CardHeader>

        <CardContent className="relative flex-1 p-0 overflow-hidden">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="absolute inset-0 w-full h-full object-contain "
          />
        </CardContent>
      </Card>

      <Card className="relative w-1/2 overflow-hidden">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-muted-foreground">
            3D Mesh Output
          </CardTitle>
        </CardHeader>

        <CardContent className="flex-1 p-0">
          <div
            ref={mountRef}
            className="absolute inset-0"
          />
        </CardContent>
      </Card>

    </CardContent>
    
    <CardFooter className="border-t text-xs text-muted-foreground">
      Real-time face landmarks → 3D mesh inference
    </CardFooter>

  </Card>
</section>

    </div>

    </div>
  );
}