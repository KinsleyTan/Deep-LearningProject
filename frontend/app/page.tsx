"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { FACEMESH_TRIANGLE } from "@/lib/facemesh_triangles";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";

type Landmark = {
  x: number;
  y: number;
  z: number;
};

export default function Home() {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const meshRef = useRef<THREE.Mesh | null>(null);

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
    camera.position.z = 2;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(
      mountRef.current.clientWidth,
      mountRef.current.clientHeight
    );

    mountRef.current.appendChild(renderer.domElement);

    sceneRef.current = scene;
    cameraRef.current = camera;
    rendererRef.current = renderer;

    // 🔹 Persistent point cloud
    const geometry = new THREE.BufferGeometry();
    geometry.setIndex(FACEMESH_TRIANGLE);
    
    const material = new THREE.MeshStandardMaterial({
      color: 0x00ffcc,
      side: THREE.DoubleSide,
    });

    const faceMesh = new THREE.Mesh(geometry, material);
    scene.add(faceMesh);

    meshRef.current = faceMesh;
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(0, 0, 2);
    scene.add(light);

    scene.add(new THREE.AmbientLight(0xffffff, 0.3));

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      renderer.dispose();
    };
  }, []);

  /* =========================
     2️⃣ INIT WEBCAM
  ========================= */
  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      });
  }, []);

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

    const res = await fetch("http://127.0.0.1:8000/infer", {
      method: "POST",
      body: formData,
    });

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
    
    meshRef.current.geometry.dispose();
    meshRef.current.geometry = geometry;

    geometry.computeBoundingBox();

    const box = geometry.boundingBox!;
    const size = new THREE.Vector3();
    box.getSize(size);

    const scale = 1 / Math.max(size.x, size.y);
    meshRef.current.scale.setScalar(scale);
  };

  /* =========================
     4️⃣ RUN LOOP (5 FPS)
  ========================= */
  useEffect(() => {
    const id = setInterval(captureAndInfer, 50);
    return () => clearInterval(id);
  }, []);

  return (
    <main className="w-screen h-screen flex">
      {/* Webcam (hidden or visible) */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-0.5 z-10"
        style={{ width: "50%", transform: "scaleX(-1)" }}
      />

      {/* Three.js */}
      <div
        ref={mountRef}
        className="border"
        style={{ width: "50%", height: "100vh" }}
      />
    </main>
  );
}