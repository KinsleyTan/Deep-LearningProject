"use client";

import * as THREE from "three";
import { useEffect, useRef } from "react";

type Props = {
  mode: "landmarks" | "mesh";
  data: any;
};

export function FaceViewer({ mode, data }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const objectRef = useRef<THREE.Object3D | null>(null);

  useEffect(() => {
    if (!mountRef.current) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f0f0f);

    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10);
    camera.position.z = 1.5;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(500, 500);
    mountRef.current.appendChild(renderer.domElement);

    scene.add(new THREE.DirectionalLight(0xffffff, 1));

    const animate = () => {
      requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    return () => renderer.dispose();
  }, []);

  useEffect(() => {
    if (!data || !mountRef.current) return;

    const scene = (mountRef.current.firstChild as any).__threeObj?.scene;

    if (objectRef.current) {
      scene.remove(objectRef.current);
    }

    // 🔴 LANDMARK MODE
    if (mode === "landmarks") {
      const positions = new Float32Array(
        data.landmarks.map((p: any) => [
          -(p.x - 0.5),
          -(p.y - 0.5),
          -p.z
        ]).flat()
      );

      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));

      const mat = new THREE.PointsMaterial({ size: 0.01, color: 0xff5555 });
      const points = new THREE.Points(geo, mat);

      scene.add(points);
      objectRef.current = points;
    }

    // 🔵 3D MESH MODE
    if (mode === "mesh") {
      const verts = data.vertices;

      // normalize
      const scale = 1 / Math.max(...verts.map((v: number[]) => Math.abs(v[0])));

      const positions = new Float32Array(
        verts.flatMap((v: number[]) => [
          -v[0] * scale,
          v[1] * scale,
          v[2] * scale
        ])
      );

      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geo.setIndex(data.faces.flat());
      geo.computeVertexNormals();

      const mat = new THREE.MeshStandardMaterial({
        color: 0xffcccc,
        wireframe: false
      });

      const mesh = new THREE.Mesh(geo, mat);
      scene.add(mesh);
      objectRef.current = mesh;
    }
  }, [data, mode]);

  return <div ref={mountRef} />;
}
