"use client";

import Image from "next/image";
import { FlipWords } from "@/components/ui/flip-words";
import { Button } from "@/components/ui/button";
import { motion,   } from "framer-motion";
import { useEffect, useState } from "react";
import {
  DraggableCardBody,
  DraggableCardContainer,
} from "@/components/ui/draggable-card";

export default function Hero2Dto3D() {
  const items = [
    {
      title: "2d to 3d",
      image:
        "./image-to-3d.jpg",
      className: "absolute top-10 left-[20%] rotate-[-5deg]",
    },
    {
      title: "Social Media filters",
      image:
        "./social-media-filter.jpg",
      className: "absolute top-40 left-[25%] rotate-[-7deg]",
    },
    {
      title: "Avatars and Vtubers",
      image:
        "./apple-avatars.jpg",
      className: "absolute top-40 left-[25%] rotate-[-7deg]",
    },
]
  const words = [
    "VTuber Rigging",
    "Social Media Filters",
    "Face Mesh Tracking",
    "AR Experiences",
    "Real-time Avatars",
  ];

  const images = [
  "/hero/Screenshot (315).png",
  "/hero/Screenshot (316).png",
//   "/hero/step-3-mesh.png",
//   "/hero/step-4-rig.png",
    ];

    const [index, setIndex] = useState(0);

    useEffect(() => {
        const id = setInterval(() => {
        setIndex((prev) => (prev + 1) % images.length);
        }, 10_000);
        return () => clearInterval(id);
    }, []);

  return (
    <section className="w-screen px-0 h-7xl flex items-center overflow-hidden bg-background">
      {/* Grid Layout */}
      <div className="mx-auto max-w-7xl w-full grid grid-cols-1 md:grid-cols-5 gap-6 px-0 lg:px-2">

        {/* LEFT – Text */}
        <div className="h-full md:col-span-3 lg:col-span-3 flex flex-col justify-end p-2  space-y-6">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-4xl md:text-5xl font-bold leading-tight"
          >
            Turn <span className="text-primary">2D</span> into
            <br />
            <FlipWords words={words} className="text-primary" />
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-muted-foreground max-w-md"
          >
            Convert landmarks and images into expressive 3D meshes for real-time
            avatars, filters, and interactive applications.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35 }}
          >
            <a href = "#inference">
            <Button size="lg" className="rounded-xl">
              Try Now
            </Button>
            </a>
          </motion.div>
        </div>

        {/* RIGHT – Visual */}
        <div className="md:col-span-2 flex items-center justify-center aspect-square w-full h-full">
            <DraggableCardContainer className="relative flex min-h-screen w-full items-center justify-center overflow-clip">
            {items.map((item) => (
                <DraggableCardBody
                    key={item.title}
                    className={item.className}
                >
                    <img
                    src={item.image}
                    alt={item.title}
                    className="pointer-events-none relative z-10 h-80 w-80 object-cover"
                    />
                    <h3 className="mt-4 text-center text-2xl font-bold text-neutral-700 dark:text-neutral-300">
                    {item.title}
                    </h3>
                </DraggableCardBody>
                ))}
        </DraggableCardContainer>
        </div>

      </div>

      {/* Subtle gradient overlay */}
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-background via-transparent to-transparent" />
    </section>
  );
}
