"use client";

import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

export function ModeToggle({ value, onChange }: any) {
  return (
    <ToggleGroup
      type="single"
      value={value}
      onValueChange={onChange}
      className="mb-4"
    >
      <ToggleGroupItem value="landmarks">Landmarks</ToggleGroupItem>
      <ToggleGroupItem value="mesh">3D Mesh</ToggleGroupItem>
    </ToggleGroup>
  );
}
