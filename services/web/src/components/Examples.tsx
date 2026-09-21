import { useEffect, useState } from "react";

export interface Example {
  image_id: string;
  file: string;
  truth: "melanoma" | "benigna";
  note: string;
}

interface ExamplesFile {
  attribution: string;
  items: Example[];
}

/** Cuatro imágenes de validación de ISIC 2020 (CC-BY-NC), listadas en
 * public/examples/examples.json con su image_id y su diagnóstico confirmado. */
export default function Examples({
  onPick,
  disabled,
}: {
  onPick: (ex: Example, blob: Blob) => void;
  disabled: boolean;
}) {
  const [data, setData] = useState<ExamplesFile | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/examples/examples.json")
      .then((r) => (r.ok ? (r.json() as Promise<ExamplesFile>) : Promise.reject(new Error(String(r.status)))))
      .then(setData)
      .catch(() => setError("No se pudieron cargar los ejemplos."));
  }, []);

  const pick = async (ex: Example) => {
    try {
      const r = await fetch(`/${ex.file}`);
      if (!r.ok) throw new Error(String(r.status));
      onPick(ex, await r.blob());
    } catch {
      setError("No se pudo cargar la imagen de ejemplo.");
    }
  };

  if (error) return <p className="small muted">{error}</p>;
  if (!data) return null;

  return (
    <div>
      <h3>O pruebe con un ejemplo</h3>
      <div className="examples">
        {data.items.map((ex) => (
          <button
            key={ex.image_id}
            type="button"
            className="example"
            disabled={disabled}
            onClick={() => pick(ex)}
            title={ex.note}
          >
            <img src={`/${ex.file}`} alt={`Dermatoscopía ${ex.image_id}, lesión ${ex.truth} confirmada`} />
            <span>
              <code>{ex.image_id}</code>
              <br />
              {ex.truth === "melanoma" ? "Melanoma confirmado" : "Benigna confirmada"}
            </span>
          </button>
        ))}
      </div>
      <p className="small muted">{data.attribution}</p>
    </div>
  );
}
