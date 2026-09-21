import { useId, useRef, useState } from "react";

const ACCEPT = "image/jpeg,image/png,image/webp";

/** Zona de arrastrar o seleccionar archivo. No valida nada del modelo: la API decide. */
export default function Uploader({
  onFile,
  disabled,
}: {
  onFile: (file: File) => void;
  disabled: boolean;
}) {
  const [active, setActive] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const id = useId();

  const pick = (files: FileList | null) => {
    const f = files?.[0];
    if (f) onFile(f);
  };

  return (
    <div
      className={`dropzone${active ? " active" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setActive(true);
      }}
      onDragLeave={() => setActive(false)}
      onDrop={(e) => {
        e.preventDefault();
        setActive(false);
        if (!disabled) pick(e.dataTransfer.files);
      }}
    >
      <p>
        <strong>Arrastre una dermatoscopía aquí</strong> o seleccione un archivo.
      </p>
      <p className="small muted">JPEG, PNG o WebP; hasta 10 MB. Puede ser la foto original sin recortar.</p>
      <label htmlFor={id} className="sr-only">
        Seleccionar imagen
      </label>
      <input
        id={id}
        ref={input}
        type="file"
        accept={ACCEPT}
        disabled={disabled}
        onChange={(e) => {
          pick(e.target.files);
          e.target.value = "";
        }}
      />
    </div>
  );
}
