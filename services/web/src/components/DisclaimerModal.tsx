import { useEffect, useRef, useState } from "react";

const KEY = "melanoma-triage:aviso-aceptado";

/** Aviso al cargar la aplicación. Se recuerda en sessionStorage (cada sesión lo vuelve a
 * mostrar). El texto es el `disclaimer` de la API; mientras no llega, un texto mínimo. */
export default function DisclaimerModal({ disclaimer }: { disclaimer: string | null }) {
  const [accepted, setAccepted] = useState<boolean>(() => {
    try {
      return window.sessionStorage.getItem(KEY) === "1";
    } catch {
      return false;
    }
  });
  const button = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!accepted) button.current?.focus();
  }, [accepted]);

  if (accepted) return null;

  const accept = () => {
    try {
      window.sessionStorage.setItem(KEY, "1");
    } catch {
      /* sin almacenamiento: el modal se muestra igual la próxima vez */
    }
    setAccepted(true);
  };

  return (
    <div className="modal-backdrop" role="presentation">
      <div className="modal" role="dialog" aria-modal="true" aria-labelledby="aviso-titulo">
        <h2 id="aviso-titulo">Antes de usar esta herramienta</h2>
        <p>
          {disclaimer ??
            "Herramienta de apoyo al triage de melanoma con fines educativos. No es un diagnóstico y no sustituye la valoración de un dermatólogo."}
        </p>
        <button ref={button} type="button" onClick={accept}>
          Entiendo
        </button>
      </div>
    </div>
  );
}
