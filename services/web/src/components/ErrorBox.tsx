import { ApiError } from "../api";

/** Mensajes de error para el usuario (F7.4). Nunca un stack trace. */
export function messageFor(e: unknown): { text: string; retry: boolean } {
  if (e instanceof ApiError) {
    if (e.status === 0) return { text: "El servicio no está disponible.", retry: true };
    if (e.status === 415) return { text: "Formato no aceptado. Use JPEG, PNG o WebP.", retry: false };
    if (e.status === 413) return { text: "La imagen supera 10 MB.", retry: false };
    if (e.status === 422) return { text: e.message, retry: false };
    if (e.status >= 500) return { text: "El servicio no está disponible.", retry: true };
    return { text: e.message, retry: false };
  }
  return { text: "Ocurrió un error inesperado.", retry: true };
}

export default function ErrorBox({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const m = messageFor(error);
  return (
    <div className="error" role="alert">
      <p>{m.text}</p>
      {m.retry && onRetry && (
        <button type="button" className="secondary" onClick={onRetry}>
          Reintentar
        </button>
      )}
    </div>
  );
}
