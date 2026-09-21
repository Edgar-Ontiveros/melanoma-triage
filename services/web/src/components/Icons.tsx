/** Íconos propios, inline SVG (sin marcas ajenas ni recursos externos). */

/** Un círculo con un punto descentrado: una lesión bajo el dermatoscopio. */
export function LesionIcon({ size = 26 }: { size?: number }) {
  return (
    <svg viewBox="0 0 32 32" width={size} height={size} aria-hidden="true" focusable="false">
      <circle cx="16" cy="16" r="13" fill="#fff" stroke="currentColor" strokeWidth="2.5" />
      <circle cx="19" cy="14" r="4.5" fill="currentColor" />
    </svg>
  );
}

/** Flecha de subida dentro de un círculo. */
export function UploadIcon({ size = 40 }: { size?: number }) {
  return (
    <svg viewBox="0 0 40 40" width={size} height={size} aria-hidden="true" focusable="false">
      <circle cx="20" cy="20" r="18" fill="none" stroke="currentColor" strokeWidth="2" />
      <path
        d="M20 27V13m0 0-6 6m6-6 6 6"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
