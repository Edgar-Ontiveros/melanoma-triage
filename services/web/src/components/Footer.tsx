/** Monograma EO· (public/eo.svg) inline para que herede el color del texto del pie. */
function EoMonogram() {
  return (
    <svg viewBox="0 0 32 32" width={18} height={18} aria-hidden="true" focusable="false">
      <g stroke="currentColor" strokeWidth="3.6" fill="none">
        <path d="M11.4 7.8H4.8v16.4h6.6M4.8 16h5.6" />
        <circle cx="22" cy="16" r="7.2" />
      </g>
      <circle cx="29.4" cy="27.6" r="2.4" fill="currentColor" />
    </svg>
  );
}

export default function Footer({
  onModel,
  modelVersion,
}: {
  onModel: (e: React.MouseEvent) => void;
  modelVersion: string | null;
}) {
  return (
    <footer className="bottom">
      <div className="container">
        <p className="legal">
          Apoyo a la priorización de lesiones con sospecha de melanoma en dermatoscopía, con fines educativos. No es un
          diagnóstico, no detecta otros cánceres de piel, no sustituye a un dermatólogo y no tiene
          aprobación regulatoria. Las imágenes no se almacenan.{" "}
          <a href="/modelo" onClick={onModel}>
            Ficha del modelo, desempeño y limitaciones
          </a>
          .
        </p>
        <div className="row">
          <p className="credit">
            <EoMonogram />
            <a href="https://edgar-ontiveros.com/" rel="author">
              Edgar Eduardo Ontiveros Lara
            </a>
          </p>
          {modelVersion && (
            <span className="version" title="Versión del modelo servido">
              {modelVersion}
            </span>
          )}
        </div>
      </div>
    </footer>
  );
}
