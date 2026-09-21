export default function Footer({ onModel }: { onModel: (e: React.MouseEvent) => void }) {
  return (
    <footer className="bottom">
      <div className="container">
        <p>
          Apoyo al triage de melanoma en dermatoscopía, con fines educativos. No es un
          diagnóstico, no detecta otros cánceres de piel, no sustituye a un dermatólogo y no tiene
          aprobación regulatoria. Las imágenes no se almacenan.{" "}
          <a href="/modelo" onClick={onModel}>
            Ficha del modelo, desempeño y limitaciones
          </a>
          .
        </p>
        <p className="credit">
          <img
            src="/eo.svg"
            alt=""
            width={18}
            height={18}
            onError={(e) => {
              e.currentTarget.style.display = "none";
            }}
          />
          <a href="https://edgar-ontiveros.com/" rel="author">
            Edgar Eduardo Ontiveros Lara
          </a>
        </p>
      </div>
    </footer>
  );
}
