export default function Footer({ onModel }: { onModel: (e: React.MouseEvent) => void }) {
  return (
    <footer className="bottom">
      <div className="container">
        Apoyo al triage de melanoma en dermatoscopía, con fines educativos. No es un diagnóstico,
        no detecta otros cánceres de piel, no sustituye a un dermatólogo y no tiene aprobación
        regulatoria. Las imágenes no se almacenan.{" "}
        <a href="/modelo" onClick={onModel}>
          Ficha del modelo, desempeño y limitaciones
        </a>
        .
      </div>
    </footer>
  );
}
