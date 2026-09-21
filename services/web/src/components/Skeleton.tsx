/** Esqueleto de carga mientras responde la API: bloques grises animados con la forma de las
 * tarjetas de resultado. */
export default function Skeleton() {
  const block = (h: number, w = "100%") => (
    <div className="skeleton" style={{ height: h, width: w, marginBottom: 12 }} aria-hidden="true" />
  );
  return (
    <div aria-busy="true" aria-label="Analizando la imagen">
      <section className="card">
        {block(12, "40%")}
        {block(48, "30%")}
        {block(16, "60%")}
        {block(10)}
      </section>
      <section className="card">
        {block(12, "40%")}
        {block(24, "50%")}
        {block(16)}
        {block(16, "80%")}
      </section>
      <section className="card">
        {block(12, "40%")}
        <div className="cam-wrap">
          <div className="skeleton cam-img" style={{ background: undefined }} />
          <div className="cam-controls">
            {block(16, "70%")}
            {block(24)}
            {block(16, "90%")}
          </div>
        </div>
      </section>
    </div>
  );
}
