import { formatPercent, logPosition, probabilityLine } from "../lib/format";

/** Bloque 1: probabilidad calibrada en grande, tasa base y razón, barra logarítmica de 0,1 %
 * a 100 % con dos marcas (tasa base y umbral) y el valor actual como punto. Un solo color;
 * nada de semáforo. */
export default function ProbabilityBlock({
  probability,
  threshold,
  prevalence,
}: {
  probability: number;
  threshold: number;
  prevalence: number | null;
}) {
  const ticks = [1 / 1000, 1 / 100, 1 / 10, 1];
  const pos = logPosition(probability) * 100;
  return (
    <section className="card" aria-labelledby="prob-title">
      <h2 id="prob-title" className="block-title">
        1 · Probabilidad calibrada de melanoma
      </h2>
      <p className="prob-big" aria-live="polite">
        {formatPercent(probability)}
      </p>
      <p className="prob-line">
        {prevalence !== null
          ? probabilityLine(probability, prevalence).replace(/^[^—]+— /, "— ")
          : "— tasa base no disponible"}
      </p>
      <div
        className="logbar"
        role="img"
        aria-label={`Barra logarítmica de 0,1 % a 100 %: probabilidad ${formatPercent(probability)}, umbral ${formatPercent(threshold, 2)}${
          prevalence !== null ? `, tasa base ${formatPercent(prevalence)}` : ""
        }`}
      >
        <div className="fill" style={{ width: `${pos}%` }} />
        {prevalence !== null && (
          <>
            <div className="mark" style={{ left: `${logPosition(prevalence) * 100}%` }} />
            <div className="mark-label" style={{ left: `${logPosition(prevalence) * 100}%` }}>
              tasa base {formatPercent(prevalence)}
            </div>
          </>
        )}
        <div className="mark" style={{ left: `${logPosition(threshold) * 100}%` }} />
        <div className="mark-label below" style={{ left: `${logPosition(threshold) * 100}%` }}>
          umbral {formatPercent(threshold, 2)}
        </div>
        <div className="dot" style={{ left: `${pos}%` }} />
        {ticks.map((t, i) => (
          <span
            key={t}
            className={`axis${i === 0 ? " first" : i === ticks.length - 1 ? " last" : ""}`}
            style={{ left: `${logPosition(t) * 100}%` }}
          >
            {formatPercent(t, t < 1 / 100 ? 1 : 0)}
          </span>
        ))}
      </div>
      <p className="caption" style={{ marginTop: 0 }}>
        Probabilidad calibrada sobre validación: de cada 100 lesiones con esta puntuación, esta es
        la cantidad que resultó melanoma. La tasa base es la prevalencia del conjunto de prueba.
      </p>
    </section>
  );
}
