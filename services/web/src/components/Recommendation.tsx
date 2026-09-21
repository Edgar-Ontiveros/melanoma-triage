import type { OperatingPoint } from "../api";
import { recommendation } from "../lib/recommendation";

/** Bloque 2: recomendación de priorización. Texto fijo; números de operating_point (respuesta) y
 * del VPN del test (/model-info). */
export default function Recommendation({
  refer,
  operatingPoint,
  npvTest,
}: {
  refer: boolean;
  operatingPoint: OperatingPoint;
  npvTest: number | null;
}) {
  const r = recommendation({
    refer,
    sensitivityTest: operatingPoint.sensitivity_test,
    specificityTest: operatingPoint.specificity_test,
    npvTest,
  });
  return (
    <section className="card" aria-labelledby="rec-title">
      <h2 id="rec-title" className="block-title">
        2 · Recomendación de priorización
      </h2>
      <p className="rec-headline">{r.headline}</p>
      <p>{r.body}</p>
    </section>
  );
}
