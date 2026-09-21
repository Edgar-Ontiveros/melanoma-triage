import type { OperatingPoint } from "../api";
import { recommendation } from "../lib/recommendation";

/** Bloque 2: recomendación de triage. Texto fijo; números de operating_point (respuesta) y
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
      <h2 id="rec-title" style={{ marginTop: 0 }}>
        2. Recomendación de triage
      </h2>
      <p className="rec-headline">{r.headline}</p>
      <p>{r.body}</p>
    </section>
  );
}
