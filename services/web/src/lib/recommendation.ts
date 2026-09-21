// Texto de la recomendación de priorización (F7.2). El texto es fijo; los números se interpolan
// desde la respuesta de /predict (operating_point) y de /model-info (metrics). Nada aquí
// conoce el umbral ni la prevalencia: si faltan, se dice «n/d» en vez de inventar.

import { oneInN, per100 } from "./format";

export interface RecommendationInput {
  refer: boolean;
  /** sensibilidad medida en el test (operating_point.sensitivity_test) */
  sensitivityTest: number | null;
  /** especificidad medida en el test (operating_point.specificity_test) */
  specificityTest: number | null;
  /** valor predictivo negativo en τ95 (metrics.tau_95.npv), para «1 de cada N» */
  npvTest: number | null;
}

export interface Recommendation {
  headline: string;
  body: string;
  refer: boolean;
}

export function recommendation(input: RecommendationInput): Recommendation {
  if (input.refer) {
    const sens = input.sensitivityTest === null ? "n/d" : per100(input.sensitivityTest, 1);
    const benignReferred =
      input.specificityTest === null ? "n/d" : per100(1 - input.specificityTest, 0);
    return {
      refer: true,
      headline: "Referir a dermatología.",
      body:
        `Con este umbral el sistema detecta ${sens} de cada 100 melanomas y refiere también ` +
        `a ${benignReferred} de cada 100 lesiones benignas. Una referencia no es un diagnóstico.`,
    };
  }
  const missRate = input.npvTest === null ? null : 1 - input.npvTest;
  const oneIn = missRate === null ? "n/d" : oneInN(missRate);
  return {
    refer: false,
    headline: "Por debajo del umbral de referencia.",
    body:
      `Aproximadamente ${oneIn} lesiones en este grupo es melanoma. ` +
      "Ante duda clínica, referir de todos modos.",
  };
}
