import { describe, expect, it } from "vitest";

import { recommendation } from "./recommendation";

const test = {
  sensitivityTest: 0.9651162790697675,
  specificityTest: 0.4283778552071235,
  npvTest: 0.9986462093862816,
};

describe("recommendation", () => {
  it("por encima del umbral: texto completo con números interpolados", () => {
    const r = recommendation({ refer: true, ...test });
    expect(r.refer).toBe(true);
    expect(r.headline).toBe("Referir a dermatología.");
    expect(r.body).toBe(
      "Con este umbral el sistema detecta 96,5 de cada 100 melanomas y refiere también a 57 de cada 100 lesiones benignas. Una referencia no es un diagnóstico.",
    );
  });

  it("por debajo del umbral: 1 de cada N desde el VPN", () => {
    const r = recommendation({ refer: false, ...test });
    expect(r.refer).toBe(false);
    expect(r.headline).toBe("Por debajo del umbral de referencia.");
    expect(r.body).toBe(
      "Aproximadamente 1 de cada 750 lesiones en este grupo es melanoma. Ante duda clínica, referir de todos modos.",
    );
  });

  it("sin métricas no inventa números", () => {
    const r = recommendation({ refer: true, sensitivityTest: null, specificityTest: null, npvTest: null });
    expect(r.body).toContain("detecta n/d de cada 100 melanomas");
    expect(r.body).toContain("a n/d de cada 100 lesiones benignas");
    const below = recommendation({ refer: false, sensitivityTest: null, specificityTest: null, npvTest: null });
    expect(below.body).toContain("Aproximadamente n/d lesiones");
  });
});
