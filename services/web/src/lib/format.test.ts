import { describe, expect, it } from "vitest";

import {
  formatDateEs,
  formatInt,
  formatPercent,
  formatRatio,
  formatSeconds,
  logPosition,
  oneInN,
  per100,
  probabilityLine,
} from "./format";

describe("formatPercent", () => {
  it("un decimal y coma decimal", () => {
    expect(formatPercent(0.0324)).toBe("3,2 %");
    expect(formatPercent(0.0155)).toBe("1,6 %"); // redondeo hacia arriba en la mitad
    expect(formatPercent(0.0004)).toBe("0,0 %");
    expect(formatPercent(1)).toBe("100,0 %");
  });
  it("no revienta con valores no finitos", () => {
    expect(formatPercent(Number.NaN)).toBe("—");
  });
});

describe("formatRatio / probabilityLine", () => {
  it("razón contra la tasa base con un decimal", () => {
    expect(formatRatio(0.032, 0.016)).toBe("2,0");
    expect(formatRatio(0.0005, 0.0164)).toBe("0,0");
    expect(formatRatio(0.5, 0.0164)).toBe("30"); // ≥ 10: sin decimales
  });
  it("frase completa como en la spec", () => {
    expect(probabilityLine(0.032, 0.016)).toBe("3,2 % — 2,0 veces la tasa base (1,6 %)");
    expect(probabilityLine(0.0164, 0.0164)).toBe("1,6 % — igual a la tasa base (1,6 %)");
  });
  it("sin tasa base no inventa", () => {
    expect(formatRatio(0.03, 0)).toBe("—");
  });
});

describe("per100 / oneInN", () => {
  it("sensibilidad y benignos referidos", () => {
    expect(per100(0.9651162790697675, 1)).toBe("96,5");
    expect(per100(1 - 0.4283778552071235, 0)).toBe("57");
  });
  it("1 de cada N con redondeo legible", () => {
    expect(oneInN(1 - 0.9986462093862816)).toBe("1 de cada 750"); // VPN del test → 1/739
    expect(oneInN(0.5)).toBe("1 de cada 2");
    expect(oneInN(0.0123)).toBe("1 de cada 80");
    expect(oneInN(0)).toBe("—");
  });
});

describe("logPosition", () => {
  it("escala logarítmica de 0,1 % a 100 %", () => {
    expect(logPosition(0.001)).toBe(0);
    expect(logPosition(1)).toBe(1);
    expect(logPosition(0.01)).toBeCloseTo(1 / 3, 6);
    expect(logPosition(0.1)).toBeCloseTo(2 / 3, 6);
    expect(logPosition(0)).toBe(0);
  });
});

describe("formatSeconds", () => {
  it("segundos con un decimal", () => {
    expect(formatSeconds(1234)).toBe("1,2 s");
  });
});

describe("formatDateEs / formatInt", () => {
  it("fecha en español y miles", () => {
    expect(formatDateEs("2026-09-18")).toBe("18 de septiembre de 2026");
    expect(formatDateEs("2026-09-21T18:19:24Z")).toBe("21 de septiembre de 2026");
    expect(formatInt(5252)).toBe("5,252");
  });
});
