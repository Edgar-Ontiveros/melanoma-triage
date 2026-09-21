// Formateo de probabilidades y de la barra logarítmica. Funciones puras, probadas con
// Vitest: aquí un error de redondeo cambia el significado de la pantalla.

/** Porcentaje con un decimal y coma decimal española: p = 0,0324 → "3,2 %". */
export function formatPercent(p: number, decimals = 1): string {
  if (!Number.isFinite(p)) return "—";
  const value = (p * 100).toFixed(decimals).replace(".", ",");
  return `${value} %`;
}

/** Razón contra la tasa base, con un decimal: 3,2 % entre 1,6 % → "2,0". */
export function formatRatio(p: number, base: number): string {
  if (!Number.isFinite(p) || !Number.isFinite(base) || base <= 0) return "—";
  const r = p / base;
  const decimals = r >= 10 ? 0 : 1;
  return r.toFixed(decimals).replace(".", ",");
}

/** Frase completa: "3,2 % — 2,0 veces la tasa base (1,6 %)". */
export function probabilityLine(p: number, base: number): string {
  const ratio = formatRatio(p, base);
  const veces = ratio === "1,0" ? "igual a la tasa base" : `${ratio} veces la tasa base`;
  return `${formatPercent(p)} — ${veces} (${formatPercent(base)})`;
}

/** Fracción por cada 100: sensibilidad 96,51 % → "96,5"; 1 − especificidad → benignos referidos "57". */
export function per100(x: number, decimals = 1): string {
  return (x * 100).toFixed(decimals).replace(".", ",");
}

/** "1 de cada N": inverso de una tasa, redondeado a un número «redondo» legible. */
export function oneInN(rate: number): string {
  if (!Number.isFinite(rate) || rate <= 0) return "—";
  const n = 1 / rate;
  let rounded: number;
  if (n < 10) rounded = Math.round(n);
  else if (n < 100) rounded = Math.round(n / 5) * 5;
  else if (n < 1000) rounded = Math.round(n / 50) * 50;
  else rounded = Math.round(n / 100) * 100;
  return `1 de cada ${rounded.toLocaleString("es-MX")}`;
}

export const LOG_BAR_MIN = 1 / 1000; // 0,1 %
export const LOG_BAR_MAX = 1.0; // 100 %

/** Posición (0–1) de una probabilidad en la barra logarítmica de 0,1 % a 100 %. */
export function logPosition(p: number, min = LOG_BAR_MIN, max = LOG_BAR_MAX): number {
  if (!Number.isFinite(p) || p <= min) return 0;
  if (p >= max) return 1;
  return (Math.log10(p) - Math.log10(min)) / (Math.log10(max) - Math.log10(min));
}

/** Tiempo transcurrido en segundos con un decimal: 1234 ms → "1,2 s". */
export function formatSeconds(ms: number): string {
  return `${(ms / 1000).toFixed(1).replace(".", ",")} s`;
}

const MESES = [
  "enero",
  "febrero",
  "marzo",
  "abril",
  "mayo",
  "junio",
  "julio",
  "agosto",
  "septiembre",
  "octubre",
  "noviembre",
  "diciembre",
];

/** "2026-09-18" (o ISO con hora) → "18 de septiembre de 2026". */
export function formatDateEs(iso: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
  if (!m) return iso;
  const [, y, mo, d] = m;
  return `${Number(d)} de ${MESES[Number(mo) - 1]} de ${y}`;
}

/** 5252 → "5,252" (miles con coma, como en el reporte de F4). */
export function formatInt(n: number): string {
  return n.toLocaleString("en-US");
}
