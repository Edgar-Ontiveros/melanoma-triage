// Cliente de la API (F6/F7). Todo número del modelo llega por aquí; el frontend no tiene
// umbrales, prevalencias ni métricas propias. Contrato: docs/API.md.

export const API_PREFIX = "/api";

export interface OperatingPoint {
  sensitivity_target: number;
  sensitivity_test: number | null;
  specificity_test: number | null;
}

export interface CamOut {
  status: "ok" | "no_positive_evidence";
  grid: number[][];
  png_base64: string | null;
  crop_png_base64: string;
}

export interface PredictResponse {
  model_version: string;
  probability: number;
  refer: boolean;
  threshold: number;
  operating_point: OperatingPoint;
  cam: CamOut;
  input: { width: number; height: number; format: string };
  latency_ms: number;
  timings_ms: Record<string, number>;
  disclaimer: string;
}

export interface Interval {
  point: number;
  lo: number;
  hi: number;
}

export interface Metrics {
  split: string;
  date: string;
  n_images: number;
  n_positives: number;
  n_patients: number;
  auroc: Interval;
  auprc: Interval;
  prevalence: number;
  tau_95: {
    threshold: number;
    sensitivity: number;
    specificity: number;
    ppv: number;
    npv: number;
    tp: number;
    fn: number;
    fp: number;
    tn: number;
  };
  ci: { level: number; unit: string; n_resamples: number };
}

export interface ModelInfo {
  model_version: string;
  manifest: {
    model_version: string;
    git_sha: string;
    date: string;
    kind?: string;
    checkpoint: { file: string; sha256: string };
    onnx: { opset: Record<string, number>; features_shape: number[] };
    versions: Record<string, string>;
  };
  metrics: Metrics;
  thresholds: {
    operating: string;
    tau_95: { tau: number; sensitivity_target: number; validation: { sensitivity: number; specificity: number } };
    tau_90: { tau: number; sensitivity_target: number; validation: { sensitivity: number; specificity: number } };
  };
  calibration: { method: string; a: number; b: number; fitted_on: string; n_images: number; n_positives: number };
  preprocess: { input_size: number; stage1: { long_side: number } };
  disclaimer: string;
}

export interface Limitations {
  markdown: string;
  source: string;
  n_sections: number;
}

/** Error con el código HTTP y el mensaje de la API (nunca un stack trace). */
export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function detailOf(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return "petición inválida";
  } catch {
    /* sin cuerpo JSON */
  }
  return res.statusText || `error ${res.status}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_PREFIX}${path}`, init);
  } catch {
    throw new ApiError(0, "El servicio no está disponible.");
  }
  if (!res.ok) throw new ApiError(res.status, await detailOf(res));
  return (await res.json()) as T;
}

export function predict(file: File | Blob, filename = "imagen"): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file, filename);
  return request<PredictResponse>("/predict", { method: "POST", body: form });
}

export function modelInfo(): Promise<ModelInfo> {
  return request<ModelInfo>("/model-info");
}

export function limitations(): Promise<Limitations> {
  return request<Limitations>("/limitations");
}

export function health(): Promise<{ status: string; model_version: string }> {
  return request("/health");
}
