import { marked } from "marked";
import { useEffect, useState } from "react";

import { ApiError, limitations, type Interval, type Limitations, type ModelInfo } from "../api";
import type { ModelInfoState } from "../App";
import ErrorBox from "../components/ErrorBox";
import { formatDateEs, formatInt, formatPercent } from "../lib/format";

const REPO = "https://github.com/Edgar-Ontiveros/melanoma-triage";

function ci(x: Interval, decimals = 3): string {
  const f = (v: number) => v.toFixed(decimals).replace(".", ",");
  return `${f(x.point)} [${f(x.lo)}, ${f(x.hi)}]`;
}

function Performance({ info }: { info: ModelInfo }) {
  const m = info.metrics;
  const t = m.tau_95;
  const rows: [string, string][] = [
    ["AUC-ROC", ci(m.auroc)],
    ["AUPRC", ci(m.auprc)],
    ["Sensibilidad en τ95", `${formatPercent(t.sensitivity)} (${t.tp} de ${t.tp + t.fn} melanomas)`],
    ["Especificidad en τ95", `${formatPercent(t.specificity)} (${formatInt(t.tn)} de ${formatInt(t.tn + t.fp)} benignas)`],
    ["Valor predictivo positivo", formatPercent(t.ppv)],
    ["Valor predictivo negativo", formatPercent(t.npv, 2)],
    ["Prevalencia en el conjunto de prueba", formatPercent(m.prevalence, 2)],
  ];
  return (
    <>
      <table>
        <thead>
          <tr>
            <th scope="col">Métrica</th>
            <th scope="col">Valor [intervalo de confianza del {Math.round(m.ci.level * 100)} %]</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k}>
              <th scope="row">{k}</th>
              <td className="num">{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>
        Medido una sola vez sobre {formatInt(m.n_images)} imágenes de {formatInt(m.n_patients)}{" "}
        pacientes que el modelo nunca vio, el {formatDateEs(m.date)}. Intervalos por bootstrap por
        paciente ({formatInt(m.ci.n_resamples)} remuestreos). El umbral τ95 se fijó sobre
        validación para una sensibilidad objetivo de {formatPercent(info.thresholds.tau_95.sensitivity_target, 0)}.
      </p>
    </>
  );
}

function LimitationsSection() {
  const [data, setData] = useState<Limitations | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [pending, setPending] = useState(false);

  const load = () => {
    setError(null);
    setPending(false);
    limitations()
      .then(setData)
      .catch((e: unknown) => {
        if (e instanceof ApiError && e.status === 404) setPending(true);
        else setError(e);
      });
  };
  useEffect(load, []);

  if (pending) {
    return (
      <div className="notice" role="status">
        <strong>Pendiente.</strong> Las limitaciones del modelo (<code>docs/f4_limitations.md</code>)
        todavía no están escritas. Esta ficha no está completa sin ellas.
      </div>
    );
  }
  if (error !== null) return <ErrorBox error={error} onRetry={load} />;
  if (!data) return <p className="muted">Cargando…</p>;
  const html = marked.parse(data.markdown, { async: false }) as string;
  return (
    <div className="limitations">
      <p className="small muted">
        Renderizado desde <code>{data.source}</code> ({data.n_sections} bloques).
      </p>
      <div dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  );
}

export default function ModelPage({ state }: { state: ModelInfoState }) {
  const info = state.info;
  return (
    <>
      <h1>Ficha del modelo</h1>
      <p className="subtitle">
        Qué es este sistema, cómo se midió, cómo leer su probabilidad y qué no puede hacer.
      </p>

      <section className="card" aria-labelledby="s1">
        <h2 id="s1" className="block-title">
          1 · Qué es y qué no es
        </h2>
        <ul>
          <li>Apoyo a la priorización de lesiones con sospecha de melanoma en imágenes dermatoscópicas: sugiere si conviene referir.</li>
          <li>No detecta otros cánceres de piel ni otras enfermedades; no es un diagnóstico.</li>
          <li>Sin aprobación regulatoria (COFEPRIS, FDA, CE). Uso educativo y de investigación.</li>
        </ul>
      </section>

      <section className="card" aria-labelledby="s2">
        <h2 id="s2" className="block-title">
          2 · Desempeño validado
        </h2>
        {info ? (
          <Performance info={info} />
        ) : state.loading ? (
          <p className="muted">Cargando…</p>
        ) : (
          <ErrorBox error={new ApiError(0, state.error ?? "")} onRetry={state.reload} />
        )}
      </section>

      <section className="card" aria-labelledby="s3">
        <h2 id="s3" className="block-title">
          3 · Cómo leer la probabilidad
        </h2>
        <p>
          La probabilidad que muestra la pantalla de análisis está calibrada
          {info ? ` (${info.calibration.method === "platt" ? "escalado de Platt" : info.calibration.method}, ajustado sobre ${formatInt(info.calibration.n_images)} imágenes de validación con ${info.calibration.n_positives} melanomas)` : ""}
          : un 3 % significa que, de cada 100 lesiones que recibieron esa puntuación en
          validación, alrededor de 3 resultaron melanoma. No es la probabilidad de que «el modelo
          acierte», y una probabilidad baja no descarta melanoma: el umbral de referencia está muy
          por debajo del 50 % precisamente para no dejar pasar casos.
        </p>
      </section>

      <section className="card" aria-labelledby="s4">
        <h2 id="s4" className="block-title">
          4 · Limitaciones
        </h2>
        <LimitationsSection />
      </section>

      <section className="card" aria-labelledby="s5">
        <h2 id="s5" className="block-title">
          5 · Datos y licencias
        </h2>
        <ul>
          <li>
            Entrenamiento y evaluación: ISIC 2020 Challenge Dataset (International Skin Imaging
            Collaboration), licencia CC-BY-NC 4.0,{" "}
            <a href="https://doi.org/10.34970/2020-ds01">doi:10.34970/2020-ds01</a>. Rotemberg et
            al., <em>Sci Data</em> 8, 34 (2021). Uso no comercial; las imágenes de ejemplo se
            reproducen con esa atribución.
          </li>
          <li>
            Evaluación externa: Diverse Dermatology Images (DDI, Stanford), acuerdo de uso académico;
            sus imágenes no se reproducen en esta interfaz ni en el repositorio.
          </li>
          <li>
            Tesis de licenciatura de Edgar Eduardo Ontiveros Lara: clasificador educativo de lesiones
            cutáneas para detección asistida de melanoma. Código MIT en <a href={REPO}>{REPO}</a>.
          </li>
        </ul>
      </section>

      <section className="card" aria-labelledby="s6">
        <h2 id="s6" className="block-title">
          6 · Versión
        </h2>
        {info ? (
          <table>
            <tbody>
              <tr>
                <th scope="row">Versión del modelo</th>
                <td>
                  <code>{info.model_version}</code>
                  {info.manifest.kind === "test" && " (paquete de prueba con pesos aleatorios)"}
                </td>
              </tr>
              <tr>
                <th scope="row">Commit</th>
                <td>
                  <a href={`${REPO}/commit/${info.manifest.git_sha}`}>
                    <code>{info.manifest.git_sha}</code>
                  </a>
                </td>
              </tr>
              <tr>
                <th scope="row">Exportado</th>
                <td>{formatDateEs(info.manifest.date)}</td>
              </tr>
              <tr>
                <th scope="row">Paquete</th>
                <td>
                  <a href={`${REPO}/releases/tag/model-${info.model_version}`}>
                    release model-{info.model_version}
                  </a>
                  ; ONNX opset {info.manifest.onnx.opset["ai.onnx"]}, entrada {info.preprocess.input_size} px.
                </td>
              </tr>
              <tr>
                <th scope="row">Repositorio</th>
                <td>
                  <a href={REPO}>{REPO}</a>
                </td>
              </tr>
            </tbody>
          </table>
        ) : (
          <p className="muted">{state.loading ? "Cargando…" : "Ficha del modelo no disponible."}</p>
        )}
        <p className="thesis">
          <img src="/escudo-color.png" alt="Escudo de la Universidad Autónoma de Chihuahua" width={44} height={44} />
          <span>
            Proyecto de tesis de licenciatura — Edgar Eduardo Ontiveros Lara, matrícula a335951.
            Facultad de Ingeniería, Universidad Autónoma de Chihuahua, 2026.
          </span>
        </p>
      </section>
    </>
  );
}
