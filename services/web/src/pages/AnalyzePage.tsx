import { useEffect, useRef, useState } from "react";

import { predict, type PredictResponse } from "../api";
import type { ModelInfoState } from "../App";
import CamViewer from "../components/CamViewer";
import Details from "../components/Details";
import ErrorBox from "../components/ErrorBox";
import Examples, { type Example } from "../components/Examples";
import ProbabilityBlock from "../components/ProbabilityBlock";
import Recommendation from "../components/Recommendation";
import Uploader from "../components/Uploader";
import { formatSeconds } from "../lib/format";

interface Pending {
  blob: Blob;
  name: string;
  previewUrl: string;
  example: Example | null;
}

export default function AnalyzePage({ state }: { state: ModelInfoState }) {
  const [pending, setPending] = useState<Pending | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const startedAt = useRef(0);

  useEffect(() => {
    if (!loading) return;
    startedAt.current = performance.now();
    setElapsed(0);
    const id = window.setInterval(() => setElapsed(performance.now() - startedAt.current), 100);
    return () => window.clearInterval(id);
  }, [loading]);

  useEffect(() => () => {
    if (pending) URL.revokeObjectURL(pending.previewUrl);
  }, [pending]);

  const analyze = async (p: Pending) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await predict(p.blob, p.name));
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  };

  const submit = (blob: Blob, name: string, example: Example | null) => {
    const p: Pending = { blob, name, previewUrl: URL.createObjectURL(blob), example };
    setPending(p);
    void analyze(p);
  };

  const metrics = state.info?.metrics ?? null;

  return (
    <>
      <p className="muted">
        Suba una dermatoscopía y obtenga una probabilidad calibrada de melanoma, una recomendación
        de triage y el mapa de lo que el modelo evaluó. Herramienta educativa: no es un
        diagnóstico.
      </p>

      <Uploader onFile={(f) => submit(f, f.name, null)} disabled={loading} />
      <Examples onPick={(ex, blob) => submit(blob, `${ex.image_id}.jpg`, ex)} disabled={loading} />

      {pending && (
        <section className="card" aria-labelledby="entrada-title">
          <h2 id="entrada-title" style={{ marginTop: 0 }}>
            Imagen enviada
          </h2>
          <div className="preview">
            <img src={pending.previewUrl} alt={`Vista previa de ${pending.name}`} />
            <div>
              <p>
                <code>{pending.name}</code>
              </p>
              {pending.example && (
                <p className="small">
                  Ejemplo de validación de ISIC 2020 (CC-BY-NC): {pending.example.note}.
                </p>
              )}
              {loading && (
                <p aria-live="polite">
                  <span className="spinner" aria-hidden="true" />
                  Analizando… {formatSeconds(elapsed)}
                </p>
              )}
            </div>
          </div>
        </section>
      )}

      {error !== null && pending && <ErrorBox error={error} onRetry={() => void analyze(pending)} />}

      {result && (
        <>
          <ProbabilityBlock
            probability={result.probability}
            threshold={result.threshold}
            prevalence={metrics?.prevalence ?? null}
          />
          <Recommendation
            refer={result.refer}
            operatingPoint={result.operating_point}
            npvTest={metrics?.tau_95.npv ?? null}
          />
          <CamViewer cam={result.cam} />
          <Details result={result} />
          <p className="small muted">{result.disclaimer}</p>
        </>
      )}
    </>
  );
}
