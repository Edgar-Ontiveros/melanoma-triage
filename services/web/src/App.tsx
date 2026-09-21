import { useCallback, useEffect, useState } from "react";

import { ApiError, modelInfo, type ModelInfo } from "./api";
import DisclaimerModal from "./components/DisclaimerModal";
import Footer from "./components/Footer";
import { LesionIcon } from "./components/Icons";
import AnalyzePage from "./pages/AnalyzePage";
import ModelPage from "./pages/ModelPage";

export type Route = "/" | "/modelo";

function currentRoute(): Route {
  return window.location.pathname.startsWith("/modelo") ? "/modelo" : "/";
}

/** Estado compartido: /model-info se pide una vez; si falla, la pantalla de análisis sigue
 * funcionando (la respuesta de /predict trae lo esencial) y la del modelo muestra el error. */
export interface ModelInfoState {
  info: ModelInfo | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

export default function App() {
  const [route, setRoute] = useState<Route>(currentRoute);
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [infoLoading, setInfoLoading] = useState(true);

  const loadInfo = useCallback(() => {
    setInfoLoading(true);
    setInfoError(null);
    modelInfo()
      .then((i) => setInfo(i))
      .catch((e: unknown) =>
        setInfoError(e instanceof ApiError ? e.message : "No se pudo leer la ficha del modelo."),
      )
      .finally(() => setInfoLoading(false));
  }, []);

  useEffect(() => {
    loadInfo();
  }, [loadInfo]);

  useEffect(() => {
    const onPop = () => setRoute(currentRoute());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const navigate = (to: Route) => (e: React.MouseEvent) => {
    e.preventDefault();
    if (to !== route) {
      window.history.pushState({}, "", to);
      setRoute(to);
      window.scrollTo(0, 0);
    }
  };

  const infoState: ModelInfoState = { info, error: infoError, loading: infoLoading, reload: loadInfo };

  return (
    <>
      <DisclaimerModal disclaimer={info?.disclaimer ?? null} />
      <header className="top">
        <div className="container">
          <a href="/" className="brand" onClick={navigate("/")} style={{ color: "var(--accent)" }}>
            <LesionIcon />
            <span style={{ color: "var(--text)" }}>Detección asistida de melanoma</span>
          </a>
          <nav aria-label="Secciones">
            <a href="/" onClick={navigate("/")} aria-current={route === "/" ? "page" : undefined}>
              Análisis
            </a>
            <a
              href="/modelo"
              onClick={navigate("/modelo")}
              aria-current={route === "/modelo" ? "page" : undefined}
            >
              Ficha del modelo
            </a>
          </nav>
        </div>
      </header>
      <main className="container" id="main">
        {route === "/modelo" ? <ModelPage state={infoState} /> : <AnalyzePage state={infoState} />}
      </main>
      <Footer onModel={navigate("/modelo")} modelVersion={info?.model_version ?? null} />
    </>
  );
}
