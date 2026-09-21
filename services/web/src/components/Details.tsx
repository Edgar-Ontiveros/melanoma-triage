import type { PredictResponse } from "../api";

/** Bloque 4, plegado: versión, latencia, dimensiones de entrada y el JSON completo (sin las
 * imágenes en base64, que se resumen para que el texto sea legible). */
export default function Details({ result }: { result: PredictResponse }) {
  const compact = {
    ...result,
    cam: {
      ...result.cam,
      png_base64: result.cam.png_base64 ? `<${result.cam.png_base64.length} caracteres base64>` : null,
      crop_png_base64: `<${result.cam.crop_png_base64.length} caracteres base64>`,
    },
  };
  return (
    <section className="card">
      <details>
        <summary>4. Detalles técnicos</summary>
        <table>
          <tbody>
            <tr>
              <th scope="row">Versión del modelo</th>
              <td>
                <code>{result.model_version}</code>
              </td>
            </tr>
            <tr>
              <th scope="row">Latencia en el servidor</th>
              <td>{result.latency_ms.toFixed(0)} ms</td>
            </tr>
            <tr>
              <th scope="row">Imagen recibida</th>
              <td>
                {result.input.width} × {result.input.height} px, {result.input.format}
              </td>
            </tr>
            <tr>
              <th scope="row">Estado del mapa</th>
              <td>
                <code>{result.cam.status}</code>
              </td>
            </tr>
          </tbody>
        </table>
        <pre>{JSON.stringify(compact, null, 2)}</pre>
      </details>
    </section>
  );
}
