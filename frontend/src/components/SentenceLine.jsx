import React from "react";

const VERDICT_STYLES = {
  supported: { label: "Supported", className: "verdict-supported" },
  contradicted: { label: "Contradicted", className: "verdict-contradicted" },
  unverifiable: { label: "Unverifiable", className: "verdict-unverifiable" },
};

export default function SentenceLine({ sentence, verdict }) {
  const style = verdict ? VERDICT_STYLES[verdict.verdict] : null;

  return (
    <div className={`sentence-row ${style ? style.className : "verdict-pending"}`}>
      <div className="sentence-text">{sentence.text}</div>
      <div className="sentence-meta">
        {!verdict && <span className="badge badge-pending">checking…</span>}
        {verdict && (
          <>
            <span className={`badge badge-${verdict.verdict}`}>
              {style.label} · {Math.round(verdict.confidence * 100)}%
            </span>
            <span className="latency">{verdict.latency_ms.toFixed(0)}ms</span>
          </>
        )}
      </div>
      {verdict && verdict.evidence.length > 0 && (
        <details className="evidence">
          <summary>{verdict.evidence.length} source{verdict.evidence.length > 1 ? "s" : ""}</summary>
          <ul>
            {verdict.evidence.map((e, i) => (
              <li key={i}>
                <span className="evidence-source">{e.source}</span>
                <span className="evidence-score">{e.score.toFixed(2)}</span>
                <p>{e.text}</p>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
