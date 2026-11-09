import React, { useEffect, useMemo, useState } from "react";
import { useHalluciShieldSocket } from "./hooks/useHalluciShieldSocket.js";
import SentenceLine from "./components/SentenceLine.jsx";

const WS_URL = `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws/generate`;

export default function App() {
  const [prompt, setPrompt] = useState("Tell me about the Eiffel Tower");
  const [uploadState, setUploadState] = useState({ status: "idle", message: "" });
  const [sources, setSources] = useState([]);
  const [engineMode, setEngineMode] = useState("offline");
  const { text, sentences, verdicts, isStreaming, totalLatencyMs, error, send } =
    useHalluciShieldSocket(WS_URL);

  const summary = useMemo(() => {
    const counts = { supported: 0, contradicted: 0, unverifiable: 0 };
    Object.values(verdicts).forEach((v) => counts[v.verdict]++);
    return counts;
  }, [verdicts]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!prompt.trim() || isStreaming) return;
    send(prompt.trim());
  };

  const loadSources = async () => {
    try {
      const [sourcesRes, healthRes] = await Promise.all([
        fetch("/api/sources"),
        fetch("/api/health"),
      ]);
      if (sourcesRes.ok) setSources((await sourcesRes.json()).sources);
      if (healthRes.ok) setEngineMode((await healthRes.json()).mode || "offline");
    } catch {
    }
  };

  useEffect(() => {
    loadSources();
  }, []);

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setUploadState({ status: "uploading", message: `Indexing ${file.name}` });
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch("/api/ingest/file", { method: "POST", body: formData });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Upload failed");
      setUploadState({ status: "success", message: `${file.name} added · ${result.chunks_added} chunks` });
      loadSources();
    } catch (error) {
      setUploadState({ status: "error", message: error.message });
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true"><span /></span>
          <div>
            <h1>HalluciShield</h1>
            <p>TRUSTED AI, VERIFIED LIVE</p>
          </div>
        </div>
        <nav className="main-nav" aria-label="Main navigation">
          <a href="#workspace">Verify a response</a>
          <a href="#knowledge">Knowledge base</a>
          <a href="#about">How it works</a>
        </nav>
        <div className="header-status">
          <span className={`status-dot ${isStreaming ? "is-live" : ""}`} />
          <span>{isStreaming ? "Monitoring live output" : `System ready (${engineMode})`}</span>
          {totalLatencyMs !== null && <strong>{totalLatencyMs.toFixed(0)}ms</strong>}
        </div>
      </header>

      <main>
        <section className="hero" id="about">
          <div className="hero-copy-block">
            <p className="eyebrow">AI SAFETY · EVIDENCE · CLARITY</p>
            <h2>YOUR ANSWERS:<br /><em>PROVEN.</em></h2>
            <p className="hero-copy">HalluciShield watches every generated claim and checks it against the knowledge you trust. Ask freely. Verify instantly.</p>
            <a className="hero-link" href="#workspace">Start verifying <span aria-hidden="true">-&gt;</span></a>
          </div>
          <div className="hero-art" aria-label="Live verification status">
            <span className="art-line line-one" />
            <span className="art-line line-two" />
            <span className="art-ring" />
            <span className="art-center">HS</span>
            <span className="art-caption">REAL-TIME<br />CLAIM CHECK</span>
          </div>
        </section>

        <section className="stats-row" aria-label="HalluciShield capabilities">
          <div><strong>01</strong><span>Streamed responses</span></div>
          <div><strong>02</strong><span>Sentence-level checks</span></div>
          <div><strong>03</strong><span>Source-backed evidence</span></div>
          <div><strong>04</strong><span>Private local engine</span></div>
        </section>

        <section className="workspace-section" id="workspace">
          <div className="section-intro">
            <p className="eyebrow">THE VERIFICATION DESK</p>
            <h2>Ask. Watch. Know.</h2>
            <p>Run a response through your indexed sources and see exactly where confidence comes from.</p>
          </div>
          <form className="prompt-form" onSubmit={handleSubmit}>
          <div className="input-wrap">
            <span className="input-label">QUERY</span>
            <input
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Ask something the model will answer..."
              disabled={isStreaming}
            />
          </div>
          <button type="submit" disabled={isStreaming}>
            <span>{isStreaming ? "Generating" : "Run verification"}</span>
            <span className="button-arrow" aria-hidden="true">-&gt;</span>
          </button>
          </form>
          {error && <p className="upload-message error" style={{ marginTop: "8px" }}>{error}</p>}
        </section>

        <section className="knowledge-bar">
          <div className="knowledge-copy">
            <span className="knowledge-icon" aria-hidden="true">+</span>
            <div>
              <strong>Add to knowledge base</strong>
              <span>Upload a UTF-8 text document for retrieval</span>
            </div>
          </div>
          <label className="upload-button">
            <input type="file" accept=".txt,.md,.markdown,.csv,text/plain,text/markdown,text/csv" onChange={handleFileUpload} disabled={uploadState.status === "uploading"} />
            {uploadState.status === "uploading" ? "Indexing..." : "Choose file"}
          </label>
          {uploadState.message && <span className={`upload-message ${uploadState.status}`}>{uploadState.message}</span>}
        </section>

        <section className="source-library" id="knowledge">
          <div className="library-heading">
            <div><span className="panel-index">03</span><h2>Knowledge base</h2></div>
            <span>{sources.length} source{sources.length === 1 ? "" : "s"} · {sources.reduce((total, source) => total + source.chunks, 0)} chunks</span>
          </div>
          <div className="source-list">
            {sources.map((source) => (
              <div className="source-item" key={source.source}>
                <span className="file-glyph">TXT</span>
                <strong>{source.source}</strong>
                <span>{source.chunks} chunk{source.chunks === 1 ? "" : "s"}</span>
              </div>
            ))}
          </div>
        </section>

        <div className="summary-bar" aria-label="Verification summary">
          <div className="summary-title">CURRENT RUN <span>{Object.keys(verdicts).length} claims</span></div>
          <span className="summary-item supported"><i />Supported <strong>{summary.supported}</strong></span>
          <span className="summary-item contradicted"><i />Contradicted <strong>{summary.contradicted}</strong></span>
          <span className="summary-item unverifiable"><i />Unverifiable <strong>{summary.unverifiable}</strong></span>
        </div>

        <div className="content">
          <section className="stream-panel raw-stream">
            <div className="panel-heading">
              <div><span className="panel-index">01</span><h2>Live generation</h2></div>
              <span className="panel-state">{text ? "STREAMING" : "WAITING"}</span>
            </div>
            <p className={`raw-text ${text ? "has-text" : ""}`}>{text || "Output will stream here as it's generated..."}</p>
          </section>

          <section className="stream-panel verified-stream">
            <div className="panel-heading">
              <div><span className="panel-index">02</span><h2>Claim verification</h2></div>
              <span className="panel-state">{sentences.length ? `${sentences.length} CLAIMS` : "READY"}</span>
            </div>
            {sentences.length === 0 && (
              <p className="empty-hint">Verified sentences will appear here as they're checked.</p>
            )}
            {sentences.map((s) => (
              <SentenceLine key={s.sentence_id} sentence={s} verdict={verdicts[s.sentence_id]} />
            ))}
          </section>
        </div>
      </main>
      <footer className="app-footer">
        <div className="footer-brand"><span className="brand-mark" aria-hidden="true"><span /></span><strong>HALLUCISHIELD</strong></div>
        <p>Evidence-aware response monitoring for a clearer way to work with AI.</p>
        <span>RAG + NLI · LOCAL FIRST</span>
      </footer>
    </div>
  );
}
