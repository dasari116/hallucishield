import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Manages the WebSocket connection to the HalluciShield backend and
 * accumulates the live pipeline state: streamed text, per-sentence
 * verdicts (keyed by sentence_id), and connection status.
 */
export function useHalluciShieldSocket(wsUrl) {
  const [text, setText] = useState("");
  const [sentences, setSentences] = useState([]); // {sentence_id, text}
  const [verdicts, setVerdicts] = useState({}); // sentence_id -> verdict payload
  const [isStreaming, setIsStreaming] = useState(false);
  const [totalLatencyMs, setTotalLatencyMs] = useState(null);
  const [error, setError] = useState(null);
  const socketRef = useRef(null);

  const ensureSocket = useCallback(() => {
    if (
      socketRef.current &&
      (socketRef.current.readyState === WebSocket.OPEN ||
        socketRef.current.readyState === WebSocket.CONNECTING)
    ) {
      return socketRef.current;
    }
    const socket = new WebSocket(wsUrl);
    socketRef.current = socket;

    socket.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      switch (msg.type) {
        case "start":
          setError(null);
          setText("");
          setSentences([]);
          setVerdicts({});
          setTotalLatencyMs(null);
          setIsStreaming(true);
          break;
        case "token":
          setText((prev) => prev + msg.text);
          break;
        case "sentence":
          setSentences((prev) => [...prev, { sentence_id: msg.sentence_id, text: msg.text }]);
          break;
        case "verdict":
          setVerdicts((prev) => ({ ...prev, [msg.sentence_id]: msg }));
          break;
        case "done":
          setIsStreaming(false);
          setTotalLatencyMs(msg.total_latency_ms);
          break;
        default:
          break;
      }
    };

    socket.onerror = () => {
      setIsStreaming(false);
      setError("Connection to verification server failed. Ensure backend is running on port 8000.");
    };

    socket.onclose = () => {
      setIsStreaming(false);
      if (socketRef.current === socket) {
        socketRef.current = null;
      }
    };

    return socket;
  }, [wsUrl]);

  useEffect(() => {
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }
    };
  }, []);

  const send = useCallback(
    (prompt) => {
      setError(null);
      const socket = ensureSocket();
      const fire = () => socket.send(prompt);
      if (socket.readyState === WebSocket.OPEN) fire();
      else socket.addEventListener("open", fire, { once: true });
    },
    [ensureSocket]
  );

  return { text, sentences, verdicts, isStreaming, totalLatencyMs, error, send };
}
