import { useState, useRef, useCallback } from "react";

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

const YOUTUBE_LINKS = {
  A: "https://www.youtube.com/results?search_query=ASL+sign+letter+A",
  B: "https://www.youtube.com/results?search_query=ASL+sign+letter+B",
  C: "https://www.youtube.com/results?search_query=ASL+sign+letter+C",
  D: "https://www.youtube.com/results?search_query=ASL+sign+letter+D",
  E: "https://www.youtube.com/results?search_query=ASL+sign+letter+E",
  F: "https://www.youtube.com/results?search_query=ASL+sign+letter+F",
  G: "https://www.youtube.com/results?search_query=ASL+sign+letter+G",
  H: "https://www.youtube.com/results?search_query=ASL+sign+letter+H",
  I: "https://www.youtube.com/results?search_query=ASL+sign+letter+I",
  J: "https://www.youtube.com/results?search_query=ASL+sign+letter+J",
  K: "https://www.youtube.com/results?search_query=ASL+sign+letter+K",
  L: "https://www.youtube.com/results?search_query=ASL+sign+letter+L",
  M: "https://www.youtube.com/results?search_query=ASL+sign+letter+M",
  N: "https://www.youtube.com/results?search_query=ASL+sign+letter+N",
  O: "https://www.youtube.com/results?search_query=ASL+sign+letter+O",
  P: "https://www.youtube.com/results?search_query=ASL+sign+letter+P",
  Q: "https://www.youtube.com/results?search_query=ASL+sign+letter+Q",
  R: "https://www.youtube.com/results?search_query=ASL+sign+letter+R",
  S: "https://www.youtube.com/results?search_query=ASL+sign+letter+S",
  T: "https://www.youtube.com/results?search_query=ASL+sign+letter+T",
  U: "https://www.youtube.com/results?search_query=ASL+sign+letter+U",
  V: "https://www.youtube.com/results?search_query=ASL+sign+letter+V",
  W: "https://www.youtube.com/results?search_query=ASL+sign+letter+W",
  X: "https://www.youtube.com/results?search_query=ASL+sign+letter+X",
  Y: "https://www.youtube.com/results?search_query=ASL+sign+letter+Y",
  Z: "https://www.youtube.com/results?search_query=ASL+sign+letter+Z",
};

// ASL fingerspelling reference descriptions
const ASL_DESCRIPTIONS = {
  A: "closed fist with thumb resting on the side",
  B: "flat hand, fingers together pointing up, thumb tucked",
  C: "curved hand forming a C shape",
  D: "index finger points up, other fingers and thumb form a circle",
  E: "fingers bent/curled, thumb tucked under",
  F: "index finger and thumb touch forming a circle, other fingers up",
  G: "index finger and thumb point sideways horizontally",
  H: "index and middle fingers extended horizontally",
  I: "pinky finger extended, others closed",
  J: "pinky extended, draw a J motion",
  K: "index and middle fingers up in V, thumb between them",
  L: "L-shape: index up, thumb out",
  M: "three fingers folded over thumb",
  N: "two fingers folded over thumb",
  O: "all fingers curved touching thumb forming an O",
  P: "like K but pointing downward",
  Q: "like G but pointing downward",
  R: "index and middle fingers crossed",
  S: "closed fist with thumb over fingers",
  T: "thumb between index and middle finger",
  U: "index and middle fingers together pointing up",
  V: "index and middle fingers spread in V/peace sign",
  W: "index, middle, ring fingers spread open",
  X: "index finger hooked/bent",
  Y: "thumb and pinky extended (hang loose)",
  Z: "index finger draws a Z",
};

export default function SignLanguageTrainer() {
  const [phase, setPhase] = useState("idle"); // idle | select | camera | result
  const [selectedLetter, setSelectedLetter] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const [score, setScore] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [loading, setLoading] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const canvasRef = useRef(null);

  const startCamera = useCallback(async () => {
    try {
      setCameraError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      setCameraError("Camera access denied. Please allow camera permissions.");
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  }, []);

  const captureImage = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);
    return canvas.toDataURL("image/jpeg", 0.8).split(",")[1];
  }, []);

  const analyzeSign = useCallback(
    async (imageBase64) => {
      setLoading(true);
      try {
        const prompt = `You are an expert ASL (American Sign Language) instructor evaluating a student's fingerspelling.

The student is attempting to sign the letter "${selectedLetter}" in ASL.
The correct ASL sign for "${selectedLetter}" is: ${ASL_DESCRIPTIONS[selectedLetter]}.

Look at the hand in the image and evaluate how accurately the student is forming this sign.

Respond ONLY with valid JSON in this exact format:
{
  "score": <integer between 10 and 100, multiples of 10>,
  "detected": "<what hand shape you see in the image>",
  "correct_form": "<brief description of correct ${selectedLetter} sign>",
  "feedback": "<2-3 sentences of specific, encouraging feedback on what they did right and what to improve>",
  "tips": ["<tip 1>", "<tip 2>"]
}

Be generous but honest. If no hand is visible, give a score of 10 and note the hand isn't visible.`;

        const response = await fetch("https://api.anthropic.com/v1/messages", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: "claude-sonnet-4-20250514",
            max_tokens: 1000,
            messages: [
              {
                role: "user",
                content: [
                  {
                    type: "image",
                    source: {
                      type: "base64",
                      media_type: "image/jpeg",
                      data: imageBase64,
                    },
                  },
                  { type: "text", text: prompt },
                ],
              },
            ],
          }),
        });

        const data = await response.json();
        const text = data.content
          .map((b) => b.text || "")
          .join("")
          .replace(/```json|```/g, "")
          .trim();
        const parsed = JSON.parse(text);
        return parsed;
      } catch (e) {
        console.error(e);
        return {
          score: 10,
          detected: "Unable to analyze",
          correct_form: ASL_DESCRIPTIONS[selectedLetter],
          feedback:
            "We had trouble analyzing your image. Please try again with better lighting.",
          tips: ["Ensure good lighting", "Keep your hand clearly in frame"],
        };
      } finally {
        setLoading(false);
      }
    },
    [selectedLetter]
  );

  const handleMarkSign = () => setPhase("select");

  const handleSelectLetter = (letter) => setSelectedLetter(letter);

  const handleSubmitLetter = async () => {
    if (!selectedLetter) return;
    setPhase("camera");
    setTimeout(() => startCamera(), 100);
  };

  const handleCapture = async () => {
    const imgData = captureImage();
    if (!imgData) return;
    setCapturedImage("data:image/jpeg;base64," + imgData);
    stopCamera();
    const result = await analyzeSign(imgData);
    setScore(result.score);
    setFeedback(result);
    setPhase("result");
  };

  const handleReset = () => {
    stopCamera();
    setCapturedImage(null);
    setScore(null);
    setFeedback(null);
    setSelectedLetter(null);
    setPhase("idle");
  };

  const handleTryAgain = () => {
    setCapturedImage(null);
    setScore(null);
    setFeedback(null);
    setPhase("camera");
    setTimeout(() => startCamera(), 100);
  };

  const getScoreColor = (s) => {
    if (s >= 80) return "#00e5a0";
    if (s >= 50) return "#f0c040";
    return "#ff6b6b";
  };

  const getScoreLabel = (s) => {
    if (s >= 90) return "Excellent!";
    if (s >= 70) return "Good Job!";
    if (s >= 50) return "Keep Practicing";
    return "Needs Work";
  };

  return (
    <div style={styles.root}>
      <canvas ref={canvasRef} style={{ display: "none" }} />

      {/* Header */}
      <header style={styles.header}>
        <div style={styles.logo}>
          <span style={styles.logoIcon}>🤟</span>
          <div>
            <div style={styles.logoTitle}>SignMaster</div>
            <div style={styles.logoSub}>ASL Fingerspelling Trainer</div>
          </div>
        </div>
      </header>

      <main style={styles.main}>
        {/* IDLE */}
        {phase === "idle" && (
          <div style={styles.idleContainer}>
            <div style={styles.heroGlow} />
            <h1 style={styles.heroTitle}>
              Learn Sign Language
              <br />
              <span style={styles.heroAccent}>One Letter at a Time</span>
            </h1>
            <p style={styles.heroDesc}>
              Practice ASL fingerspelling with real-time AI feedback. Get scored
              instantly and improve with curated video resources.
            </p>
            <button style={styles.primaryBtn} onClick={handleMarkSign}>
              <span>✋</span> Mark a Sign
            </button>
            <div style={styles.featureRow}>
              {["AI Scoring", "Webcam Analysis", "Video Guides"].map((f) => (
                <div key={f} style={styles.featureChip}>
                  {f}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* SELECT LETTER */}
        {phase === "select" && (
          <div style={styles.card}>
            <h2 style={styles.cardTitle}>Choose a Letter to Practice</h2>
            <p style={styles.cardSub}>
              Select an ASL alphabet letter you want to sign
            </p>
            <div style={styles.alphabetGrid}>
              {ALPHABET.map((letter) => (
                <button
                  key={letter}
                  style={{
                    ...styles.letterBtn,
                    ...(selectedLetter === letter
                      ? styles.letterBtnActive
                      : {}),
                  }}
                  onClick={() => handleSelectLetter(letter)}
                >
                  {letter}
                </button>
              ))}
            </div>
            {selectedLetter && (
              <div style={styles.selectedInfo}>
                <span style={styles.selectedBadge}>{selectedLetter}</span>
                <div>
                  <div style={styles.selectedLabel}>
                    How to sign &ldquo;{selectedLetter}&rdquo;
                  </div>
                  <div style={styles.selectedDesc}>
                    {ASL_DESCRIPTIONS[selectedLetter]}
                  </div>
                </div>
              </div>
            )}
            <div style={styles.btnRow}>
              <button style={styles.ghostBtn} onClick={handleReset}>
                ← Back
              </button>
              <button
                style={{
                  ...styles.primaryBtn,
                  opacity: selectedLetter ? 1 : 0.4,
                }}
                disabled={!selectedLetter}
                onClick={handleSubmitLetter}
              >
                Open Camera →
              </button>
            </div>
          </div>
        )}

        {/* CAMERA */}
        {phase === "camera" && (
          <div style={styles.card}>
            <div style={styles.cameraHeader}>
              <span style={styles.letterPill}>{selectedLetter}</span>
              <div>
                <h2 style={styles.cardTitle}>
                  Sign &ldquo;{selectedLetter}&rdquo;
                </h2>
                <p style={styles.cardSub}>{ASL_DESCRIPTIONS[selectedLetter]}</p>
              </div>
            </div>
            {cameraError ? (
              <div style={styles.errorBox}>{cameraError}</div>
            ) : (
              <div style={styles.videoWrapper}>
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  style={styles.video}
                />
                <div style={styles.videoOverlay}>
                  <div style={styles.corner} style={{ ...styles.corner, top: 8, left: 8, borderTop: "3px solid #00e5a0", borderLeft: "3px solid #00e5a0" }} />
                  <div style={{ ...styles.corner, top: 8, right: 8, borderTop: "3px solid #00e5a0", borderRight: "3px solid #00e5a0" }} />
                  <div style={{ ...styles.corner, bottom: 8, left: 8, borderBottom: "3px solid #00e5a0", borderLeft: "3px solid #00e5a0" }} />
                  <div style={{ ...styles.corner, bottom: 8, right: 8, borderBottom: "3px solid #00e5a0", borderRight: "3px solid #00e5a0" }} />
                </div>
              </div>
            )}
            <p style={styles.cameraHint}>
              Position your hand clearly in frame with good lighting
            </p>
            <div style={styles.btnRow}>
              <button style={styles.ghostBtn} onClick={() => { stopCamera(); setPhase("select"); }}>
                ← Back
              </button>
              <button
                style={{ ...styles.primaryBtn, background: loading ? "#333" : undefined }}
                onClick={handleCapture}
                disabled={loading || !!cameraError}
              >
                {loading ? "Analyzing..." : "📸 Capture & Analyze"}
              </button>
            </div>
          </div>
        )}

        {/* RESULT */}
        {phase === "result" && feedback && (
          <div style={styles.card}>
            <div style={styles.resultHeader}>
              <img src={capturedImage} alt="captured" style={styles.capturedImg} />
              <div style={styles.scoreCircle}>
                <svg viewBox="0 0 120 120" style={{ width: 120, height: 120 }}>
                  <circle cx="60" cy="60" r="54" fill="none" stroke="#1a1a2e" strokeWidth="10" />
                  <circle
                    cx="60" cy="60" r="54"
                    fill="none"
                    stroke={getScoreColor(score)}
                    strokeWidth="10"
                    strokeDasharray={`${(score / 100) * 339} 339`}
                    strokeLinecap="round"
                    transform="rotate(-90 60 60)"
                    style={{ transition: "stroke-dasharray 1s ease" }}
                  />
                </svg>
                <div style={styles.scoreInner}>
                  <div style={{ ...styles.scoreNum, color: getScoreColor(score) }}>{score}%</div>
                  <div style={styles.scoreLabel}>{getScoreLabel(score)}</div>
                </div>
              </div>
            </div>

            <div style={styles.feedbackBox}>
              <div style={styles.feedbackTitle}>AI Feedback</div>
              <p style={styles.feedbackText}>{feedback.feedback}</p>
              {feedback.tips && (
                <ul style={styles.tipsList}>
                  {feedback.tips.map((tip, i) => (
                    <li key={i} style={styles.tipItem}>
                      <span style={styles.tipDot}>→</span> {tip}
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div style={styles.videoSection}>
              <div style={styles.videoSectionTitle}>📺 Improve with Video Resources</div>
              <div style={styles.videoLinks}>
                <a
                  href={YOUTUBE_LINKS[selectedLetter]}
                  target="_blank"
                  rel="noreferrer"
                  style={styles.videoLink}
                >
                  🎥 How to Sign &ldquo;{selectedLetter}&rdquo; — Search YouTube
                </a>
                <a
                  href={`https://www.youtube.com/results?search_query=ASL+fingerspelling+tutorial`}
                  target="_blank"
                  rel="noreferrer"
                  style={styles.videoLink}
                >
                  🎥 ASL Fingerspelling Full Alphabet
                </a>
                <a
                  href={`https://www.youtube.com/results?search_query=ASL+alphabet+practice+for+beginners`}
                  target="_blank"
                  rel="noreferrer"
                  style={{ ...styles.videoLink, background: "rgba(0,229,160,0.08)", borderColor: "rgba(0,229,160,0.3)" }}
                >
                  ⭐ ASL Beginner Practice Guide
                </a>
              </div>
            </div>

            <div style={styles.btnRow}>
              <button style={styles.ghostBtn} onClick={handleReset}>
                🏠 Home
              </button>
              <button style={styles.secondaryBtn} onClick={handleTryAgain}>
                🔄 Try Again
              </button>
              <button style={styles.primaryBtn} onClick={() => { stopCamera(); setSelectedLetter(null); setPhase("select"); }}>
                Next Letter →
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

const styles = {
  root: {
    minHeight: "100vh",
    background: "#0a0a14",
    color: "#e8e8f0",
    fontFamily: "'Courier New', monospace",
    display: "flex",
    flexDirection: "column",
  },
  header: {
    padding: "16px 24px",
    borderBottom: "1px solid rgba(255,255,255,0.06)",
    background: "rgba(255,255,255,0.02)",
    backdropFilter: "blur(10px)",
  },
  logo: {
    display: "flex",
    alignItems: "center",
    gap: 12,
  },
  logoIcon: { fontSize: 32 },
  logoTitle: {
    fontSize: 20,
    fontWeight: 700,
    letterSpacing: 3,
    color: "#00e5a0",
    textTransform: "uppercase",
  },
  logoSub: {
    fontSize: 11,
    color: "#666",
    letterSpacing: 2,
    textTransform: "uppercase",
  },
  main: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },
  idleContainer: {
    textAlign: "center",
    maxWidth: 560,
    position: "relative",
  },
  heroGlow: {
    position: "absolute",
    top: "50%",
    left: "50%",
    transform: "translate(-50%, -50%)",
    width: 400,
    height: 400,
    background: "radial-gradient(circle, rgba(0,229,160,0.08) 0%, transparent 70%)",
    pointerEvents: "none",
    zIndex: 0,
  },
  heroTitle: {
    fontSize: 42,
    fontWeight: 900,
    lineHeight: 1.2,
    marginBottom: 16,
    letterSpacing: -1,
    position: "relative",
    zIndex: 1,
  },
  heroAccent: {
    color: "#00e5a0",
  },
  heroDesc: {
    fontSize: 16,
    color: "#888",
    lineHeight: 1.7,
    marginBottom: 36,
    position: "relative",
    zIndex: 1,
  },
  primaryBtn: {
    background: "#00e5a0",
    color: "#0a0a14",
    border: "none",
    padding: "14px 32px",
    borderRadius: 4,
    fontSize: 16,
    fontWeight: 700,
    fontFamily: "'Courier New', monospace",
    cursor: "pointer",
    letterSpacing: 1,
    display: "inline-flex",
    alignItems: "center",
    gap: 8,
    transition: "all 0.2s",
    textTransform: "uppercase",
  },
  featureRow: {
    display: "flex",
    gap: 12,
    justifyContent: "center",
    marginTop: 28,
  },
  featureChip: {
    border: "1px solid rgba(0,229,160,0.3)",
    color: "#00e5a0",
    padding: "6px 14px",
    borderRadius: 2,
    fontSize: 12,
    letterSpacing: 2,
    textTransform: "uppercase",
  },
  card: {
    background: "rgba(255,255,255,0.03)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 8,
    padding: 32,
    width: "100%",
    maxWidth: 680,
  },
  cardTitle: {
    fontSize: 24,
    fontWeight: 700,
    marginBottom: 6,
    letterSpacing: -0.5,
  },
  cardSub: {
    fontSize: 14,
    color: "#666",
    marginBottom: 24,
  },
  alphabetGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(9, 1fr)",
    gap: 8,
    marginBottom: 24,
  },
  letterBtn: {
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.1)",
    color: "#e8e8f0",
    padding: "12px 0",
    borderRadius: 4,
    fontSize: 16,
    fontWeight: 700,
    fontFamily: "'Courier New', monospace",
    cursor: "pointer",
    transition: "all 0.15s",
  },
  letterBtnActive: {
    background: "rgba(0,229,160,0.15)",
    border: "1px solid #00e5a0",
    color: "#00e5a0",
  },
  selectedInfo: {
    display: "flex",
    alignItems: "center",
    gap: 16,
    background: "rgba(0,229,160,0.06)",
    border: "1px solid rgba(0,229,160,0.2)",
    borderRadius: 6,
    padding: "14px 20px",
    marginBottom: 24,
  },
  selectedBadge: {
    fontSize: 40,
    fontWeight: 900,
    color: "#00e5a0",
    minWidth: 48,
    textAlign: "center",
  },
  selectedLabel: {
    fontSize: 13,
    color: "#888",
    marginBottom: 4,
    textTransform: "uppercase",
    letterSpacing: 1,
  },
  selectedDesc: {
    fontSize: 15,
    color: "#ccc",
  },
  btnRow: {
    display: "flex",
    gap: 12,
    flexWrap: "wrap",
  },
  ghostBtn: {
    background: "transparent",
    border: "1px solid rgba(255,255,255,0.15)",
    color: "#888",
    padding: "12px 20px",
    borderRadius: 4,
    fontSize: 14,
    fontFamily: "'Courier New', monospace",
    cursor: "pointer",
  },
  secondaryBtn: {
    background: "transparent",
    border: "1px solid rgba(0,229,160,0.4)",
    color: "#00e5a0",
    padding: "12px 20px",
    borderRadius: 4,
    fontSize: 14,
    fontFamily: "'Courier New', monospace",
    cursor: "pointer",
    fontWeight: 700,
  },
  cameraHeader: {
    display: "flex",
    alignItems: "center",
    gap: 16,
    marginBottom: 20,
  },
  letterPill: {
    fontSize: 36,
    fontWeight: 900,
    color: "#00e5a0",
    background: "rgba(0,229,160,0.1)",
    border: "2px solid rgba(0,229,160,0.3)",
    width: 64,
    height: 64,
    borderRadius: 8,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  videoWrapper: {
    position: "relative",
    borderRadius: 6,
    overflow: "hidden",
    background: "#000",
    marginBottom: 12,
    aspectRatio: "4/3",
  },
  video: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
    transform: "scaleX(-1)",
  },
  videoOverlay: {
    position: "absolute",
    inset: 0,
    pointerEvents: "none",
  },
  corner: {
    position: "absolute",
    width: 24,
    height: 24,
  },
  cameraHint: {
    fontSize: 12,
    color: "#555",
    textAlign: "center",
    marginBottom: 20,
    letterSpacing: 0.5,
  },
  errorBox: {
    background: "rgba(255,107,107,0.1)",
    border: "1px solid rgba(255,107,107,0.3)",
    color: "#ff6b6b",
    padding: 16,
    borderRadius: 4,
    marginBottom: 20,
    fontSize: 14,
  },
  resultHeader: {
    display: "flex",
    gap: 24,
    alignItems: "center",
    marginBottom: 24,
  },
  capturedImg: {
    width: 140,
    height: 140,
    objectFit: "cover",
    borderRadius: 6,
    border: "2px solid rgba(255,255,255,0.1)",
    transform: "scaleX(-1)",
    flexShrink: 0,
  },
  scoreCircle: {
    position: "relative",
    flexShrink: 0,
  },
  scoreInner: {
    position: "absolute",
    inset: 0,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
  },
  scoreNum: {
    fontSize: 26,
    fontWeight: 900,
    lineHeight: 1,
  },
  scoreLabel: {
    fontSize: 10,
    color: "#888",
    letterSpacing: 1,
    textTransform: "uppercase",
  },
  feedbackBox: {
    background: "rgba(255,255,255,0.03)",
    border: "1px solid rgba(255,255,255,0.07)",
    borderRadius: 6,
    padding: "18px 20px",
    marginBottom: 20,
  },
  feedbackTitle: {
    fontSize: 12,
    letterSpacing: 2,
    textTransform: "uppercase",
    color: "#00e5a0",
    marginBottom: 10,
  },
  feedbackText: {
    fontSize: 15,
    lineHeight: 1.7,
    color: "#ccc",
    marginBottom: 12,
  },
  tipsList: {
    listStyle: "none",
    padding: 0,
    margin: 0,
  },
  tipItem: {
    fontSize: 13,
    color: "#aaa",
    marginBottom: 6,
    display: "flex",
    gap: 8,
  },
  tipDot: {
    color: "#00e5a0",
  },
  videoSection: {
    marginBottom: 24,
  },
  videoSectionTitle: {
    fontSize: 13,
    letterSpacing: 2,
    textTransform: "uppercase",
    color: "#888",
    marginBottom: 12,
  },
  videoLinks: {
    display: "flex",
    flexDirection: "column",
    gap: 8,
  },
  videoLink: {
    display: "block",
    background: "rgba(255,255,255,0.03)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 4,
    padding: "12px 16px",
    color: "#ccc",
    textDecoration: "none",
    fontSize: 14,
    transition: "all 0.2s",
  },
};