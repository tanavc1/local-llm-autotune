import { ImageResponse } from "next/og";

export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function Image() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          backgroundColor: "#f7f5ef",
          padding: "64px 72px",
          fontFamily: "sans-serif",
        }}
      >
        {/* Top: wordmark + tagline */}
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
            {/* Accent bar */}
            <div
              style={{
                width: "6px",
                height: "56px",
                borderRadius: "3px",
                background: "#963d24",
              }}
            />
            <span
              style={{
                fontSize: "72px",
                fontWeight: "800",
                color: "#262b27",
                letterSpacing: "-2px",
                lineHeight: 1,
              }}
            >
              autotune
            </span>
          </div>
          <span
            style={{
              fontSize: "28px",
              color: "#575e56",
              letterSpacing: "-0.3px",
              marginLeft: "22px",
            }}
          >
            Local LLM Inference Optimizer
          </span>
        </div>

        {/* Middle: stat cards */}
        <div style={{ display: "flex", gap: "20px", marginLeft: "22px" }}>
          {[
            { value: "−39%", label: "faster TTFT", color: "#963d24", bg: "#eeece4", border: "#dcded2" },
            { value: "−67%", label: "KV cache RAM", color: "#3f6074", bg: "#eeece4", border: "#dcded2" },
            { value: "zero", label: "config changes", color: "#38614b", bg: "#eeece4", border: "#dcded2" },
          ].map((stat) => (
            <div
              key={stat.label}
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "6px",
                backgroundColor: stat.bg,
                border: `1.5px solid ${stat.border}`,
                borderRadius: "3px",
                padding: "20px 28px",
                minWidth: "220px",
              }}
            >
              <span style={{ fontSize: "48px", fontWeight: "800", color: stat.color, lineHeight: 1, letterSpacing: "-1px" }}>
                {stat.value}
              </span>
              <span style={{ fontSize: "18px", color: "#575e56", letterSpacing: "0px" }}>
                {stat.label}
              </span>
            </div>
          ))}
        </div>

        {/* Bottom: install command + compatibility */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginLeft: "22px" }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "12px",
              backgroundColor: "#dcded2",
              border: "1px solid #dcded2",
              borderRadius: "12px",
              padding: "14px 24px",
            }}
          >
            <span style={{ fontSize: "18px", color: "#575e56", fontFamily: "monospace" }}>$</span>
            <span style={{ fontSize: "22px", color: "#38614b", fontFamily: "monospace", letterSpacing: "-0.3px" }}>
              pip install llm-autotune
            </span>
          </div>
          <div style={{ display: "flex", gap: "16px" }}>
            {["Ollama", "MLX", "LM Studio"].map((name) => (
              <span
                key={name}
                style={{
                  fontSize: "16px",
                  color: "#575e56",
                  backgroundColor: "#dcded2",
                  border: "1px solid #dcded2",
                  borderRadius: "8px",
                  padding: "8px 16px",
                }}
              >
                {name}
              </span>
            ))}
          </div>
        </div>
      </div>
    ),
    size
  );
}
