import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "autotune — Local LLM Inference Optimizer",
  description:
    "39% faster time-to-first-word. 3× less KV cache. Drop-in optimization for Ollama, LM Studio, and Apple Silicon MLX. Zero config changes.",
  openGraph: {
    title: "autotune — Local LLM Inference Optimizer",
    description: "39% faster TTFT. 3× less KV cache. Drop-in for Ollama & MLX.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
