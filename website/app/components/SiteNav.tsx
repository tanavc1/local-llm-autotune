"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
const links = [["/#how-it-works", "How it works"], ["/#benchmarks", "Benchmarks"], ["/#dashboard", "Dashboard"], ["/install", "Install"], ["/commands", "Commands"], ["/what-we-do", "All we do"], ["/#docker", "Docker"]];
export function SiteNav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  return <header className="site-header"><nav className="site-nav" aria-label="Main navigation">
    <Link href="/" className="wordmark" onClick={() => setOpen(false)}><svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M3 16V8m6 12V4m6 13V7m6 8V9" stroke="currentColor" strokeWidth="2.5" /></svg>autotune <span>v1.6.0</span></Link>
    <button className="menu-toggle" aria-expanded={open} aria-controls="site-links" onClick={() => setOpen(!open)}>{open ? "Close −" : "Menu +"}</button>
    <div id="site-links" className={`site-links ${open ? "is-open" : ""}`}>{links.map(([href,label]) => <Link key={href} href={href} aria-current={pathname === href ? "page" : undefined} onClick={() => setOpen(false)}>{label}</Link>)}<a className="github-link" href="https://github.com/tanavc1/local-llm-autotune" target="_blank" rel="noopener noreferrer">GitHub ↗</a></div>
  </nav></header>;
}
