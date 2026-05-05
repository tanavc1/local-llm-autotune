import Link from "next/link";

export default function NotFound() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 text-center text-[#e8e8f0]">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute top-1/2 left-1/2 h-[500px] w-[500px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-violet-600/8 blur-[100px]" />
      </div>
      <div className="relative z-10">
        <div className="text-8xl font-bold text-violet-300/30 mb-4">404</div>
        <h1 className="text-2xl font-bold text-white mb-3">Page not found</h1>
        <p className="text-white/50 mb-8 max-w-sm">
          The page you&apos;re looking for doesn&apos;t exist. Try the home page.
        </p>
        <Link
          href="/"
          className="rounded-xl bg-violet-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-violet-900/40 transition hover:bg-violet-500"
        >
          Back to autotune →
        </Link>
      </div>
    </div>
  );
}
