import Link from "next/link";

export default function NotFound() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 text-center text-ink">

      <div className="relative z-10">
        <div className="text-8xl font-bold text-rust-300/30 mb-4">404</div>
        <h1 className="text-2xl font-bold text-ink mb-3">Page not found</h1>
        <p className="text-muted mb-8 max-w-sm">
          The page you&apos;re looking for doesn&apos;t exist. Try the home page.
        </p>
        <Link
          href="/"
          className="rounded-md bg-rust-600 px-6 py-3 text-sm font-semibold text-ink  transition hover:bg-rust-500"
        >
          Back to autotune →
        </Link>
      </div>
    </div>
  );
}
