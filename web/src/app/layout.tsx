import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Uzcosmos Change Detection API",
  description: "AI-Based Building Change Detection for Cadastral Monitoring",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <main className="min-h-screen relative overflow-hidden">
          {/* Subtle Grid Background Pattern */}
          <div className="absolute inset-0 z-0 opacity-20 pointer-events-none" 
               style={{backgroundImage: 'linear-gradient(#ffffff0a 1px, transparent 1px), linear-gradient(90deg, #ffffff0a 1px, transparent 1px)', backgroundSize: '40px 40px'}}>
          </div>
          
          <nav className="relative z-10 w-full px-8 py-6 flex items-center justify-between border-b border-white/10 bg-black/20 backdrop-blur-sm">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center font-bold text-white shadow-[0_0_15px_rgba(0,102,255,0.6)]">
                UZ
              </div>
              <h1 className="text-xl font-semibold tracking-wide">Uzcosmos Change Detect</h1>
            </div>
          </nav>
          
          <div className="relative z-10">
            {children}
          </div>
        </main>
      </body>
    </html>
  );
}
