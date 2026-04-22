/** App — root component wiring router, React Query, and WS context. */
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NavBar } from "./components/NavBar";
import { WSProvider } from "./context/WSContext";
import { IngestPage, StatsPage, GeneratePage, DBPage, HelpPage } from "./pages";

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <WSProvider>
        <BrowserRouter>
          <div className="min-h-screen bg-slate-50">
            <NavBar />
            <main className="max-w-6xl mx-auto p-6">
              <Routes>
                <Route path="/" element={<Navigate to="/ingest" replace />} />
                <Route path="/ingest" element={<IngestPage />} />
                <Route path="/stats" element={<StatsPage />} />
                <Route path="/generate" element={<GeneratePage />} />
                <Route path="/db" element={<DBPage />} />
                <Route path="/help" element={<HelpPage />} />
              </Routes>
            </main>
          </div>
        </BrowserRouter>
      </WSProvider>
    </QueryClientProvider>
  );
}
