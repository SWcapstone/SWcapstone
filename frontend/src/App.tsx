import { useEffect, useState } from "react";
import { fetchDashboard } from "./api/mlops";
import type { AdminTab, Audience, LineId } from "./app/types";
import { TopBar } from "./components/TopBar";
import { MessageBanner } from "./components/ui";
import type { DashboardResponse } from "./types/mlops";
import { AdminPage } from "./views/AdminPage";
import { FieldPage } from "./views/FieldPage";

export default function App() {
  const [audience, setAudience] = useState<Audience>("field");
  const [adminTab, setAdminTab] = useState<AdminTab>("ops");
  const [selectedLine, setSelectedLine] = useState<LineId>("LINE-A");
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  async function refresh() {
    try {
      const response = await fetchDashboard();
      setDashboard(response);
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "대시보드를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  const hasActiveTraining = Boolean(
    dashboard?.training_runs.some((run) => ["preparing", "running", "stopping"].includes(run.status ?? ""))
  );

  useEffect(() => {
    if (!hasActiveTraining) return;
    const timer = window.setInterval(() => {
      void refresh();
    }, 1000);
    return () => window.clearInterval(timer);
  }, [hasActiveTraining]);

  return (
    <div className="h-screen w-screen overflow-hidden bg-[#f4f6f8] font-['Noto_Sans_KR','Segoe_UI','Malgun_Gothic','Apple_SD_Gothic_Neo','Arial','sans-serif'] text-[#111827]">
      <div className="flex h-full flex-col">
        <TopBar
          audience={audience}
          setAudience={setAudience}
          adminTab={adminTab}
          setAdminTab={setAdminTab}
          selectedLine={selectedLine}
          setSelectedLine={setSelectedLine}
        />

        <main className="min-h-0 flex-1 overflow-auto bg-[#f4f6f8] px-5 py-4">
          {loading ? <MessageBanner message="대시보드를 불러오는 중입니다." tone="slate" dark={audience === "field"} /> : null}
          {error ? <div className="mb-4"><MessageBanner message={error} tone="red" dark={audience === "field"} /></div> : null}
          {dashboard && audience === "field" ? <FieldPage selectedLine={selectedLine} onRefresh={refresh} /> : null}
          {dashboard && audience === "admin" ? <AdminPage adminTab={adminTab} dashboard={dashboard} onRefresh={refresh} /> : null}
        </main>
      </div>
    </div>
  );
}
