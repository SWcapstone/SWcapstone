import type { DashboardResponse } from "../types/mlops";
import { baseLineStatus, LINES } from "../app/constants";
import { asset, formatDate } from "../app/utils";
import { Badge, Card, EmptyState } from "../components/ui";
import type { Tone } from "../app/types";

type QueueItem = {
  title: string;
  detail: string;
  value: string;
  tone: Tone;
};

export function OpsView({ dashboard }: { dashboard: DashboardResponse }) {
  const deployCandidateCount = dashboard.model_versions.filter((model) => !["production", "failed", "stopped"].includes(model.status)).length;
  const warningLogCount = dashboard.logs.filter((log) => log.level === "warning" || log.level === "error").length;

  const lineStatus = LINES.map((lineId) => {
    const base = baseLineStatus[lineId];
    if (dashboard.deployment.canary_line === lineId) {
      return {
        lineId,
        ...base,
        state: "주의",
        alert: `Canary ${dashboard.deployment.canary_model_id ?? "-"}`,
      };
    }
    return { lineId, ...base };
  });

  const workQueue: QueueItem[] = [
    {
      title: "피드백 반영",
      detail: "버전 탭에서 데이터셋으로 묶기",
      value: `${dashboard.feedback_items.length}건`,
      tone: dashboard.feedback_items.length ? "amber" : "green",
    },
    {
      title: "배포 검토",
      detail: "학습 탭에서 검증 또는 승인",
      value: `${deployCandidateCount}건`,
      tone: deployCandidateCount ? "blue" : "green",
    },
    {
      title: "로그 확인",
      detail: "경고/오류 이벤트 우선 확인",
      value: `${warningLogCount}건`,
      tone: warningLogCount ? "red" : "green",
    },
  ];

  return (
    <div className="grid h-full min-h-0 grid-cols-[1.05fr_.95fr] gap-4">
        <div className="grid min-h-0 grid-rows-[auto_1fr] gap-4">
          <Card className="p-5">
            <div className="mb-4 text-xl font-black text-slate-950">확인할 일</div>
            <div className="grid grid-cols-3 gap-4">
              {workQueue.map((item) => (
                <div key={item.title} className="ui-panel-muted p-4">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <div className="text-base font-black text-slate-950">{item.title}</div>
                      <div className="mt-1 text-sm font-semibold text-slate-600">{item.detail}</div>
                    </div>
                    <Badge tone={item.tone}>{item.value}</Badge>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          <Card className="flex min-h-0 flex-col p-5">
            <div className="mb-4 text-xl font-black text-slate-950">라인 운영 상태</div>

            <div className="ui-panel min-h-0 overflow-auto">
              <table className="min-w-full text-sm">
                <thead className="sticky top-0 bg-[#f8fafc] text-slate-700">
                  <tr>
                    {["라인", "상태", "수율", "응답", "카메라", "알림"].map((heading) => (
                      <th key={heading} className="px-4 py-3 text-left text-sm font-black">
                        {heading}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {lineStatus.map((line) => (
                    <tr key={line.lineId} className="border-t border-[#e5e8eb] bg-white">
                      <td className="px-4 py-3 font-bold text-slate-900">{line.lineId}</td>
                      <td className="px-4 py-3">
                        <Badge tone={line.state === "주의" ? "amber" : "green"}>{line.state}</Badge>
                      </td>
                      <td className="px-4 py-3">{line.yieldRate}%</td>
                      <td className="px-4 py-3">{line.latency}ms</td>
                      <td className="px-4 py-3">{line.camera}</td>
                      <td className="px-4 py-3">{line.alert}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>

        <div className="grid min-h-0 grid-rows-[1fr_1fr] gap-4">
          <Card className="flex min-h-0 flex-col p-5">
            <div className="mb-4 flex items-center justify-between gap-4">
              <div className="text-xl font-black text-slate-950">운영 로그</div>
            </div>
            <div className="min-h-0 overflow-auto pr-1">
              <div className="space-y-4">
                {dashboard.logs.slice(0, 8).map((log) => (
                  <div key={log.id} className="ui-panel p-4">
                    <div className="flex items-start justify-between gap-4">
                      <div className="font-bold text-slate-900">{log.message}</div>
                      <Badge tone={log.level === "warning" ? "amber" : log.level === "error" ? "red" : "blue"}>{log.level}</Badge>
                    </div>
                    <div className="mt-2 text-sm font-semibold text-slate-500">{formatDate(log.time)}</div>
                  </div>
                ))}
                {!dashboard.logs.length ? <EmptyState>표시할 로그가 없습니다.</EmptyState> : null}
              </div>
            </div>
          </Card>

          <Card className="flex min-h-0 flex-col p-5">
            <div className="mb-4 text-xl font-black text-slate-950">최근 피드백</div>
            <div className="min-h-0 overflow-auto pr-1">
              <div className="space-y-4">
                {dashboard.feedback_items.slice(0, 6).map((item) => (
                  <div key={item.id} className="ui-panel grid grid-cols-[72px_1fr] gap-4 p-4">
                    <img src={asset(item.image_url)} alt={item.id} className="h-16 w-16 rounded-[10px] bg-slate-100 object-cover" />
                    <div className="min-w-0">
                      <div className="font-bold text-slate-900">{item.feedback_type}</div>
                      <div className="mt-1 truncate text-sm text-slate-600">{item.comment || "메모 없음"}</div>
                      <div className="mt-2 text-sm text-slate-500">{formatDate(item.created_at)}</div>
                    </div>
                  </div>
                ))}
                {!dashboard.feedback_items.length ? <EmptyState>최근 피드백이 없습니다.</EmptyState> : null}
              </div>
            </div>
          </Card>
        </div>
    </div>
  );
}
