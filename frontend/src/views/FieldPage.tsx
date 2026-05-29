import { useEffect, useState } from "react";
import { CheckCircle2, Clock3, Cpu, ShieldAlert } from "lucide-react";
import { predictAnomaly } from "../api/anomaly";
import { uploadFeedback } from "../api/mlops";
import type { PredictResponse } from "../types/anomaly";
import type { LineId, Tone, ViewMode } from "../app/types";
import { clamp01, formatMetric, formatPercent, toDisplayDecision } from "../app/utils";
import { DarkCard, MessageBanner, Stat } from "../components/ui";
import { cls } from "../app/utils";

function ActionButton({
  title,
  hint,
  tone,
  disabled,
  onClick,
}: {
  title: string;
  hint: string;
  tone: Tone;
  disabled?: boolean;
  onClick: () => void;
}) {
  const toneMap: Record<Tone, string> = {
    blue: "border-rose-800 bg-rose-800 text-white hover:bg-rose-700",
    red: "border-red-800 bg-red-800 text-white hover:bg-red-700",
    amber: "border-orange-800 bg-orange-800 text-white hover:bg-orange-700",
    green: "border-rose-900 bg-rose-900 text-white hover:bg-rose-800",
    slate: "border-slate-700 bg-slate-700 text-white hover:bg-slate-600",
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={cls(
        "flex aspect-square min-h-[210px] flex-col items-center justify-center rounded-[14px] border px-5 text-center transition active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-50",
        toneMap[tone]
      )}
    >
      <span className="text-[44px] font-black leading-[1.02] tracking-tight">{title}</span>
      <span className="mt-3 text-[18px] font-bold leading-tight opacity-85">{hint}</span>
    </button>
  );
}

function ModeSwitch({ mode, setMode }: { mode: ViewMode; setMode: (mode: ViewMode) => void }) {
  const labels: Record<ViewMode, string> = {
    raw: "원본",
    heatmap: "이상 영역",
    overlay: "검사 오버레이",
  };

  return (
    <div className="grid h-full grid-cols-3 gap-4">
      {(["raw", "heatmap", "overlay"] as ViewMode[]).map((value) => (
        <button
          key={value}
          onClick={() => setMode(value)}
          className={cls(
            "rounded-[10px] border text-[18px] font-black tracking-wide transition",
            mode === value ? "border-blue-500 bg-blue-500 text-white" : "border-slate-800 bg-slate-950 text-slate-300"
          )}
        >
          {labels[value]}
        </button>
      ))}
    </div>
  );
}

function describeHeatmap(score: number | null) {
  if (score == null) return { value: "-", detail: "검사 전" };
  if (score >= 4) return { value: "높음", detail: `점수 ${formatMetric(score, 2)}` };
  if (score >= 2) return { value: "중간", detail: `점수 ${formatMetric(score, 2)}` };
  return { value: "낮음", detail: `점수 ${formatMetric(score, 2)}` };
}

function CameraViewport({
  mode,
  rawImageUrl,
  overlayImageUrl,
  heatmapImageUrl,
}: {
  mode: ViewMode;
  rawImageUrl: string;
  overlayImageUrl: string;
  heatmapImageUrl: string;
}) {
  const imageSrc = mode === "heatmap" && heatmapImageUrl ? heatmapImageUrl : rawImageUrl;
  const showOverlay = mode === "overlay" && rawImageUrl && overlayImageUrl;

  return (
    <div className="relative h-full min-h-0 overflow-hidden rounded-[14px] border border-slate-800 bg-[#040b17]">
      {!imageSrc ? (
        <div className="absolute inset-0 grid place-items-center text-base font-bold text-slate-400">
          이미지 업로드 후 검사 결과가 표시됩니다.
        </div>
      ) : (
        <>
          <img src={imageSrc} alt="inspection view" className="h-full w-full bg-slate-950 object-contain" />
          {showOverlay ? (
            <img src={overlayImageUrl} alt="heatmap overlay" className="pointer-events-none absolute inset-0 h-full w-full object-contain opacity-80" />
          ) : null}
        </>
      )}
    </div>
  );
}

export function FieldPage({
  selectedLine,
  onRefresh,
}: {
  selectedLine: LineId;
  onRefresh: () => Promise<void>;
}) {
  const [mode, setMode] = useState<ViewMode>("raw");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [rawImageUrl, setRawImageUrl] = useState("");
  const [overlayImageUrl, setOverlayImageUrl] = useState("");
  const [heatmapImageUrl, setHeatmapImageUrl] = useState("");
  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    return () => {
      if (rawImageUrl.startsWith("blob:")) URL.revokeObjectURL(rawImageUrl);
    };
  }, [rawImageUrl]);

  const decision = predictResult ? toDisplayDecision(predictResult.decision) : "대기";
  const confidence = predictResult ? clamp01(predictResult.gate_score) : 0;
  const latency = predictResult ? Math.round(predictResult.latency.total_latency_ms) : 0;
  const heatmapScore = predictResult?.heatmap_score ?? null;
  const heatmapDisplay = describeHeatmap(heatmapScore);

  async function handlePredict() {
    if (!selectedFile) {
      setMessage("먼저 이미지를 업로드하세요.");
      return;
    }

    try {
      setIsPredicting(true);
      setMessage("");
      const result = await predictAnomaly(selectedFile);
      setPredictResult(result);
      const overlayUrl = result.heatmap_overlay ? `data:image/png;base64,${result.heatmap_overlay}` : "";
      const heatmapUrl = result.normalized_score_heatmap ? `data:image/png;base64,${result.normalized_score_heatmap}` : overlayUrl;
      setOverlayImageUrl(overlayUrl);
      setHeatmapImageUrl(heatmapUrl);
      setMode(overlayUrl ? "overlay" : heatmapUrl ? "heatmap" : "raw");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "검사 요청에 실패했습니다.");
    } finally {
      setIsPredicting(false);
    }
  }

  async function handleFeedback(
    feedbackType: "false_positive" | "false_negative" | "confirmed_anomaly" | "needs_review",
    label: "normal" | "anomaly" | "unlabeled",
    comment: string
  ) {
    if (!selectedFile) {
      setMessage("피드백을 남기려면 검사 이미지를 먼저 선택하세요.");
      return;
    }

    try {
      await uploadFeedback({
        file: selectedFile,
        feedbackType,
        label,
        operator: "operator-01",
        comment,
        line: selectedLine,
        predictedLabel: predictResult?.decision ?? "",
        gateScore: predictResult?.gate_score,
        heatmapScore: predictResult?.heatmap_score,
      });
      setMessage("피드백이 저장되었습니다.");
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "피드백 저장에 실패했습니다.");
    }
  }

  return (
    <div className="relative grid h-full min-h-0 grid-cols-[1.28fr_0.72fr] gap-4">
      {message ? (
        <div className="pointer-events-none absolute right-4 top-4 z-30 w-[360px]">
          <MessageBanner message={message} tone="blue" />
        </div>
      ) : null}

      <DarkCard className="grid min-h-0 grid-rows-[auto_auto_1fr_auto_auto] gap-4 p-5">
        <div className="flex items-center justify-between gap-4">
          <div>
            <div className="text-2xl font-black text-slate-50">현장 검사</div>
          </div>
        </div>

        <div className="grid grid-cols-[1fr_220px] gap-4">
          <label className="flex min-h-[74px] min-w-0 cursor-pointer items-center justify-center rounded-[14px] border border-dashed border-slate-700 bg-slate-900 px-4 py-3 text-[22px] font-black text-slate-100 hover:bg-slate-800">
            <span className="max-w-full truncate">{selectedFile ? selectedFile.name : "이미지 업로드"}</span>
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (!file) return;
                if (rawImageUrl.startsWith("blob:")) URL.revokeObjectURL(rawImageUrl);
                setSelectedFile(file);
                setPredictResult(null);
                setMessage("");
                setOverlayImageUrl("");
                setHeatmapImageUrl("");
                setRawImageUrl(URL.createObjectURL(file));
                setMode("raw");
              }}
            />
          </label>
          <button
            onClick={handlePredict}
            disabled={isPredicting}
            className="rounded-[14px] border border-blue-500 bg-blue-500 px-6 py-3 text-[22px] font-black text-white disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isPredicting ? "검사 중" : "검사 실행"}
          </button>
        </div>

        <div className="min-h-0">
          <CameraViewport mode={mode} rawImageUrl={rawImageUrl} overlayImageUrl={overlayImageUrl} heatmapImageUrl={heatmapImageUrl} />
        </div>

        <div className="h-[78px]">
          <ModeSwitch mode={mode} setMode={setMode} />
        </div>

        <div className="grid grid-cols-4 gap-4">
          <Stat label="상태" value={isPredicting ? "검사 중" : decision} icon={CheckCircle2} tone={decision === "이상" ? "red" : "green"} dark />
          <Stat label="확률" value={formatPercent(confidence)} icon={Cpu} tone="blue" dark />
          <Stat label="응답" value={latency ? `${latency}ms` : "-"} icon={Clock3} tone="slate" dark />
          <Stat label="결함 의심도" value={heatmapDisplay.value} sub={heatmapDisplay.detail} icon={ShieldAlert} tone="amber" dark />
        </div>
      </DarkCard>

      <DarkCard className="grid min-h-0 grid-rows-[auto_1fr] gap-4 p-5">
        <div>
          <div className="text-2xl font-black text-slate-50">작업자 피드백</div>
        </div>

        <div className="grid grid-cols-2 gap-4 self-start">
          <ActionButton title="오탐" hint="정상인데 이상" tone="blue" onClick={() => handleFeedback("false_positive", "normal", "오탐 저장")} />
          <ActionButton title="미탐" hint="이상인데 정상" tone="red" onClick={() => handleFeedback("false_negative", "anomaly", "미탐 저장")} />
          <ActionButton title="이상" hint="불량 확정" tone="amber" onClick={() => handleFeedback("confirmed_anomaly", "anomaly", "이상 확정")} />
          <ActionButton title="보류" hint="관리자 확인" tone="slate" onClick={() => handleFeedback("needs_review", "unlabeled", "검토 보류")} />
        </div>
      </DarkCard>
    </div>
  );
}

