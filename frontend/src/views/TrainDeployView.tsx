import { useEffect, useState } from "react";
import {
  createTrainingRun,
  promoteModel,
  rollbackDeployment,
  saveTrainingRecipe,
  startCanary,
  stopTrainingRun,
  uploadArchitecture,
} from "../api/mlops";
import { LINES } from "../app/constants";
import type { BusyAction, LineId } from "../app/types";
import { cls, formatKoreaTimestamp, pickCanaryModel, pickProductionModel, pickStagingModel, recipeDraftFrom } from "../app/utils";
import type { DashboardResponse, ModelVersion, TrainingRecipe } from "../types/mlops";
import { Badge, Card, MessageBanner } from "../components/ui";

const TRAIN_SAMPLE_PRESETS = [
  { id: "full", label: "전체 데이터", trainSamples: null, valSamples: null },
  { id: "quick", label: "빠른 검증 · train 200 / val 80", trainSamples: 200, valSamples: 80 },
  { id: "small", label: "소형 · train 600 / val 160", trainSamples: 600, valSamples: 160 },
  { id: "medium", label: "중형 · train 1200 / val 240", trainSamples: 1200, valSamples: 240 },
] as const;

type TrainSamplePresetId = (typeof TRAIN_SAMPLE_PRESETS)[number]["id"];

function DeploymentCard({
  title,
  model,
  subtitle,
}: {
  title: string;
  model: ModelVersion | null;
  subtitle: string;
}) {
  return (
    <div className="ui-panel-muted min-w-0 p-4">
      <div className="text-sm font-black text-slate-500">{title}</div>
      <div className="mt-1 truncate text-base font-black text-slate-950">{model?.name ?? model?.id ?? "-"}</div>
      <div className="mt-1 text-sm font-semibold text-slate-600">{subtitle}</div>
      <div className="mt-2 text-sm font-bold text-slate-700">
        F1 {model?.metrics.f1 ?? "-"} / {model?.metrics.latency_ms ?? "-"}ms
      </div>
    </div>
  );
}

export function TrainDeployView({
  dashboard,
  onRefresh,
}: {
  dashboard: DashboardResponse;
  onRefresh: () => Promise<void>;
}) {
  const productionModel = pickProductionModel(dashboard);
  const stagingModel = pickStagingModel(dashboard);
  const canaryModel = pickCanaryModel(dashboard);
  const candidateModel = canaryModel ?? stagingModel ?? dashboard.model_versions[0] ?? null;
  const recipes = dashboard.training_recipes ?? [];
  const firstRecipe = recipes[0];
  const activeRun = dashboard.training_runs.find((run) => ["preparing", "running", "stopping"].includes(run.status ?? ""));
  const latestRun = activeRun ?? dashboard.training_runs[0] ?? null;

  const [busyAction, setBusyAction] = useState<BusyAction>(null);
  const [message, setMessage] = useState("");
  const [selectedDatasetId, setSelectedDatasetId] = useState(dashboard.active_dataset_id);
  const [selectedSamplePresetId, setSelectedSamplePresetId] = useState<TrainSamplePresetId>("full");
  const [selectedBaseModelId, setSelectedBaseModelId] = useState(productionModel?.id ?? dashboard.model_versions[0]?.id ?? "");
  const [selectedRecipeId, setSelectedRecipeId] = useState(firstRecipe?.id ?? "");
  const [selectedDeployModelId, setSelectedDeployModelId] = useState(candidateModel?.id ?? "");
  const [selectedCanaryLine, setSelectedCanaryLine] = useState<LineId>("LINE-B");
  const [modelName, setModelName] = useState(() => formatKoreaTimestamp());
  const [epochCount, setEpochCount] = useState(firstRecipe?.default_epochs ?? 3);
  const [recipeDraft, setRecipeDraft] = useState<TrainingRecipe>(() => recipeDraftFrom(firstRecipe));
  const [architectureKind, setArchitectureKind] = useState<"gate" | "heatmap">("gate");
  const [architectureName, setArchitectureName] = useState("");
  const [architectureFile, setArchitectureFile] = useState<File | null>(null);

  useEffect(() => {
    if (!dashboard.dataset_versions.some((dataset) => dataset.id === selectedDatasetId)) {
      setSelectedDatasetId(dashboard.active_dataset_id);
    }
  }, [dashboard.active_dataset_id, dashboard.dataset_versions, selectedDatasetId]);

  const selectedRecipe = recipes.find((recipe) => recipe.id === selectedRecipeId) ?? firstRecipe;
  const selectedSamplePreset =
    TRAIN_SAMPLE_PRESETS.find((preset) => preset.id === selectedSamplePresetId) ?? TRAIN_SAMPLE_PRESETS[0];
  const selectedDeployModel = dashboard.model_versions.find((model) => model.id === selectedDeployModelId) ?? candidateModel;
  const runProgress = Math.min(100, Math.max(0, latestRun?.progress ?? 0));
  const runTone = latestRun?.status === "completed" ? "green" : latestRun?.status === "failed" ? "red" : activeRun ? "amber" : "blue";
  const runMetrics = latestRun?.final_metrics ?? {};
  const hasBatchProgress = runMetrics.batch !== undefined && runMetrics.total_batches !== undefined;

  function selectRecipe(recipe: TrainingRecipe) {
    setSelectedRecipeId(recipe.id);
    setRecipeDraft(recipeDraftFrom(recipe));
    setEpochCount(recipe.default_epochs ?? recipe.epochs ?? 3);
  }

  async function handleSaveRecipe() {
    try {
      setBusyAction("train");
      setMessage("");
      const saved = await saveTrainingRecipe({ ...recipeDraft, default_epochs: epochCount, epochs: epochCount });
      const recipe = (saved as { recipe?: TrainingRecipe }).recipe;
      if (recipe?.id) setSelectedRecipeId(recipe.id);
      setMessage("레시피를 저장했습니다.");
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "레시피 저장에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleStartTraining() {
    if (!selectedDatasetId || !selectedBaseModelId || !selectedRecipeId) {
      setMessage("데이터셋, 기준 모델, 레시피를 선택하세요.");
      return;
    }

    try {
      setBusyAction("train");
      setMessage("");
      await createTrainingRun({
        modelName: modelName.trim() || formatKoreaTimestamp(),
        knownModelIds: dashboard.model_versions.map((model) => model.id),
        datasetVersionId: selectedDatasetId,
        baseModelVersionId: selectedBaseModelId,
        recipeId: selectedRecipeId,
        targetLine: selectedCanaryLine,
        epochs: epochCount,
        batchSize: recipeDraft.batch_size,
        learningRate: recipeDraft.learning_rate,
        optimizer: recipeDraft.optimizer,
        maxTrainSamples: selectedSamplePreset.trainSamples,
        maxValSamples: selectedSamplePreset.valSamples,
      });
      setModelName(formatKoreaTimestamp());
      setMessage("학습을 시작했습니다.");
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "학습 시작에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleStopTraining() {
    try {
      setBusyAction("stop");
      setMessage("");
      await stopTrainingRun(activeRun?.id);
      setMessage("중지 요청을 보냈습니다.");
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "학습 중지에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleUploadArchitecture() {
    if (!architectureFile || !architectureName.trim()) return;
    try {
      setBusyAction("train");
      setMessage("");
      await uploadArchitecture({ file: architectureFile, kind: architectureKind, name: architectureName.trim() });
      setArchitectureName("");
      setArchitectureFile(null);
      setMessage("아키텍처를 등록했습니다.");
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "아키텍처 등록에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleStartCanary() {
    if (!selectedDeployModel) {
      setMessage("시험 모델을 선택하세요.");
      return;
    }
    try {
      setBusyAction("canary");
      setMessage("");
      await startCanary(selectedDeployModel.id, selectedCanaryLine);
      setMessage(`${selectedCanaryLine}에서 시험 라인을 시작했습니다.`);
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "시험 라인 시작에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleApprove() {
    if (!selectedDeployModel) return;
    try {
      setBusyAction("approve");
      setMessage("");
      await promoteModel(selectedDeployModel.id, "production");
      setMessage("Production 배포가 완료되었습니다.");
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "배포 승인에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  async function handleRollback() {
    try {
      setBusyAction("rollback");
      setMessage("");
      await rollbackDeployment();
      setMessage("이전 Production으로 롤백했습니다.");
      await onRefresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "롤백에 실패했습니다.");
    } finally {
      setBusyAction(null);
    }
  }

  return (
    <div className="grid h-full min-h-0 grid-cols-[1.05fr_.95fr] gap-4">
        <Card className="flex min-h-0 flex-col p-5">
          <div className="mb-4 text-xl font-black text-slate-950">학습 구성</div>

          {message ? <div className="mb-4"><MessageBanner message={message} tone="blue" /></div> : null}

          <div className="min-h-0 overflow-hidden">
            <div className="space-y-4">
              <div className="ui-panel-muted p-4">
                <div className="mb-4 text-base font-black text-slate-950">기본 선택</div>
                <div className="grid grid-cols-[1.08fr_.92fr] gap-4">
                  <div className="grid grid-cols-2 gap-3">
                    <label>
                      <div className="text-sm font-black text-slate-500">데이터</div>
                      <select value={selectedDatasetId} onChange={(event) => setSelectedDatasetId(event.target.value)} className="ui-control mt-1 w-full">
                        {dashboard.dataset_versions.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name} · {dataset.sample_count}</option>)}
                      </select>
                    </label>
                    <label>
                      <div className="text-sm font-black text-slate-500">학습 규모</div>
                      <select value={selectedSamplePresetId} onChange={(event) => setSelectedSamplePresetId(event.target.value as TrainSamplePresetId)} className="ui-control mt-1 w-full">
                        {TRAIN_SAMPLE_PRESETS.map((preset) => <option key={preset.id} value={preset.id}>{preset.label}</option>)}
                      </select>
                    </label>
                    <label>
                      <div className="text-sm font-black text-slate-500">기준 모델</div>
                      <select value={selectedBaseModelId} onChange={(event) => setSelectedBaseModelId(event.target.value)} className="ui-control mt-1 w-full">
                        {dashboard.model_versions.map((model) => <option key={model.id} value={model.id}>{model.id} · {model.status}</option>)}
                      </select>
                    </label>
                    <label>
                      <div className="text-sm font-black text-slate-500">모델 이름</div>
                      <input value={modelName} onChange={(event) => setModelName(event.target.value)} className="ui-control mt-1 w-full" />
                    </label>
                    <label>
                      <div className="text-sm font-black text-slate-500">반복</div>
                      <input type="number" min={1} max={200} value={epochCount} onChange={(event) => setEpochCount(Math.max(1, Number(event.target.value)))} className="ui-control mt-1 w-full" />
                    </label>
                  </div>

                  <div className="ui-panel p-4">
                    <div className="mb-4 text-sm font-black text-slate-700">새로운 모델 불러오기</div>
                    <div className="grid grid-cols-[.58fr_1fr] gap-3">
                      <select value={architectureKind} onChange={(event) => setArchitectureKind(event.target.value as "gate" | "heatmap")} className="ui-control">
                        <option value="gate">gate</option><option value="heatmap">heatmap</option>
                      </select>
                      <input value={architectureName} onChange={(event) => setArchitectureName(event.target.value)} className="ui-control" placeholder="Architecture name" />
                    </div>
                    <div className="mt-3 grid grid-cols-[1fr_88px] gap-3">
                      <input type="file" accept=".json,.yaml,.yml,.txt,.py" onChange={(event) => setArchitectureFile(event.target.files?.[0] ?? null)} className="ui-control min-w-0" />
                      <button onClick={handleUploadArchitecture} disabled={busyAction !== null || !architectureFile || !architectureName.trim()} className="ui-button ui-button-primary">
                        등록
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              <div className="ui-panel p-4">
                <div className="mb-4 text-base font-black text-slate-950">레시피 선택 및 수정</div>
                <div className="grid grid-cols-[.9fr_1.1fr] gap-4">
                  <div className="ui-panel max-h-48 overflow-auto p-2">
                    <div className="space-y-3">
                      {recipes.map((recipe) => (
                        <button key={recipe.id} type="button" onClick={() => selectRecipe(recipe)} className={cls("w-full rounded-[10px] border p-2 text-left", recipe.id === selectedRecipeId ? "border-blue-600 bg-blue-50" : "border-[#d8dee6] bg-white")}>
                          <div className="text-sm font-black text-slate-950">{recipe.name}</div>
                          <div className="mt-1 grid grid-cols-2 gap-1 text-xs font-semibold text-slate-600">
                            <span>batch {recipe.batch_size}</span><span>lr {recipe.learning_rate}</span><span>{recipe.optimizer}</span><span>{recipe.scheduler}</span>
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="ui-panel-muted p-4">
                    <div className="grid grid-cols-3 gap-3">
                      <input className="ui-control" value={recipeDraft.name} onChange={(event) => setRecipeDraft({ ...recipeDraft, name: event.target.value })} placeholder="name" />
                      <select className="ui-control" value={recipeDraft.optimizer} onChange={(event) => setRecipeDraft({ ...recipeDraft, optimizer: event.target.value })}>
                        {["AdamW", "Adam", "SGD"].map((optimizer) => <option key={optimizer}>{optimizer}</option>)}
                      </select>
                      <input className="ui-control" type="number" min={1} value={recipeDraft.batch_size} onChange={(event) => setRecipeDraft({ ...recipeDraft, batch_size: Number(event.target.value) })} placeholder="batch" />
                      <input className="ui-control" type="number" step="0.0001" value={recipeDraft.learning_rate} onChange={(event) => setRecipeDraft({ ...recipeDraft, learning_rate: Number(event.target.value) })} placeholder="lr" />
                      <input className="ui-control" type="number" step="0.001" value={recipeDraft.weight_decay} onChange={(event) => setRecipeDraft({ ...recipeDraft, weight_decay: Number(event.target.value) })} placeholder="weight decay" />
                      <select className="ui-control" value={recipeDraft.scheduler} onChange={(event) => setRecipeDraft({ ...recipeDraft, scheduler: event.target.value })}>
                        {["cosine", "step", "none"].map((scheduler) => <option key={scheduler}>{scheduler}</option>)}
                      </select>
                    </div>
                    <button onClick={handleSaveRecipe} disabled={busyAction !== null} className="ui-button ui-button-dark mt-3 w-full py-2">
                      레시피 저장
                    </button>
                  </div>
                </div>
              </div>

            <div className="ui-panel p-4">
              <div className="mb-4 flex items-center justify-between gap-4">
                <div className="text-base font-black text-slate-950">학습 상태</div>
                <Badge tone={runTone}>{latestRun?.status ?? "not-created"}</Badge>
              </div>
              <div className="h-2.5 overflow-hidden rounded-full bg-slate-200">
                <div className="h-full rounded-full bg-blue-600 transition-all" style={{ width: `${runProgress}%` }} />
              </div>
              <div className="mt-3 flex justify-between text-sm font-bold text-slate-600">
                <span>{runProgress}%</span><span>{latestRun?.device ?? "cpu"}</span><span>{latestRun?.name ?? selectedRecipe?.name ?? "-"}</span>
              </div>
              <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs font-bold text-slate-500">
                <span>{latestRun?.current_step ?? "IDLE"}</span>
                {hasBatchProgress ? (
                  <span>
                    batch {runMetrics.batch}/{runMetrics.total_batches}
                    {runMetrics.batch_loss !== undefined ? ` / loss ${runMetrics.batch_loss.toFixed(4)}` : ""}
                    {runMetrics.running_loss !== undefined ? ` / avg ${runMetrics.running_loss.toFixed(4)}` : ""}
                  </span>
                ) : null}
                {runMetrics.val_f1 !== undefined || runMetrics.val_loss !== undefined ? (
                  <span>
                    F1 {runMetrics.val_f1?.toFixed?.(4) ?? "-"} / loss {runMetrics.val_loss?.toFixed?.(4) ?? "-"}
                  </span>
                ) : null}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <button onClick={handleStartTraining} disabled={busyAction !== null || Boolean(activeRun)} className="ui-button ui-button-primary py-2.5">
                {busyAction === "train" ? "학습 시작 중" : "학습 시작"}
              </button>
              <button onClick={handleStopTraining} disabled={busyAction !== null || !activeRun} className="ui-button ui-button-danger py-2.5">
                {busyAction === "stop" ? "중지 요청 중" : "학습 중지"}
              </button>
            </div>
            </div>
          </div>
        </Card>

        <Card className="flex min-h-0 flex-col p-5">
          <div className="mb-4 text-xl font-black text-slate-950">배포</div>
          <div className="min-h-0 overflow-auto pr-1">
            <div className="ui-panel-muted grid gap-4 p-4">
              <label>
                <div className="flex items-center justify-between gap-2 text-sm font-black text-slate-500">
                  <span>배포 모델</span>
                </div>
                <select value={selectedDeployModelId} onChange={(event) => setSelectedDeployModelId(event.target.value)} className="ui-control mt-2 w-full">
                  {dashboard.model_versions.map((model) => <option key={model.id} value={model.id}>{model.name ?? model.id} · {model.status}</option>)}
                </select>
              </label>
              <label>
                <div className="flex items-center justify-between gap-2 text-sm font-black text-slate-500">
                  <span>시험 라인</span>
                </div>
                <select value={selectedCanaryLine} onChange={(event) => setSelectedCanaryLine(event.target.value as LineId)} className="ui-control mt-2 w-full">
                  {LINES.map((line) => <option key={line}>{line}</option>)}
                </select>
              </label>
            </div>

            <div className="mt-4 grid gap-4">
              <div className="grid grid-cols-3 gap-4">
                <DeploymentCard title="Production" model={productionModel} subtitle="운영 중" />
                <DeploymentCard title="Staging" model={stagingModel} subtitle="검증 대기" />
                <DeploymentCard title="시험" model={canaryModel} subtitle={dashboard.deployment.canary_line ? `${dashboard.deployment.canary_line} 검증 중` : "없음"} />
              </div>
              <div className="grid grid-cols-3 gap-4">
                <button onClick={handleStartCanary} disabled={busyAction !== null} className="ui-button ui-button-neutral">
                  {busyAction === "canary" ? "시작 중" : "시험 라인 시작"}
                </button>
                <button onClick={handleApprove} disabled={busyAction !== null} className="ui-button ui-button-primary">
                  {busyAction === "approve" ? "승인 중" : "배포 승인"}
                </button>
                <button onClick={handleRollback} disabled={busyAction !== null} className="ui-button ui-button-danger">
                  {busyAction === "rollback" ? "롤백 중" : "롤백"}
                </button>
              </div>
            </div>
          </div>
        </Card>
    </div>
  );
}

