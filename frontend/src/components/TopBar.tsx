import { ChevronDown, Factory } from "lucide-react";
import { useState } from "react";
import { adminTabs, audienceItems, LINES } from "../app/constants";
import type { AdminTab, Audience, LineId } from "../app/types";
import { cls } from "../app/utils";

function AudienceSwitch({
  audience,
  setAudience,
}: {
  audience: Audience;
  setAudience: (audience: Audience) => void;
}) {
  return (
    <div className="inline-flex h-[50px] rounded-[14px] border border-[#d8dee6] bg-white p-1 shadow-toss">
      {audienceItems.map((item) => {
        const active = audience === item.key;
        const Icon = item.icon;
        return (
          <button
            key={item.key}
            onClick={() => setAudience(item.key)}
            className={cls(
              "flex h-full min-w-[106px] items-center justify-center gap-2 rounded-[11px] px-4 text-sm font-bold transition",
              active ? "bg-[#2563eb] text-white" : "text-[#475569] hover:bg-[#eef2f7]"
            )}
          >
            <Icon className="h-4 w-4" />
            {item.label}
          </button>
        );
      })}
    </div>
  );
}

function ContextControls({
  audience,
  adminTab,
  setAdminTab,
  selectedLine,
  setSelectedLine,
}: {
  audience: Audience;
  adminTab: AdminTab;
  setAdminTab: (tab: AdminTab) => void;
  selectedLine: LineId;
  setSelectedLine: (line: LineId) => void;
}) {
  const [lineOpen, setLineOpen] = useState(false);

  if (audience === "admin") {
    return (
      <div className="inline-flex h-[50px] rounded-[14px] border border-[#d8dee6] bg-white p-1 shadow-toss">
        {adminTabs.map((item) => {
          const active = adminTab === item.key;
          const Icon = item.icon;
          return (
            <button
              key={item.key}
              onClick={() => setAdminTab(item.key)}
              className={cls(
                "flex h-full items-center gap-2 rounded-[11px] px-4 text-sm font-bold transition",
                active ? "bg-[#111827] text-white" : "text-[#475569] hover:bg-[#eef2f7]"
              )}
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </button>
          );
        })}
      </div>
    );
  }

  return (
    <div className="relative">
      <button
        onClick={() => setLineOpen((value) => !value)}
        className="inline-flex h-[50px] items-center gap-2 rounded-[14px] border border-[#d8dee6] bg-white px-5 text-sm font-bold text-[#111827] shadow-toss"
      >
        <Factory className="h-4 w-4" />
        {selectedLine}
        <ChevronDown className={cls("h-4 w-4 transition", lineOpen && "rotate-180")} />
      </button>

      {lineOpen ? (
        <div className="absolute left-0 top-[56px] z-30 w-[170px] rounded-[14px] border border-[#d8dee6] bg-white p-2 shadow-toss">
          {LINES.map((line) => (
            <button
              key={line}
              onClick={() => {
                setSelectedLine(line);
                setLineOpen(false);
              }}
              className={cls(
                "flex w-full rounded-[11px] px-3 py-3 text-left text-sm font-bold transition",
                selectedLine === line ? "bg-[#eff6ff] text-[#2563eb]" : "text-[#475569] hover:bg-[#eef2f7]"
              )}
            >
              {line}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function TopBar({
  audience,
  setAudience,
  adminTab,
  setAdminTab,
  selectedLine,
  setSelectedLine,
}: {
  audience: Audience;
  setAudience: (audience: Audience) => void;
  adminTab: AdminTab;
  setAdminTab: (tab: AdminTab) => void;
  selectedLine: LineId;
  setSelectedLine: (line: LineId) => void;
}) {
  return (
    <header className="sticky top-0 z-20 border-b border-[#d8dee6] bg-white/95 px-6 py-4 backdrop-blur">
      <div className="flex items-center justify-between gap-6">
        <div className="flex h-[50px] items-center text-[24px] font-black tracking-tight text-[#111827]">
          (주)철강왕 MLOps Dashboard
        </div>

        <div className="flex items-center gap-4">
          <ContextControls
            audience={audience}
            adminTab={adminTab}
            setAdminTab={setAdminTab}
            selectedLine={selectedLine}
            setSelectedLine={setSelectedLine}
          />
          <AudienceSwitch audience={audience} setAudience={setAudience} />
        </div>
      </div>
    </header>
  );
}
