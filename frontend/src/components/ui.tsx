import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import type { Tone } from "../app/types";
import { cls } from "../app/utils";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <section className={cls("rounded-[16px] border border-[#d8dee6] bg-white shadow-toss", className)}>
      {children}
    </section>
  );
}

export function DarkCard({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <section className={cls("rounded-[16px] border border-slate-800 bg-[#08111f] shadow-[0_16px_36px_rgba(0,0,0,0.28)]", className)}>
      {children}
    </section>
  );
}

export function Badge({
  children,
  tone = "slate",
  dark = false,
}: {
  children: ReactNode;
  tone?: Tone;
  dark?: boolean;
}) {
  const toneMap = dark
    ? {
        slate: "border-slate-700 bg-slate-900 text-slate-200",
        blue: "border-sky-800 bg-sky-950 text-sky-200",
        green: "border-emerald-800 bg-emerald-950 text-emerald-200",
        amber: "border-amber-800 bg-amber-950 text-amber-200",
        red: "border-rose-800 bg-rose-950 text-rose-200",
      }
    : {
        slate: "border-[#d8dee6] bg-[#eef2f7] text-[#475569]",
        blue: "border-[#bfdbfe] bg-[#eff6ff] text-[#1d4ed8]",
        green: "border-[#a7f3d0] bg-[#ecfdf5] text-[#047857]",
        amber: "border-[#fde68a] bg-[#fffbeb] text-[#b45309]",
        red: "border-[#fecaca] bg-[#fef2f2] text-[#b91c1c]",
      };

  return (
    <span className={cls("inline-flex items-center rounded-[10px] border px-3 py-1 text-sm font-bold", toneMap[tone])}>
      {children}
    </span>
  );
}

export function Stat({
  label,
  value,
  sub,
  icon: Icon,
  tone = "slate",
  dark = false,
}: {
  label: string;
  value: string;
  sub?: string;
  icon?: LucideIcon;
  tone?: Tone;
  dark?: boolean;
}) {
  const toneMap = dark
    ? {
        slate: "border-slate-800 bg-slate-950 text-slate-100",
        blue: "border-slate-800 bg-slate-950 text-slate-100",
        green: "border-slate-800 bg-slate-950 text-slate-100",
        amber: "border-slate-800 bg-slate-950 text-slate-100",
        red: "border-slate-800 bg-slate-950 text-slate-100",
      }
    : {
        slate: "border-[#d8dee6] bg-white text-[#111827]",
        blue: "border-[#bfdbfe] bg-[#eff6ff] text-[#111827]",
        green: "border-[#a7f3d0] bg-[#ecfdf5] text-[#111827]",
        amber: "border-[#fde68a] bg-[#fffbeb] text-[#111827]",
        red: "border-[#fecaca] bg-[#fef2f2] text-[#111827]",
      };

  return (
    <div className={cls("min-w-0 rounded-[14px] border p-4", toneMap[tone])}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-sm font-bold text-[#475569]">{label}</div>
          <div className="mt-2 truncate text-2xl font-black leading-tight">{value}</div>
          {sub ? <div className={cls("mt-1 truncate text-sm", dark ? "text-slate-400" : "text-[#475569]")}>{sub}</div> : null}
        </div>
        {Icon ? <Icon className="mt-1 h-5 w-5 shrink-0 text-[#64748b]" /> : null}
      </div>
    </div>
  );
}

export function MessageBanner({
  message,
  tone = "slate",
  dark = false,
}: {
  message: string;
  tone?: Tone;
  dark?: boolean;
}) {
  const toneMap = dark
    ? {
        slate: "border-slate-700 bg-slate-950 text-slate-200",
        blue: "border-sky-900 bg-sky-950 text-sky-200",
        green: "border-emerald-900 bg-emerald-950 text-emerald-200",
        amber: "border-amber-900 bg-amber-950 text-amber-200",
        red: "border-rose-900 bg-rose-950 text-rose-200",
      }
    : {
        slate: "border-[#d8dee6] bg-white text-[#475569]",
        blue: "border-[#bfdbfe] bg-[#eff6ff] text-[#1d4ed8]",
        green: "border-[#a7f3d0] bg-[#ecfdf5] text-[#047857]",
        amber: "border-[#fde68a] bg-[#fffbeb] text-[#b45309]",
        red: "border-[#fecaca] bg-[#fef2f2] text-[#b91c1c]",
      };

  return <div className={cls("rounded-[14px] border px-4 py-3 text-sm font-semibold", toneMap[tone])}>{message}</div>;
}

export function EmptyState({ children, dark = false }: { children: ReactNode; dark?: boolean }) {
  return (
    <div className={cls("rounded-[14px] border p-5 text-center text-sm font-semibold", dark ? "border-slate-800 text-slate-400" : "border-[#d8dee6] text-[#64748b]")}>
      {children}
    </div>
  );
}
