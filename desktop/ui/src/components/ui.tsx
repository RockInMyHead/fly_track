import { useEffect, type ButtonHTMLAttributes, type HTMLAttributes, type ReactNode } from "react";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "secondary" | "outline" | "ghost" | "destructive" | "accent";
  size?: "sm" | "md" | "lg" | "icon";
};

export function Button({ className, variant = "default", size = "md", ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md font-medium transition-colors",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60 disabled:pointer-events-none disabled:opacity-45",
        {
          default: "bg-primary text-primary-foreground hover:bg-primary/90",
          accent: "bg-accent text-primary-foreground hover:bg-accent/90",
          secondary: "bg-muted text-foreground hover:bg-muted/70",
          outline: "border bg-transparent hover:bg-muted/60",
          ghost: "hover:bg-muted/60",
          destructive: "bg-destructive text-white hover:bg-destructive/90",
        }[variant],
        { sm: "h-8 px-3 text-xs", md: "h-10 px-4 text-sm", lg: "h-12 px-6 text-base", icon: "h-9 w-9" }[size],
        className,
      )}
      {...props}
    />
  );
}

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("rounded-lg border bg-card/80 shadow-sm backdrop-blur", className)} {...props} />;
}

export function CardHeader({ title, icon, right, className }: { title: ReactNode; icon?: ReactNode; right?: ReactNode; className?: string }) {
  return (
    <div className={cn("flex items-center justify-between gap-3 border-b px-5 py-3.5", className)}>
      <div className="flex items-center gap-2.5 text-sm font-semibold tracking-wide">
        {icon && <span className="text-primary">{icon}</span>}
        {title}
      </div>
      {right}
    </div>
  );
}

export function Progress({ value, className, tone = "primary" }: { value: number; className?: string; tone?: "primary" | "accent" | "success" }) {
  return (
    <div className={cn("h-2 w-full overflow-hidden rounded-full bg-muted", className)}>
      <div
        className={cn("h-full rounded-full transition-[width] duration-500", {
          primary: "bg-primary",
          accent: "bg-accent",
          success: "bg-success",
        }[tone])}
        style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
      />
    </div>
  );
}

export function Badge({ children, tone = "muted", className }: { children: ReactNode; tone?: "muted" | "primary" | "success" | "warning" | "destructive"; className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11px] font-semibold",
        {
          muted: "bg-muted text-muted-foreground",
          primary: "bg-primary/15 text-primary",
          success: "bg-success/15 text-success",
          warning: "bg-warning/15 text-warning",
          destructive: "bg-destructive/15 text-destructive",
        }[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function Dialog({
  open,
  onClose,
  title,
  description,
  children,
  footer,
  wide,
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6 backdrop-blur-sm" onMouseDown={onClose}>
      <div
        className={cn("flex max-h-[85vh] w-full flex-col rounded-xl border bg-card shadow-2xl", wide ? "max-w-3xl" : "max-w-lg")}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4 border-b px-6 py-4">
          <div>
            <div className="text-base font-semibold">{title}</div>
            {description && <div className="mt-1 text-sm text-muted-foreground">{description}</div>}
          </div>
          <Button variant="ghost" size="icon" onClick={onClose} aria-label="Закрыть">
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="min-h-0 flex-1 overflow-auto px-6 py-4">{children}</div>
        {footer && <div className="flex justify-end gap-2 border-t px-6 py-3">{footer}</div>}
      </div>
    </div>
  );
}
