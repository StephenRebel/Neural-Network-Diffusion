import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


STEP_KEYS = ("steps", "step", "epoch", "checkpoints", "batch_idx")
ACC_METRICS = ("acc", "accuracy")
SECONDARY_PRIORITY = ("generated_norm",)


def load_report(report_path):
    with open(report_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def find_step_column(frame):
    for column in STEP_KEYS:
        if column in frame.columns:
            return column
    return None


def build_progress_frame(entries):
    if not isinstance(entries, list) or not entries:
        return None

    frame = pd.DataFrame(entries)
    step_column = find_step_column(frame)
    if step_column is None:
        return None

    metric_columns = [
        column
        for column in frame.columns
        if column not in STEP_KEYS and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not metric_columns:
        return None

    rows = []
    base = frame.rename(columns={step_column: "steps"}).copy()
    for metric_name in metric_columns:
        metric_frame = base[["steps", metric_name]].dropna().copy()
        if metric_frame.empty:
            continue
        metric_frame["metric"] = metric_name
        metric_frame = metric_frame.rename(columns={metric_name: "value"})
        rows.append(metric_frame)

    if not rows:
        return None

    return pd.concat(rows, ignore_index=True)


def collect_progress_frames(report):
    progress_frames = []
    for key, value in report.items():
        if not key.endswith("_progress"):
            continue
        frame = build_progress_frame(value)
        if frame is not None:
            progress_frames.append((key, frame))
    return progress_frames


def split_metrics(frame):
    primary_frame = frame[frame["metric"].isin(["loss"])]
    secondary_frame = frame[frame["metric"].isin(ACC_METRICS)]
    if secondary_frame.empty:
        for metric_name in SECONDARY_PRIORITY:
            secondary_frame = frame[frame["metric"] == metric_name]
            if not secondary_frame.empty:
                break
    if secondary_frame.empty:
        secondary_frame = frame[~frame["metric"].isin(["loss"])]
    return primary_frame, secondary_frame


def resolve_output_path(report_path, progress_name, output_path, progress_count):
    if output_path is None:
        base_name, _ = os.path.splitext(report_path)
        return f"{base_name}_{progress_name}.png"

    if os.path.isdir(output_path):
        base_name = os.path.splitext(os.path.basename(report_path))[0]
        return os.path.join(output_path, f"{base_name}_{progress_name}.png")

    if progress_count == 1:
        return output_path

    root, ext = os.path.splitext(output_path)
    if ext:
        return f"{root}_{progress_name}{ext}"
    return f"{output_path}_{progress_name}.png"


def plot_metric_frame(ax, frame, title, ylabel):
    if frame.empty:
        ax.set_axis_off()
        return

    sns.lineplot(data=frame, x="steps", y="value", hue="metric", ax=ax, errorbar=None)
    ax.set_title(title)
    ax.set_xlabel("steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(title="metric")


def plot_progress_frame(report_path, progress_name, frame, output_path=None, progress_count=1):
    sns.set_theme(style="whitegrid", context="talk")

    primary_frame, secondary_frame = split_metrics(frame)
    chart_specs = []
    if not primary_frame.empty:
        chart_specs.append((primary_frame, f"{progress_name} - loss / attached metrics", "value"))
    if not secondary_frame.empty:
        title = f"{progress_name} - secondary metrics" if chart_specs else f"{progress_name} - metrics"
        chart_specs.append((secondary_frame, title, "value"))

    figure, axes = plt.subplots(
        1,
        len(chart_specs),
        figsize=(16 if len(chart_specs) > 1 else 10, 5),
        constrained_layout=True,
    )
    if len(chart_specs) == 1:
        axes = [axes]

    for axis, (chart_frame, title, ylabel) in zip(axes, chart_specs):
        plot_metric_frame(axis, chart_frame, title, ylabel)

    resolved_output_path = resolve_output_path(report_path, progress_name, output_path, progress_count)
    os.makedirs(os.path.dirname(resolved_output_path) or ".", exist_ok=True)
    figure.savefig(resolved_output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return resolved_output_path


def plot_report(report_path, output_path=None):
    report = load_report(report_path)
    progress_frames = collect_progress_frames(report)

    if not progress_frames:
        raise ValueError(f"No *_progress entries with numeric data were found in {report_path}")

    output_paths = []
    progress_count = len(progress_frames)
    for progress_name, frame in progress_frames:
        output_paths.append(
            plot_progress_frame(
                report_path,
                progress_name,
                frame,
                output_path=output_path,
                progress_count=progress_count,
            )
        )
    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Plot *_progress entries from a JSON training report.")
    parser.add_argument("report", help="Path to the JSON report file")
    parser.add_argument("-o", "--output", help="Optional output image path or directory")
    args = parser.parse_args()

    output_paths = plot_report(args.report, args.output)
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()