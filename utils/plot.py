import numbers
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pynvml
import torch
from matplotlib.ticker import FuncFormatter


def peak_bandwidth(clock=None):
    """GB/s. clock is the memory clock in MHz, read from the device if not given."""
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    clock = p.memory_clock_rate * 1e-3 if clock is None else clock
    return p.memory_bus_width / 8 * clock * 1e6 * 2 / 1e9


def peak_flops(clock=None, flops_per_clk_sm=None):
    """TFLOP/s. clock is the SM clock in MHz, max SM clock via pynvml if not given."""
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    clock = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM) if clock is None else clock
    if flops_per_clk_sm is None:
        # dense bf16/fp16 tensor-core FLOPs per clock per SM, fp32 accumulate.
        # GeForce sm_86 / sm_89 run it at half rate
        geforce = pynvml.nvmlDeviceGetBrand(h) == pynvml.NVML_BRAND_GEFORCE
        table = {(8, 0): 2048, (8, 6): 512 if geforce else 1024,
                 (8, 9): 512 if geforce else 1024, (9, 0): 4096}
        flops_per_clk_sm = table.get((p.major, p.minor))
        if flops_per_clk_sm is None:
            raise ValueError(f"no FLOPs/clk/SM for sm_{p.major}{p.minor}, pass flops_per_clk_sm")
    return p.multi_processor_count * flops_per_clk_sm * clock * 1e6 / 1e12


def plot_bench(dims, times, title, *,
            nbytes=None, flops=None,
            peak=None,
            path='plots',
            reference=None, xlabel, subtitle=None, name=None):
    # times and reference are in ms. times: a list (one per dim), or {label: list}.
    # reference: {label: list or one time}; one time is used at every dim, like a
    # single nbytes or flops. constant throughput draws a flat line, otherwise a curve.
    # a relative path is relative to the cwd; for a folder next to the calling
    # script, pass path=Path(__file__).parent / 'plots'
    stem = name or re.sub(r'[^\w.-]+', '_', title.lower()).strip('_')
    path = Path(path) / f'{stem}.png'

    per_dim = lambda v: ([v(d) for d in dims] if callable(v)
                         else list(v) if hasattr(v, '__len__') else [v] * len(dims))

    if (nbytes is None) == (flops is None):
        raise ValueError("pass exactly one of nbytes (memory bound) or flops (compute bound)")

    if nbytes is not None:
        peak = peak_bandwidth() if peak is None else peak
        work, scale, unit = per_dim(nbytes), 1e9, 'GB/s'
    else:
        peak = peak_flops() if peak is None else peak
        work, scale, unit = per_dim(flops), 1e12, 'TFLOP/s'

    def throughput(label, ts):
        if len(ts) != len(dims):
            raise ValueError(f"{label!r} has {len(ts)} times, dims has {len(dims)}")
        return [w / (t * 1e-3) / scale for w, t in zip(work, ts)]

    series = times if isinstance(times, dict) else {None: times}
    ys = {label: throughput(label, ts) for label, ts in series.items()}

    lines = [(peak, (0, (1, 2.5)), f'theoretical peak  {peak:,.0f} {unit}')]
    curves = []

    for ref_name, t in (reference or {}).items():
        ry = throughput(ref_name, [t] * len(dims) if isinstance(t, numbers.Real) else t)
        if len(set(ry)) == 1:
            lines.append((ry[0], (0, (4, 3)), f'{ref_name}  {ry[0]:,.0f} {unit}'))
        else:
            curves.append((ry, ref_name))

    curve_styles = [(0, (4, 3)), (0, (6, 2, 1, 2)), (0, (1, 1.5))]
    if len(curves) > len(curve_styles):
        raise ValueError(f"at most {len(curve_styles)} reference curves per plot, got {len(curves)}")

    c = dict(bg='#1d2021', fg='#ebdbb2', muted='#928374', grid='#32302f',
            spine='#504945', ref='#665c54')
    palettes = {
        1: ['#788d7a'],
        2: ['#788d7a', '#af7379'],
        3: ['#788d7a', '#988457', '#af7379'],
        4: ['#788d7a', '#898960', '#a57d58', '#af7379'],
        5: ['#788d7a', '#838b66', '#988457', '#aa7a5d', '#af7379'],
    }
    if len(ys) not in palettes:
        raise ValueError(f"at most {max(palettes)} kernels per plot, got {len(ys)}")
    palette = palettes[len(ys)]
    s = dict(base=10.5, title=17, subtitle=10.5, axis_label=10.5, tick=10, ref=9.5)
    rc = {
        'font.family': 'serif',
        'font.serif': ['Lora', 'TeX Gyre Pagella', 'DejaVu Serif'],
        'mathtext.fontset': 'cm', 'font.size': s['base'],
        'figure.facecolor': c['bg'], 'axes.facecolor': c['bg'],
        'savefig.facecolor': c['bg'],
        'text.color': c['fg'], 'axes.labelcolor': c['muted'],
        'xtick.color': c['muted'], 'ytick.color': c['muted'],
        'axes.edgecolor': c['spine'], 'axes.linewidth': 0.8,
        'xtick.major.size': 0, 'ytick.major.size': 0,
        'xtick.major.pad': 7, 'ytick.major.pad': 7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.spines.left': False,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(7.6, 4.7))
        fig.subplots_adjust(left=0.115, right=0.885, top=0.78, bottom=0.13)

        fig.text(0.115, 0.925, title, fontsize=s['title'], color=c['fg'])
        if subtitle:
            fig.text(0.115, 0.865, subtitle, fontsize=s['subtitle'],
                     color=c['muted'], style='italic')

        ax.set_axisbelow(True)
        ax.grid(axis='y', color=c['grid'], lw=0.7)

        for yv, ls, txt in lines:
            ax.axhline(yv, ls=ls, lw=0.9, color=c['ref'], zorder=1)
            ax.annotate(txt, xy=(0.015, yv), xycoords=ax.get_yaxis_transform(),
                        xytext=(0, 3), textcoords='offset points',
                        fontsize=s['ref'], color=c['muted'], style='italic',
                        ha='left', va='bottom', zorder=4)

        for (ry, txt), ls in zip(curves, curve_styles):
            ax.plot(dims, ry, ls=ls, lw=1.1, color=c['ref'], zorder=2)
            ax.annotate(txt, xy=(dims[-1], ry[-1]), xytext=(0, 5),
                        textcoords='offset points', fontsize=s['ref'],
                        color=c['muted'], style='italic',
                        ha='right', va='bottom', zorder=4)

        for (label, y), col in zip(ys.items(), palette):
            ax.plot(dims, y, '-o', lw=2.2, color=col, ms=4.5, mfc=c['bg'],
                    mec=col, mew=1.5, zorder=3, solid_capstyle='round', label=label)

        if len(ys) > 1:
            ax.legend(loc='lower right', frameon=False, fontsize=s['ref'],
                      labelcolor=c['muted'], handlelength=1.6, borderaxespad=0.6)

        ax.set_xscale('log', base=2)
        ax.minorticks_off()
        ax.set_xticks(dims)
        ax.set_xticklabels([f'{d // 1024}K' if d >= 1024 else str(d) for d in dims],
                           fontsize=s['tick'])
        ax.tick_params(axis='y', labelsize=s['tick'])
        pcts = [0, 25, 50, 75, 100]
        ax.set_yticks([q / 100 * peak for q in pcts])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
        ax.set_xlim(dims[0] / 1.35, dims[-1] * 1.35)
        ax.set_ylim(0, max([peak, *(v for y in ys.values() for v in y), *(v for ry, _ in curves for v in ry)]) * 1.06)
        ax.set_xlabel(xlabel, labelpad=8, fontsize=s['axis_label'], style='italic')
        ax.set_ylabel(unit, labelpad=10, fontsize=s['axis_label'], style='italic')

        sec = ax.secondary_yaxis(
            'right', functions=(lambda g: g / peak * 100, lambda q: q / 100 * peak))
        sec.set_yticks(pcts)
        sec.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}%'))
        sec.spines['right'].set_visible(False)
        sec.tick_params(length=0, pad=7, colors=c['muted'], labelsize=s['tick'])
        sec.set_ylabel('of theoretical peak', labelpad=10, fontsize=s['axis_label'],
                       style='italic', color=c['muted'])

        ax.get_ygridlines()[-1].set_visible(False)

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)