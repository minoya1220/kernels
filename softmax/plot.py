from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import FuncFormatter

from measure import benchmark

# assumes only one output tensor
def plot_bench(dims, times, n_read, elems, title, dtype=torch.bfloat16, *,
            reference=False, memory_clock=None,
            color='#83a598', sizes=None, font='Lora', name=None):
    c = dict(bg='#1d2021', fg='#ebdbb2', muted='#928374', grid='#32302f',
            spine='#504945', band='#282828', line=color)
    s = {**dict(base=11.5, title=13.5, axis_label=11.5, tick=9.5, ref=9.5),
        **(sizes or {})}
    path = Path(__file__).parent / "plots" / f"{name or title.lower().replace(' ', '_')}.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    p = torch.cuda.get_device_properties(0)
    memory_clock = p.memory_clock_rate * 1e-3 if memory_clock is None else memory_clock
    peak = p.memory_bus_width / 8 * memory_clock * 1e6 * 2 / 1e9

    esize = torch.finfo(dtype).bits // 8
    total_bytes = (n_read + 1) * elems * esize
    gb = [total_bytes / (t * 1e-3) / 1e9 for t in times]

    lines = [(peak, ':', f'theoretical peak · {peak:.0f} GB/s')]

    if reference:
        ops = {1: (torch.abs, 'torch abs'), 2: (torch.add, 'torch add')}
        if n_read not in ops:
            raise ValueError(f"no reference op for n_read={n_read}, have {sorted(ops)}")
        fn, ref_name = ops[n_read]

        ins = [torch.randn(elems, device='cuda', dtype=dtype) for _ in range(n_read)]
        out = torch.empty_like(ins[0])
        ref = total_bytes / (benchmark(fn, ins, [out]) * 1e-3) / 1e9

        lines.append((ref, '--', f'{ref_name} ({n_read}r+1w) · {ref:.0f} GB/s'))

    rc = {
        'font.family': 'serif',
        'font.serif': [font, 'TeX Gyre Pagella', 'DejaVu Serif'],
        'mathtext.fontset': 'cm', 'font.size': s['base'],
        'figure.facecolor': c['bg'], 'axes.facecolor': c['bg'],
        'savefig.facecolor': c['bg'],
        'text.color': c['fg'], 'axes.labelcolor': c['fg'],
        'xtick.color': c['muted'], 'ytick.color': c['muted'],
        'axes.edgecolor': c['spine'], 'axes.linewidth': 0.8,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'axes.spines.top': False, 'axes.spines.right': False,
    }

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(7.6, 4.8),
                            constrained_layout=dict(w_pad=0.18, h_pad=0.16))
        top, xr = peak * 1.10, dims[-1] * 1.42

        ax.set_axisbelow(True)
        ax.grid(axis='y', color=c['grid'], lw=0.7)
        ax.axhspan(peak, top, color=c['band'], lw=0, zorder=0)

        for y, ls, txt in lines:
            ax.axhline(y, ls=ls, lw=1.0, color=c['muted'], zorder=2)
            ax.annotate(txt, xy=(0.015, y), xycoords=ax.get_yaxis_transform(),
                        xytext=(0, 3), textcoords='offset points',
                        fontsize=s['ref'], color=c['muted'],
                        ha='left', va='bottom', zorder=4)

        ax.fill_between(dims, gb, color=c['line'], alpha=0.10, zorder=1)
        ax.plot(dims, gb, 'o-', ms=5, lw=2, color=c['line'], zorder=3)

        ax.set_xscale('log', base=2)
        ax.set_xticks(dims)
        ax.set_xticklabels([str(d) for d in dims], rotation=45, ha='right',
                        rotation_mode='anchor', fontsize=s['tick'])
        ax.tick_params(axis='y', labelsize=s['tick'])
        ax.set_xlim(dims[0] / 1.55, xr)
        ax.set_ylim(0, top)
        ax.set_xlabel('row length $N$', labelpad=8, fontsize=s['axis_label'])
        ax.set_ylabel('achieved bandwidth (GB/s)', labelpad=8, fontsize=s['axis_label'])
        ax.set_title(title, fontsize=s['title'], loc='left', color=c['line'], pad=12)

        sec = ax.secondary_yaxis(
            'right', functions=(lambda g: g / peak * 100, lambda q: q / 100 * peak))
        sec.set_ylabel('% of theoretical peak', labelpad=10, fontsize=s['axis_label'])
        sec.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}%'))
        sec.spines['right'].set_visible(True)
        sec.spines['right'].set_color(c['spine'])
        sec.tick_params(colors=c['muted'], labelsize=s['tick'])
        sec.yaxis.label.set_color(c['fg'])

        fig.savefig(path, dpi=150)
        plt.close(fig)


