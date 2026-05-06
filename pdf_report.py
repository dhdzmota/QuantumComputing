"""
PDF report generator for the Quantum Circuit Simulation.

Produces an A4 document containing:
  - System parameters
  - Circuit gate table
  - Initial → final state probability table (ΔP highlighted green/red)
  - Light-themed probability-vs-time chart
"""

import io
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from fpdf import FPDF

_STATE_PALETTE = [
    '#6366f1', '#22d3ee', '#f59e0b', '#10b981',
    '#f43f5e', '#a855f7', '#3b82f6', '#84cc16',
    '#fb923c', '#14b8a6', '#e11d48', '#7c3aed',
    '#0284c7', '#d97706', '#15803d', '#be185d',
    '#818cf8', '#67e8f9', '#fcd34d', '#6ee7b7',
    '#fb7185', '#c084fc', '#93c5fd', '#bef264',
    '#fdba74', '#5eead4', '#fda4af', '#a78bfa',
    '#7dd3fc', '#fde68a', '#86efac', '#f9a8d4',
]

# Colours (R, G, B tuples)
C_DARK   = (26,  37,  64)    # header bg / title text
C_ACCENT = (79, 142, 247)    # accent blue
C_MUTED  = (100, 116, 139)   # muted text
C_ALT    = (241, 245, 249)   # alternating row bg
C_GRID   = (203, 213, 225)   # table border
C_GREEN  = (21, 128, 61)
C_RED    = (185, 28, 28)
C_WHITE  = (255, 255, 255)
C_BODY   = (55,  65,  81)


def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _light_prob_figure(T, D, states):
    """Rebuild the chart with a white background for printing."""
    T_arr = np.array(T)
    D_arr = np.array(D)
    norm  = D_arr.sum(axis=0)

    traces = [go.Scatter(
        x=T_arr, y=norm, name='‖ψ‖²',
        line=dict(color='rgba(0,0,0,0.22)', width=1.5, dash='dot'),
    )]
    for i, (d, lbl) in enumerate(zip(D, states)):
        traces.append(go.Scatter(
            x=T_arr, y=d, name=lbl,
            line=dict(color=_STATE_PALETTE[i % len(_STATE_PALETTE)], width=2),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor='white', plot_bgcolor='#f4f6fb',
        font=dict(color='#1a2540', size=11),
        xaxis=dict(title='Time (τ)', color='#374151',
                   gridcolor='rgba(0,0,0,0.07)', zeroline=False, linecolor='#ccc'),
        yaxis=dict(title='Probability', range=[-0.02, 1.08], color='#374151',
                   gridcolor='rgba(0,0,0,0.07)', zeroline=False, linecolor='#ccc'),
        legend=dict(bgcolor='white', bordercolor='#dde1ea', borderwidth=1,
                    font=dict(size=10)),
        margin=dict(l=55, r=20, t=20, b=55),
        width=720, height=340,
    )
    return fig


_DEJAVU      = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
_DEJAVU_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
_DEJAVU_MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'


class _PDF(FPDF):
    """Thin FPDF subclass with helpers for section headers and tables."""

    PAGE_W    = 210   # A4 mm
    MARGIN    = 18
    CONTENT_W = PAGE_W - 2 * MARGIN

    def __init__(self, timestamp):
        super().__init__(unit='mm', format='A4')
        self.timestamp = timestamp
        # Register DejaVu Sans for full Unicode support (arrows, ket brackets, π …)
        self.add_font('DV',  style='',  fname=_DEJAVU)
        self.add_font('DV',  style='B', fname=_DEJAVU_BOLD)
        self.add_font('DVMono', style='', fname=_DEJAVU_MONO)
        self.set_margins(self.MARGIN, self.MARGIN, self.MARGIN)
        self.set_auto_page_break(True, margin=self.MARGIN)
        self.add_page()

    # ── Structural helpers ─────────────────────────────────────────────

    def title_block(self, title, sub):
        self.set_font('DV', 'B', 16)
        self.set_text_color(*C_DARK)
        self.cell(0, 9, title, ln=True)
        self.set_font('DV', '', 8)
        self.set_text_color(*C_MUTED)
        self.cell(0, 5, sub, ln=True)
        self.ln(2)
        self.set_draw_color(*C_DARK)
        self.set_line_width(0.5)
        self.line(self.MARGIN, self.get_y(), self.PAGE_W - self.MARGIN, self.get_y())
        self.ln(4)

    def section_heading(self, text):
        self.ln(3)
        self.set_font('DV', 'B', 10)
        self.set_text_color(*C_DARK)
        self.cell(0, 6, text, ln=True)
        self.ln(1)

    def hr(self, color=C_GRID):
        self.set_draw_color(*color)
        self.set_line_width(0.3)
        self.line(self.MARGIN, self.get_y(), self.PAGE_W - self.MARGIN, self.get_y())
        self.ln(3)

    # ── Table helpers ──────────────────────────────────────────────────

    def table_header(self, cols, widths):
        self.set_fill_color(*C_DARK)
        self.set_text_color(*C_WHITE)
        self.set_font('DV', 'B', 8.5)
        self.set_draw_color(*C_GRID)
        self.set_line_width(0.25)
        for col, w in zip(cols, widths):
            self.cell(w, 7, col, border=1, fill=True)
        self.ln()

    def table_row(self, cells, widths, alt=False, col_colors=None, mono_cols=None):
        self.set_fill_color(*(C_ALT if alt else C_WHITE))
        self.set_text_color(*C_BODY)
        self.set_font('DV', '', 8.5)
        self.set_draw_color(*C_GRID)
        self.set_line_width(0.25)
        for i, (cell, w) in enumerate(zip(cells, widths)):
            bold  = col_colors and col_colors[i]
            mono  = mono_cols  and mono_cols[i]
            fname = 'DVMono' if mono else 'DV'
            style = 'B' if bold else ''
            self.set_font(fname, style, 8.5)
            if bold:
                self.set_text_color(*col_colors[i])
            self.cell(w, 6.5, str(cell), border=1, fill=True)
            self.set_text_color(*C_BODY)
        self.ln()


def generate_pdf(circuit_data, sim_data):
    """
    Build a PDF report and return the raw bytes.

    Parameters
    ----------
    circuit_data : dict  {'n_qubits': int, 'gates': [...]}
    sim_data     : dict  run_simulation output (T, D, states, n_qubits, E)
    """
    gates   = (circuit_data or {}).get('gates', [])
    n_q     = int(sim_data.get('n_qubits', 3))
    states  = sim_data['states']
    T       = sim_data['T']
    D       = sim_data['D']
    ts      = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')

    pdf = _PDF(ts)
    W   = pdf.CONTENT_W

    # ── Header ────────────────────────────────────────────────────────
    pdf.title_block(
        'Quantum Circuit Simulation Report',
        f'Generated: {ts}',
    )

    # ── System parameters ─────────────────────────────────────────────
    pdf.section_heading('System Parameters')
    col_w = [W * 0.40, W * 0.60]
    pdf.table_header(['Parameter', 'Value'], col_w)
    rows = [
        ('Qubits (N)',       str(n_q)),
        ('Basis states',     str(2 ** n_q)),
        ('Simulation steps', str(len(T))),
        ('Time span',        f'0 → {T[-1]:.4f} τ'),
        ('Circuit gates',    str(len(gates))),
    ]
    for i, (k, v) in enumerate(rows):
        pdf.table_row([k, v], col_w, alt=(i % 2 == 1))

    # ── Circuit definition ────────────────────────────────────────────
    pdf.section_heading('Circuit Definition')
    type_map = {'pi_half': 'π/2  (superposition)', 'pi': 'π  (inversion)'}
    col_w2 = [W * 0.08, W * 0.37, W * 0.55]
    pdf.table_header(['#', 'Gate Type', 'Transition'], col_w2)
    if gates:
        for i, g in enumerate(gates):
            tr = f"{g['label_k']} → {g['label_j']}"
            pdf.table_row(
                [str(i + 1), type_map.get(g['type'], g['type']), tr],
                col_w2, alt=(i % 2 == 1),
                mono_cols=[False, False, True],
            )
    else:
        pdf.set_font('DV', '', 8.5)
        pdf.set_text_color(*C_MUTED)
        pdf.cell(0, 6, 'No gates defined.', ln=True)

    # ── State evolution ───────────────────────────────────────────────
    pdf.section_heading('State Evolution  —  Initial → Final Probabilities')
    col_w3 = [W * 0.22, W * 0.22, W * 0.22, W * 0.34]
    pdf.table_header(['State', 'P(t=0)', 'P(t=T)', 'ΔP'], col_w3)
    for i, lbl in enumerate(states):
        p0  = D[i][0]
        pf  = D[i][-1]
        dp  = pf - p0
        dp_s = ('+' if dp >= 0 else '') + f'{dp:.5f}'
        col_c = [None, None, None,
                 C_GREEN if dp > 1e-4 else (C_RED if dp < -1e-4 else None)]
        pdf.table_row(
            [lbl, f'{p0:.5f}', f'{pf:.5f}', dp_s],
            col_w3, alt=(i % 2 == 1), col_colors=col_c,
            mono_cols=[True, False, False, False],
        )

    # ── Probability chart ─────────────────────────────────────────────
    pdf.ln(3)
    pdf.hr()
    pdf.section_heading('Probability vs. Time')

    fig     = _light_prob_figure(T, D, states)
    img_png = pio.to_image(fig, format='png', scale=2)
    tmp_buf = io.BytesIO(img_png)

    img_w = W
    img_h = img_w * 340 / 720
    pdf.image(tmp_buf, x=pdf.MARGIN, y=None, w=img_w, h=img_h)

    return bytes(pdf.output())
