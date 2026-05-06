import sys
import os
from datetime import datetime
import numpy as np
from dash import Dash, dcc, html, Input, Output, State, ctx, ALL
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'quantum_computer_src'))
from qc_simulation import run_simulation, compute_bloch_vectors
from qc_circuit import get_allowed_transitions, compute_energies
from qc_utils import get_all_states
from pdf_report import generate_pdf

# ── Theme ──────────────────────────────────────────────────────────────────
BG      = '#07090f'
PANEL   = '#0d1117'
CARD    = '#161b27'
BORDER  = '#1e2d45'
TEXT    = '#e2e8f0'
MUTED   = '#64748b'
ACCENT  = '#4f8ef7'
SUCCESS = '#22c55e'
DANGER  = '#f43f5e'

STATE_PALETTE = [
    '#6366f1','#22d3ee','#f59e0b','#10b981',
    '#f43f5e','#a855f7','#3b82f6','#84cc16',
    '#fb923c','#14b8a6','#e11d48','#7c3aed',
    '#0284c7','#d97706','#15803d','#be185d',
    '#818cf8','#67e8f9','#fcd34d','#6ee7b7',
    '#fb7185','#c084fc','#93c5fd','#bef264',
    '#fdba74','#5eead4','#fda4af','#a78bfa',
    '#7dd3fc','#fde68a','#86efac','#f9a8d4',
]

QUBIT_COLORS = ['#4f8ef7', '#22d3ee', '#10b981', '#f59e0b', '#f43f5e']

ALGO_LABELS = {
    2: 'Entanglement Demo',
    3: 'Quantum Teleportation',
    4: 'Multi-Qubit Cascade',
    5: 'Multi-Qubit Cascade',
}

# ── Figure helpers ─────────────────────────────────────────────────────────

def _sphere_wireframe():
    """Return Scatter3d traces for equator + two meridians + axes."""
    traces = []
    theta = np.linspace(0, 2 * np.pi, 120)

    # Graticule circles
    for (xi, yi, zi) in [
        (np.cos(theta),  np.sin(theta),  np.zeros(120)),      # equator
        (np.cos(theta),  np.zeros(120),  np.sin(theta)),      # xz
        (np.zeros(120),  np.cos(theta),  np.sin(theta)),      # yz
    ]:
        traces.append(go.Scatter3d(
            x=xi, y=yi, z=zi,
            mode='lines',
            line=dict(color='rgba(100,140,220,0.30)', width=1),
            hoverinfo='skip', showlegend=False, name='',
        ))

    # Axes
    for ax in [[1,0,0],[0,1,0],[0,0,1]]:
        traces.append(go.Scatter3d(
            x=[-1.25*ax[0], 1.25*ax[0]],
            y=[-1.25*ax[1], 1.25*ax[1]],
            z=[-1.25*ax[2], 1.25*ax[2]],
            mode='lines', line=dict(color='rgba(200,210,255,0.35)', width=1.5),
            hoverinfo='skip', showlegend=False, name='',
        ))

    # Pole / axis labels
    labels = [
        (0, 0,  1.5, '|0⟩'),
        (0, 0, -1.5, '|1⟩'),
        (1.5, 0,  0,  'x'),
        (0, 1.5,  0,  'y'),
    ]
    for lx, ly, lz, txt in labels:
        traces.append(go.Scatter3d(
            x=[lx], y=[ly], z=[lz],
            mode='text', text=[txt],
            textfont=dict(color='rgba(180,200,255,0.85)', size=13),
            hoverinfo='skip', showlegend=False, name='',
        ))
    return traces


def bloch_sphere_figure(bx, by, bz, title, color):
    """Bloch sphere with the given Bloch vector drawn as an arrow."""
    n = 36
    u  = np.linspace(0, 2 * np.pi, n)
    v  = np.linspace(0, np.pi, n)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(n), np.cos(v))

    traces: list = [
        go.Surface(
            x=xs, y=ys, z=zs,
            opacity=0.07,
            colorscale=[[0, '#040b18'], [0.5, '#0c1f40'], [1, '#1a3566']],
            surfacecolor=zs,
            showscale=False, hoverinfo='skip', name='',
        )
    ]
    traces += _sphere_wireframe()

    mag = float(np.sqrt(bx**2 + by**2 + bz**2))

    # Projection shadow on xy-plane
    if mag > 0.02:
        traces.append(go.Scatter3d(
            x=[0, bx], y=[0, by], z=[0, 0],
            mode='lines', line=dict(color='rgba(255,255,255,0.12)', width=1.5, dash='dot'),
            hoverinfo='skip', showlegend=False, name='',
        ))
        traces.append(go.Scatter3d(
            x=[bx, bx], y=[by, by], z=[0, bz],
            mode='lines', line=dict(color='rgba(255,255,255,0.12)', width=1.5, dash='dot'),
            hoverinfo='skip', showlegend=False, name='',
        ))

        # Bloch vector stem (80 % of length so arrowhead has room)
        traces.append(go.Scatter3d(
            x=[0, bx * 0.78], y=[0, by * 0.78], z=[0, bz * 0.78],
            mode='lines', line=dict(color=color, width=6),
            hoverinfo='skip', showlegend=False, name='',
        ))

        # Arrowhead cone
        traces.append(go.Cone(
            x=[bx * 0.78], y=[by * 0.78], z=[bz * 0.78],
            u=[bx * 0.22], v=[by * 0.22], w=[bz * 0.22],
            sizemode='absolute', sizeref=0.22,
            colorscale=[[0, color], [1, color]],
            showscale=False, hoverinfo='skip', name='',
            anchor='tail',
        ))

    # Endpoint dot
    traces.append(go.Scatter3d(
        x=[bx], y=[by], z=[bz],
        mode='markers',
        marker=dict(size=5, color=color, opacity=0.95,
                    line=dict(color='white', width=1)),
        hoverinfo='skip', showlegend=False, name='',
    ))

    # Origin dot
    traces.append(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=3, color='rgba(255,255,255,0.6)'),
        hoverinfo='skip', showlegend=False, name='',
    ))

    bx_s = f'{bx:+.3f}'
    by_s = f'{by:+.3f}'
    bz_s = f'{bz:+.3f}'
    subtitle = f'⟨σx⟩={bx_s}  ⟨σy⟩={by_s}  ⟨σz⟩={bz_s}'

    layout = go.Layout(
        title=dict(
            text=f'<b>{title}</b><br><sup style="color:#94a3b8">{subtitle}</sup>',
            font=dict(color=TEXT, size=13),
            x=0.5, y=0.97, xanchor='center',
        ),
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False,
                       zeroline=False, title='',
                       showbackground=False, range=[-1.6, 1.6]),
            yaxis=dict(showgrid=False, showticklabels=False,
                       zeroline=False, title='',
                       showbackground=False, range=[-1.6, 1.6]),
            zaxis=dict(showgrid=False, showticklabels=False,
                       zeroline=False, title='',
                       showbackground=False, range=[-1.6, 1.6]),
            bgcolor='rgba(0,0,0,0)',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=70, b=0),
        height=340,
    )
    return go.Figure(data=traces, layout=layout)


def empty_bloch_figure(title, color):
    traces = [go.Surface(
        x=np.outer(np.cos(np.linspace(0, 2*np.pi, 36)),
                   np.sin(np.linspace(0, np.pi, 36))),
        y=np.outer(np.sin(np.linspace(0, 2*np.pi, 36)),
                   np.sin(np.linspace(0, np.pi, 36))),
        z=np.outer(np.ones(36), np.cos(np.linspace(0, np.pi, 36))),
        opacity=0.07,
        colorscale=[[0, '#040b18'], [1, '#1a3566']],
        showscale=False, hoverinfo='skip', name='',
    )]
    traces += _sphere_wireframe()
    return go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(text=f'<b>{title}</b><br><sup style="color:#94a3b8">awaiting simulation</sup>',
                       font=dict(color=TEXT, size=13), x=0.5, y=0.97, xanchor='center'),
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                           showbackground=False, range=[-1.6, 1.6]),
                yaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                           showbackground=False, range=[-1.6, 1.6]),
                zaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                           showbackground=False, range=[-1.6, 1.6]),
                bgcolor='rgba(0,0,0,0)', aspectmode='cube',
                camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=70, b=0),
            height=340,
        ),
    )


def probability_figure(T, D, states):
    T_arr = np.array(T)
    D_arr = np.array(D)
    N_arr = D_arr.sum(axis=0)

    traces = [go.Scatter(
        x=T_arr, y=N_arr,
        name='‖ψ‖² (norm)',
        line=dict(color='rgba(255,255,255,0.35)', width=1.5, dash='dot'),
        hovertemplate='t=%{x:.3f}<br>norm=%{y:.5f}<extra></extra>',
    )]
    for idx, (d, label) in enumerate(zip(D, states)):
        traces.append(go.Scatter(
            x=T_arr, y=d,
            name=label,
            line=dict(color=STATE_PALETTE[idx % len(STATE_PALETTE)], width=2.2),
            hovertemplate=f'{label}  P=%{{y:.4f}}  t=%{{x:.3f}}<extra></extra>',
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,18,30,0.9)',
        xaxis=dict(
            title=dict(text='Time (τ)', font=dict(color=MUTED)),
            color=MUTED, gridcolor='rgba(60,90,150,0.12)',
            zeroline=False, tickfont=dict(size=11),
        ),
        yaxis=dict(
            title=dict(text='Probability', font=dict(color=MUTED)),
            range=[-0.02, 1.08],
            color=MUTED, gridcolor='rgba(60,90,150,0.12)',
            zeroline=False, tickfont=dict(size=11),
        ),
        legend=dict(
            bgcolor='rgba(10,15,28,0.85)',
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(color='#94a3b8', size=12),
        ),
        hovermode='x unified',
        margin=dict(l=55, r=20, t=20, b=55),
        height=480,
    )
    return fig


def empty_prob_figure():
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,18,30,0.9)',
        xaxis=dict(color=MUTED, gridcolor='rgba(60,90,150,0.12)', zeroline=False,
                   title=dict(text='Time (τ)', font=dict(color=MUTED))),
        yaxis=dict(color=MUTED, gridcolor='rgba(60,90,150,0.12)', zeroline=False,
                   range=[0, 1.05],
                   title=dict(text='Probability', font=dict(color=MUTED))),
        annotations=[dict(
            text='Run the simulation to see results',
            x=0.5, y=0.5, xref='paper', yref='paper',
            showarrow=False, font=dict(size=16, color=MUTED),
        )],
        margin=dict(l=55, r=20, t=20, b=55),
        height=480,
    )
    return fig


# ── Shared styles ──────────────────────────────────────────────────────────

def _label(text):
    return html.Div(text, style={
        'color': MUTED, 'fontSize': '11px', 'fontWeight': '600',
        'textTransform': 'uppercase', 'letterSpacing': '0.08em',
        'marginBottom': '5px', 'marginTop': '14px',
    })


def _input(id_, value, placeholder='', type_='number', min_=None, step=None):
    kwargs = dict(
        id=id_, value=value, placeholder=placeholder, type=type_,
        style={
            'width': '100%', 'boxSizing': 'border-box',
            'background': '#0d1b2e', 'border': f'1px solid {BORDER}',
            'borderRadius': '6px', 'color': TEXT, 'padding': '7px 10px',
            'fontSize': '13px', 'outline': 'none',
        },
        debounce=True,
    )
    if min_ is not None:
        kwargs['min'] = min_
    if step is not None:
        kwargs['step'] = step
    return dcc.Input(**kwargs)


# ── App layout ──────────────────────────────────────────────────────────────

app = Dash(__name__, title='Quantum Computer Simulator')
app.layout = html.Div(style={'background': BG, 'minHeight': '100vh',
                              'fontFamily': "'Inter','Segoe UI',sans-serif",
                              'color': TEXT, 'display': 'flex',
                              'flexDirection': 'column'}, children=[

    # ── Header ────────────────────────────────────────────────────────────
    html.Div(style={
        'background': PANEL, 'borderBottom': f'1px solid {BORDER}',
        'padding': '14px 28px', 'display': 'flex',
        'alignItems': 'center', 'gap': '16px',
    }, children=[
        html.Div('⬡', style={'fontSize': '24px', 'color': ACCENT}),
        html.Div(children=[
            html.H1('Quantum Computer Simulator',
                    style={'margin': 0, 'fontSize': '18px', 'fontWeight': '700',
                           'color': TEXT}),
            html.Div('N-Qubit Spin-Chain Simulation · Runge-Kutta 4th Order',
                     style={'fontSize': '12px', 'color': MUTED}),
        ]),
    ]),

    # ── Body ──────────────────────────────────────────────────────────────
    html.Div(style={'display': 'flex', 'flex': '1', 'overflow': 'hidden'}, children=[

        # ── Sidebar ───────────────────────────────────────────────────────
        html.Div(style={
            'width': '270px', 'minWidth': '270px',
            'background': PANEL, 'borderRight': f'1px solid {BORDER}',
            'padding': '20px 18px', 'overflowY': 'auto',
            'display': 'flex', 'flexDirection': 'column', 'gap': '2px',
        }, children=[

            html.Div('Configuration',
                     style={'fontSize': '13px', 'fontWeight': '700',
                            'color': TEXT, 'marginBottom': '4px',
                            'paddingBottom': '10px',
                            'borderBottom': f'1px solid {BORDER}'}),

            _label('Number of Qubits'),
            dcc.Dropdown(
                id='n-qubits-dd',
                options=[{'label': f'{n} qubits', 'value': n} for n in range(2, 6)],
                value=3,
                clearable=False,
                style={'background': '#0d1b2e', 'color': TEXT,
                       'border': f'1px solid {BORDER}', 'borderRadius': '6px',
                       'fontSize': '13px'},
            ),

            html.Div(id='algo-label', style={
                'fontSize': '11px', 'color': ACCENT,
                'marginTop': '5px', 'textAlign': 'center',
            }),

            html.Div(style={'height': '1px', 'background': BORDER,
                            'margin': '12px 0'}),
            html.Div('Physical Parameters',
                     style={'fontSize': '12px', 'fontWeight': '600',
                            'color': MUTED, 'marginBottom': '2px'}),

            _label('ζ  (qubit frequency spacing)'),
            _input('zeta-in', 100, min_=1, step=10),

            _label('ζ₀  (base frequency)'),
            _input('zeta0-in', 100, min_=1, step=10),

            _label('J  (Ising coupling)'),
            _input('j-in', 10, min_=0, step=1),

            html.Div(style={'height': '1px', 'background': BORDER,
                            'margin': '12px 0'}),
            html.Div('Simulation Settings',
                     style={'fontSize': '12px', 'fontWeight': '600',
                            'color': MUTED, 'marginBottom': '2px'}),

            _label('Step size'),
            dcc.Dropdown(
                id='step-dd',
                options=[
                    {'label': 'Coarse  Δt = 0.05  (fast)',    'value': 0.05},
                    {'label': 'Normal  Δt = 0.01',             'value': 0.01},
                    {'label': 'Fine    Δt = 0.005  (slow)',    'value': 0.005},
                ],
                value=0.01,
                clearable=False,
                style={'background': '#0d1b2e', 'color': TEXT,
                       'border': f'1px solid {BORDER}', 'borderRadius': '6px',
                       'fontSize': '12px'},
            ),

            _label('Initial state'),
            dcc.Dropdown(
                id='init-dd',
                options=[
                    {'label': 'Default (algorithm preset)', 'value': 'default'},
                    {'label': 'Ground state  |00…0⟩',       'value': 'ground'},
                    {'label': 'Equal superposition',         'value': 'equal'},
                    {'label': 'Custom amplitudes',           'value': 'custom'},
                ],
                value='default',
                clearable=False,
                style={'background': '#0d1b2e', 'color': TEXT,
                       'border': f'1px solid {BORDER}', 'borderRadius': '6px',
                       'fontSize': '12px'},
            ),

            html.Div(id='custom-amps-panel'),

            html.Div(style={'height': '1px', 'background': BORDER,
                            'margin': '16px 0'}),

            # Run button
            dcc.Loading(
                id='run-loading',
                type='circle',
                color=ACCENT,
                children=html.Button(
                    '▶  Run Simulation',
                    id='run-btn',
                    n_clicks=0,
                    style={
                        'width': '100%', 'padding': '11px',
                        'background': ACCENT, 'color': 'white',
                        'border': 'none', 'borderRadius': '8px',
                        'fontSize': '14px', 'fontWeight': '700',
                        'cursor': 'pointer', 'letterSpacing': '0.03em',
                    },
                ),
            ),

            html.Div(id='status-msg', style={
                'fontSize': '11px', 'color': MUTED,
                'marginTop': '8px', 'textAlign': 'center',
                'minHeight': '16px',
            }),
        ]),  # end sidebar

        # ── Main content ──────────────────────────────────────────────────
        html.Div(style={'flex': '1', 'padding': '20px 24px',
                        'overflowY': 'auto'}, children=[

            dcc.Tabs(
                id='main-tabs',
                value='prob',
                style={'marginBottom': '16px'},
                colors={'border': BORDER, 'primary': ACCENT,
                        'background': PANEL},
                children=[

                    # ── Tab 1: Probabilities ──────────────────────────────
                    dcc.Tab(
                        label='State Probabilities',
                        value='prob',
                        style={'color': MUTED, 'background': PANEL,
                               'border': f'1px solid {BORDER}',
                               'padding': '8px 16px', 'fontSize': '13px'},
                        selected_style={
                            'color': ACCENT, 'background': CARD,
                            'borderTop': f'2px solid {ACCENT}',
                            'padding': '8px 16px', 'fontSize': '13px',
                        },
                        children=[
                            html.Div(style={
                                'background': CARD,
                                'border': f'1px solid {BORDER}',
                                'borderRadius': '10px',
                                'padding': '16px',
                            }, children=[
                                dcc.Graph(
                                    id='prob-chart',
                                    figure=empty_prob_figure(),
                                    config={'displayModeBar': True,
                                            'modeBarButtonsToRemove': ['select2d','lasso2d'],
                                            'displaylogo': False},
                                ),
                            ]),
                        ],
                    ),

                    # ── Tab 2: Bloch Spheres ──────────────────────────────
                    dcc.Tab(
                        label='Bloch Spheres',
                        value='bloch',
                        style={'color': MUTED, 'background': PANEL,
                               'border': f'1px solid {BORDER}',
                               'padding': '8px 16px', 'fontSize': '13px'},
                        selected_style={
                            'color': ACCENT, 'background': CARD,
                            'borderTop': f'2px solid {ACCENT}',
                            'padding': '8px 16px', 'fontSize': '13px',
                        },
                        children=[
                            html.Div(style={
                                'background': CARD,
                                'border': f'1px solid {BORDER}',
                                'borderRadius': '10px',
                                'padding': '16px',
                            }, children=[

                                # Time slider
                                html.Div(style={
                                    'marginBottom': '20px',
                                    'padding': '12px 16px',
                                    'background': PANEL,
                                    'borderRadius': '8px',
                                    'border': f'1px solid {BORDER}',
                                }, children=[
                                    html.Div(style={
                                        'display': 'flex', 'justifyContent': 'space-between',
                                        'marginBottom': '8px',
                                    }, children=[
                                        html.Span('Time step', style={'color': MUTED, 'fontSize': '11px',
                                                                       'fontWeight': '600',
                                                                       'textTransform': 'uppercase',
                                                                       'letterSpacing': '0.07em'}),
                                        html.Span(id='time-label', children='—',
                                                  style={'color': ACCENT, 'fontSize': '12px',
                                                         'fontFamily': 'monospace'}),
                                    ]),
                                    dcc.Slider(
                                        id='time-slider',
                                        min=0, max=100, step=1, value=0,
                                        marks={0: '0', 25: '25', 50: '50',
                                               75: '75', 100: '100'},
                                        tooltip={'always_visible': False},
                                        updatemode='drag',
                                    ),
                                ]),

                                # Bloch spheres container
                                html.Div(id='bloch-container',
                                         style={'display': 'flex',
                                                'flexWrap': 'wrap',
                                                'gap': '12px',
                                                'justifyContent': 'center'}),
                            ]),
                        ],
                    ),

                    # ── Tab 3: Circuit Builder ────────────────────────────
                    dcc.Tab(
                        label='Circuit Builder',
                        value='circuit',
                        style={'color': MUTED, 'background': PANEL,
                               'border': f'1px solid {BORDER}',
                               'padding': '8px 16px', 'fontSize': '13px'},
                        selected_style={
                            'color': ACCENT, 'background': CARD,
                            'borderTop': f'2px solid {ACCENT}',
                            'padding': '8px 16px', 'fontSize': '13px',
                        },
                        children=[

                            # Top: transitions reference + circuit editor
                            html.Div(style={
                                'display': 'grid',
                                'gridTemplateColumns': '1fr 1fr',
                                'gap': '16px',
                                'marginBottom': '16px',
                            }, children=[

                                # Left: Allowed Transitions reference
                                html.Div(style={
                                    'background': CARD,
                                    'border': f'1px solid {BORDER}',
                                    'borderRadius': '10px',
                                    'padding': '16px',
                                }, children=[
                                    html.Div('Allowed Transitions', style={
                                        'fontSize': '13px', 'fontWeight': '700',
                                        'color': TEXT, 'marginBottom': '4px',
                                    }),
                                    html.Div(
                                        'Single spin-flip pairs (Hamming distance = 1). '
                                        'Only these can be driven by resonant π pulses.',
                                        style={'fontSize': '11px', 'color': MUTED,
                                               'marginBottom': '12px'},
                                    ),
                                    html.Div(
                                        id='transitions-table',
                                        style={'maxHeight': '380px', 'overflowY': 'auto'},
                                    ),
                                ]),

                                # Right: Circuit Editor
                                html.Div(style={
                                    'background': CARD,
                                    'border': f'1px solid {BORDER}',
                                    'borderRadius': '10px',
                                    'padding': '16px',
                                    'display': 'flex',
                                    'flexDirection': 'column',
                                    'gap': '6px',
                                }, children=[
                                    html.Div('Circuit Editor', style={
                                        'fontSize': '13px', 'fontWeight': '700',
                                        'color': TEXT, 'marginBottom': '4px',
                                        'paddingBottom': '8px',
                                        'borderBottom': f'1px solid {BORDER}',
                                    }),

                                    _label('Gate Type'),
                                    dcc.Dropdown(
                                        id='gate-type-dd',
                                        options=[
                                            {'label': 'π/2 pulse  — superposition (Hadamard-like)',
                                             'value': 'pi_half'},
                                            {'label': 'π pulse    — full inversion (NOT-like)',
                                             'value': 'pi'},
                                        ],
                                        value='pi_half',
                                        clearable=False,
                                        style={'background': '#0d1b2e', 'color': TEXT,
                                               'border': f'1px solid {BORDER}',
                                               'borderRadius': '6px', 'fontSize': '13px'},
                                    ),

                                    _label('Transition (allowed spin-flip pair)'),
                                    dcc.Dropdown(
                                        id='transition-dd',
                                        options=[],
                                        value=None,
                                        placeholder='Select a transition…',
                                        style={'background': '#0d1b2e', 'color': TEXT,
                                               'border': f'1px solid {BORDER}',
                                               'borderRadius': '6px', 'fontSize': '13px'},
                                    ),

                                    html.Button(
                                        '+ Add Gate',
                                        id='add-gate-btn',
                                        n_clicks=0,
                                        style={
                                            'background': SUCCESS, 'color': 'white',
                                            'border': 'none', 'borderRadius': '6px',
                                            'padding': '8px 16px', 'fontSize': '13px',
                                            'fontWeight': '600', 'cursor': 'pointer',
                                            'width': '100%', 'marginTop': '4px',
                                        },
                                    ),

                                    html.Div(style={
                                        'height': '1px', 'background': BORDER,
                                        'margin': '6px 0',
                                    }),

                                    html.Div('Current Circuit', style={
                                        'fontSize': '12px', 'fontWeight': '600',
                                        'color': MUTED,
                                    }),

                                    html.Div(
                                        id='circuit-display',
                                        style={
                                            'minHeight': '72px',
                                            'maxHeight': '190px',
                                            'overflowY': 'auto',
                                            'background': '#0d1b2e',
                                            'borderRadius': '6px',
                                            'border': f'1px solid {BORDER}',
                                            'padding': '8px',
                                        },
                                    ),

                                    html.Div(style={
                                        'display': 'flex', 'gap': '8px',
                                    }, children=[
                                        html.Button(
                                            'Remove Last',
                                            id='remove-last-btn',
                                            n_clicks=0,
                                            style={
                                                'flex': '1',
                                                'background': 'rgba(244,63,94,0.12)',
                                                'color': DANGER,
                                                'border': f'1px solid rgba(244,63,94,0.3)',
                                                'borderRadius': '6px', 'padding': '6px',
                                                'fontSize': '12px', 'cursor': 'pointer',
                                            },
                                        ),
                                        html.Button(
                                            'Clear All',
                                            id='clear-all-btn',
                                            n_clicks=0,
                                            style={
                                                'flex': '1',
                                                'background': 'rgba(100,116,139,0.1)',
                                                'color': MUTED,
                                                'border': f'1px solid {BORDER}',
                                                'borderRadius': '6px', 'padding': '6px',
                                                'fontSize': '12px', 'cursor': 'pointer',
                                            },
                                        ),
                                    ]),

                                    html.Div(style={
                                        'height': '1px', 'background': BORDER,
                                        'margin': '4px 0',
                                    }),

                                    dcc.Loading(
                                        type='circle', color=ACCENT,
                                        children=html.Button(
                                            '▶  Run Circuit',
                                            id='run-circuit-btn',
                                            n_clicks=0,
                                            style={
                                                'width': '100%', 'padding': '10px',
                                                'background': ACCENT, 'color': 'white',
                                                'border': 'none', 'borderRadius': '8px',
                                                'fontSize': '14px', 'fontWeight': '700',
                                                'cursor': 'pointer',
                                            },
                                        ),
                                    ),

                                    html.Div(id='circuit-status-msg', style={
                                        'fontSize': '11px', 'color': MUTED,
                                        'textAlign': 'center', 'minHeight': '16px',
                                    }),
                                ]),
                            ]),

                            # Bottom: Circuit simulation results
                            html.Div(style={
                                'background': CARD,
                                'border': f'1px solid {BORDER}',
                                'borderRadius': '10px',
                                'padding': '16px',
                            }, children=[
                                html.Div(style={
                                    'display': 'flex',
                                    'justifyContent': 'space-between',
                                    'alignItems': 'center',
                                    'marginBottom': '10px',
                                }, children=[
                                    html.Div('Circuit Simulation Results', style={
                                        'fontSize': '12px', 'fontWeight': '600',
                                        'color': MUTED,
                                    }),
                                    dcc.Loading(
                                        type='circle', color=ACCENT,
                                        children=html.Button(
                                            '⬇  Save PDF',
                                            id='save-pdf-btn',
                                            n_clicks=0,
                                            style={
                                                'background': 'rgba(79,142,247,0.12)',
                                                'color': ACCENT,
                                                'border': f'1px solid rgba(79,142,247,0.35)',
                                                'borderRadius': '6px',
                                                'padding': '6px 14px',
                                                'fontSize': '12px',
                                                'fontWeight': '600',
                                                'cursor': 'pointer',
                                            },
                                        ),
                                    ),
                                ]),
                                dcc.Graph(
                                    id='circuit-prob-chart',
                                    figure=empty_prob_figure(),
                                    config={
                                        'displayModeBar': True,
                                        'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                                        'displaylogo': False,
                                    },
                                ),
                            ]),
                        ],
                    ),
                ],
            ),
        ]),  # end main content
    ]),

    # Hidden data stores & download trigger
    dcc.Store(id='sim-store'),
    dcc.Store(id='circuit-store', data={'n_qubits': 3, 'gates': []}),
    dcc.Store(id='circuit-sim-store'),
    dcc.Download(id='pdf-download'),
])

# ── Helpers ─────────────────────────────────────────────────────────────────

def _parse_custom_amps(raw_values, num_s):
    """Slice, fill missing with 0, then L2-normalize. Returns a list of floats."""
    amps = [float(v or 0.0) for v in (raw_values or [])]
    amps = (amps + [0.0] * num_s)[:num_s]   # pad / trim to exactly num_s
    norm = np.sqrt(sum(a ** 2 for a in amps))
    if norm < 1e-12:                          # all-zero fallback → ground state
        amps = [0.0] * num_s
        amps[0] = 1.0
    else:
        amps = [a / norm for a in amps]
    return amps


# ── Callbacks ───────────────────────────────────────────────────────────────

@app.callback(
    Output('algo-label', 'children'),
    Input('n-qubits-dd', 'value'),
)
def update_algo_label(n):
    label = ALGO_LABELS.get(int(n or 3), '')
    return f'Algorithm: {label}' if label else ''


@app.callback(
    Output('sim-store',   'data'),
    Output('status-msg',  'children'),
    Output('time-slider', 'max'),
    Output('time-slider', 'marks'),
    Output('time-slider', 'value'),
    Input('run-btn', 'n_clicks'),
    State('n-qubits-dd', 'value'),
    State('zeta-in',     'value'),
    State('zeta0-in',    'value'),
    State('j-in',        'value'),
    State('step-dd',     'value'),
    State('init-dd',     'value'),
    State({'type': 'amp-input', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def run_sim(n_clicks, n_q, zeta, zeta0, j_val, step, init_preset, amp_values):
    n_q    = int(n_q  or 3)
    zeta   = float(zeta  or 100)
    zeta0  = float(zeta0 or 100)
    j_val  = float(j_val or 10)
    step   = float(step  or 0.01)
    num_s  = 2 ** n_q

    if init_preset == 'custom':
        amps = _parse_custom_amps(amp_values, num_s)
    elif init_preset == 'ground':
        amps = [0.0] * num_s
        amps[0] = 1.0
    elif init_preset == 'equal':
        amps = [1.0 / np.sqrt(num_s)] * num_s
    else:
        amps = None      # let run_simulation use its built-in default

    try:
        res  = run_simulation(
            n_qubits=n_q, zeta=zeta, initial_zeta=zeta0,
            j_interaction=j_val, step_size=step,
            initial_amplitudes=amps,
        )
        n_steps = len(res['T'])
        marks   = {
            0:           '0',
            n_steps//4:  f'{n_steps//4}',
            n_steps//2:  f'{n_steps//2}',
            3*n_steps//4: f'{3*n_steps//4}',
            n_steps-1:   f'{n_steps-1}',
        }
        status = f'✓  {n_steps} steps · {num_s} states · algorithm: {ALGO_LABELS.get(n_q,"")}'
        return res, status, n_steps - 1, marks, n_steps - 1
    except Exception as exc:
        return None, f'✗  {exc}', 100, {0:'0',100:'100'}, 0


@app.callback(
    Output('prob-chart', 'figure'),
    Input('sim-store', 'data'),
)
def update_prob(data):
    if not data:
        return empty_prob_figure()
    return probability_figure(data['T'], data['D'], data['states'])


@app.callback(
    Output('bloch-container', 'children'),
    Output('time-label',      'children'),
    Input('sim-store',    'data'),
    Input('time-slider',  'value'),
)
def update_bloch(data, t_idx):
    if not data:
        n_q = 3
        graphs = [
            dcc.Graph(
                figure=empty_bloch_figure(f'Qubit {k+1}', QUBIT_COLORS[k]),
                style={'flex': '1', 'minWidth': '280px', 'maxWidth': '380px'},
                config={'displayModeBar': False},
            )
            for k in range(n_q)
        ]
        return graphs, '—'

    n_q   = int(data['n_qubits'])
    t_idx = int(t_idx or 0)
    t_idx = max(0, min(t_idx, len(data['T']) - 1))
    t_val = data['T'][t_idx]

    vecs  = compute_bloch_vectors(data['cx'], data['cy'], n_q, t_idx)

    graphs = [
        dcc.Graph(
            figure=bloch_sphere_figure(
                vecs[k][0], vecs[k][1], vecs[k][2],
                f'Qubit {k+1}',
                QUBIT_COLORS[k % len(QUBIT_COLORS)],
            ),
            style={'flex': '1', 'minWidth': '260px', 'maxWidth': '380px'},
            config={'displayModeBar': False},
        )
        for k in range(n_q)
    ]
    return graphs, f't = {t_val:.4f}'


# ── Custom amplitude panel ───────────────────────────────────────────────────

@app.callback(
    Output('custom-amps-panel', 'children'),
    Input('init-dd',     'value'),
    Input('n-qubits-dd', 'value'),
)
def render_amp_panel(init_preset, n_q):
    if init_preset != 'custom':
        return []

    n_q      = int(n_q or 3)
    num_s    = 2 ** n_q
    all_s    = get_all_states(n_q)
    labels   = ['|' + ''.join(str(b) for b in all_s[i]) + '⟩' for i in range(num_s)]
    defaults = [1.0] + [0.0] * (num_s - 1)   # ground state as starting point

    inp_style = {
        'width': '80px', 'boxSizing': 'border-box',
        'background': '#0d1b2e', 'border': f'1px solid {BORDER}',
        'borderRadius': '4px', 'color': TEXT,
        'padding': '4px 6px', 'fontSize': '12px',
        'outline': 'none', 'textAlign': 'right',
    }

    rows = [
        html.Div(style={
            'display': 'flex', 'alignItems': 'center',
            'justifyContent': 'space-between',
            'padding': '2px 0',
        }, children=[
            html.Span(labels[i], style={
                'color': TEXT, 'fontSize': '12px',
                'fontFamily': 'monospace', 'letterSpacing': '0.02em',
            }),
            dcc.Input(
                id={'type': 'amp-input', 'index': i},
                type='number', value=defaults[i],
                debounce=True, style=inp_style,
            ),
        ])
        for i in range(num_s)
    ]

    return html.Div(style={
        'background': '#0a1220',
        'border': f'1px solid {BORDER}',
        'borderRadius': '6px',
        'padding': '10px 12px',
        'marginTop': '6px',
    }, children=[
        html.Div('Amplitudes (auto-normalized)', style={
            'fontSize': '11px', 'color': MUTED,
            'fontWeight': '600', 'textTransform': 'uppercase',
            'letterSpacing': '0.07em', 'marginBottom': '8px',
        }),
        html.Div(style={'maxHeight': '200px', 'overflowY': 'auto'}, children=rows),
        html.Div('Values are L2-normalized before simulation.',
                 style={'fontSize': '10px', 'color': MUTED,
                        'marginTop': '8px', 'fontStyle': 'italic'}),
    ])


# ── Circuit Builder callbacks ────────────────────────────────────────────────

@app.callback(
    Output('transitions-table', 'children'),
    Output('transition-dd',     'options'),
    Output('transition-dd',     'value'),
    Input('n-qubits-dd', 'value'),
    Input('zeta-in',     'value'),
    Input('zeta0-in',    'value'),
    Input('j-in',        'value'),
)
def update_transitions(n_q, zeta, zeta0, j_val):
    n_q   = int(n_q   or 3)
    zeta  = float(zeta  or 100)
    zeta0 = float(zeta0 or 100)
    j_val = float(j_val or 10)

    energies    = compute_energies(n_q, zeta, zeta0, j_val)
    transitions = get_allowed_transitions(n_q, energies)

    th_s = {
        'background': '#0d1b2e', 'color': MUTED,
        'fontSize': '11px', 'fontWeight': '600',
        'textTransform': 'uppercase', 'letterSpacing': '0.06em',
        'padding': '6px 8px', 'textAlign': 'left',
        'borderBottom': f'1px solid {BORDER}',
        'position': 'sticky', 'top': '0',
    }
    td_s = {
        'padding': '5px 8px', 'fontSize': '12px', 'color': TEXT,
        'borderBottom': f'1px solid rgba(30,45,69,0.5)',
        'fontFamily': 'monospace',
    }

    table = html.Table([
        html.Thead(html.Tr([
            html.Th('States', style=th_s),
            html.Th('ΔE', style={**th_s, 'textAlign': 'right'}),
            html.Th('ω resonant', style={**th_s, 'textAlign': 'right'}),
        ])),
        html.Tbody([
            html.Tr([
                html.Td(f"{t['label_k']} ↔ {t['label_j']}", style=td_s),
                html.Td(f"{t['energy_diff']:+.3f}",
                        style={**td_s, 'textAlign': 'right', 'color': ACCENT}),
                html.Td(f"{t['resonant_freq']:+.3f}",
                        style={**td_s, 'textAlign': 'right', 'color': SUCCESS}),
            ]) for t in transitions
        ]),
    ], style={'width': '100%', 'borderCollapse': 'collapse'})

    options = [
        {
            'label': f"{t['label_k']} ↔ {t['label_j']}  (ω={t['resonant_freq']:+.2f})",
            'value': f"{t['k']}-{t['j']}",
        }
        for t in transitions
    ]
    default = options[0]['value'] if options else None
    return table, options, default


@app.callback(
    Output('circuit-store',   'data'),
    Input('add-gate-btn',     'n_clicks'),
    Input('remove-last-btn',  'n_clicks'),
    Input('clear-all-btn',    'n_clicks'),
    Input('n-qubits-dd',      'value'),
    State('gate-type-dd',     'value'),
    State('transition-dd',    'value'),
    State('circuit-store',    'data'),
    prevent_initial_call=True,
)
def manage_circuit(add, rem, clr, n_q, gate_type, transition_val, circuit_data):
    n_q  = int(n_q or 3)
    data = circuit_data or {'n_qubits': n_q, 'gates': []}
    tid  = ctx.triggered_id

    if tid == 'n-qubits-dd':
        return {'n_qubits': n_q, 'gates': []}

    gates = list(data.get('gates', []))

    if tid == 'clear-all-btn':
        return {'n_qubits': n_q, 'gates': []}

    if tid == 'remove-last-btn' and gates:
        return {'n_qubits': n_q, 'gates': gates[:-1]}

    if tid == 'add-gate-btn' and transition_val:
        k, j  = [int(x) for x in transition_val.split('-')]
        all_s = get_all_states(n_q)
        lk = '|' + ''.join(str(b) for b in all_s[k - 1]) + '⟩'
        lj = '|' + ''.join(str(b) for b in all_s[j - 1]) + '⟩'
        gates.append({
            'type': gate_type or 'pi_half',
            'k': k, 'j': j,
            'label_k': lk, 'label_j': lj,
        })
        return {'n_qubits': n_q, 'gates': gates}

    return data


@app.callback(
    Output('circuit-display', 'children'),
    Input('circuit-store', 'data'),
)
def display_circuit(circuit_data):
    gates = (circuit_data or {}).get('gates', [])

    if not gates:
        return html.Div(
            'No gates added yet.',
            style={'color': MUTED, 'fontSize': '12px',
                   'fontStyle': 'italic', 'padding': '8px'},
        )

    type_style = {
        'pi_half': {'background': 'rgba(79,142,247,0.18)', 'color': ACCENT,
                    'border': f'1px solid rgba(79,142,247,0.35)'},
        'pi':      {'background': 'rgba(245,158,11,0.18)', 'color': '#f59e0b',
                    'border': '1px solid rgba(245,158,11,0.35)'},
    }
    type_label = {'pi_half': 'π/2', 'pi': 'π'}

    rows = []
    for i, g in enumerate(gates):
        ts = type_style.get(g['type'], type_style['pi_half'])
        rows.append(html.Div(style={
            'display': 'flex', 'alignItems': 'center', 'gap': '8px',
            'padding': '4px 6px', 'borderRadius': '4px',
            'background': 'rgba(255,255,255,0.02)', 'marginBottom': '3px',
        }, children=[
            html.Span(f'{i + 1}', style={
                'background': BORDER, 'color': MUTED,
                'borderRadius': '4px', 'padding': '1px 6px',
                'fontSize': '11px', 'fontWeight': '700',
                'minWidth': '18px', 'textAlign': 'center',
            }),
            html.Span(type_label[g['type']], style={
                **ts, 'borderRadius': '4px', 'padding': '1px 7px',
                'fontSize': '12px', 'fontWeight': '700', 'fontFamily': 'monospace',
            }),
            html.Span(f"{g['label_k']} → {g['label_j']}", style={
                'color': TEXT, 'fontSize': '12px', 'fontFamily': 'monospace',
            }),
        ]))
    return rows


@app.callback(
    Output('circuit-sim-store',  'data'),
    Output('circuit-status-msg', 'children'),
    Input('run-circuit-btn', 'n_clicks'),
    State('circuit-store',   'data'),
    State('n-qubits-dd',     'value'),
    State('zeta-in',         'value'),
    State('zeta0-in',        'value'),
    State('j-in',            'value'),
    State('step-dd',         'value'),
    State('init-dd',         'value'),
    State({'type': 'amp-input', 'index': ALL}, 'value'),
    prevent_initial_call=True,
)
def run_circuit_sim(n_clicks, circuit_data, n_q, zeta, zeta0, j_val, step, init_preset, amp_values):
    n_q   = int(n_q   or 3)
    zeta  = float(zeta  or 100)
    zeta0 = float(zeta0 or 100)
    j_val = float(j_val or 10)
    step  = float(step  or 0.01)
    num_s = 2 ** n_q

    gates = (circuit_data or {}).get('gates', [])
    if not gates:
        return None, '✗  Add at least one gate before running.'

    if init_preset == 'custom':
        amps = _parse_custom_amps(amp_values, num_s)
    elif init_preset == 'equal':
        amps = [1.0 / np.sqrt(num_s)] * num_s
    else:
        amps = [0.0] * num_s
        amps[0] = 1.0

    try:
        res = run_simulation(
            n_qubits=n_q, zeta=zeta, initial_zeta=zeta0,
            j_interaction=j_val, step_size=step,
            initial_amplitudes=amps, custom_gates=gates,
        )
        n_steps = len(res['T'])
        status  = f'✓  {n_steps} steps · {len(gates)} gates · {num_s} states'
        return res, status
    except Exception as exc:
        return None, f'✗  {exc}'


@app.callback(
    Output('circuit-prob-chart', 'figure'),
    Input('circuit-sim-store', 'data'),
)
def update_circuit_prob(data):
    if not data:
        return empty_prob_figure()
    return probability_figure(data['T'], data['D'], data['states'])


@app.callback(
    Output('pdf-download', 'data'),
    Input('save-pdf-btn', 'n_clicks'),
    State('circuit-store',     'data'),
    State('circuit-sim-store', 'data'),
    prevent_initial_call=True,
)
def save_pdf(n_clicks, circuit_data, sim_data):
    if not sim_data:
        return None
    pdf_bytes = generate_pdf(circuit_data, sim_data)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return dcc.send_bytes(pdf_bytes, f'quantum_circuit_{timestamp}.pdf')


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    host = '0.0.0.0'
    port = 8050
    print(f' * Open http://localhost:{port}/ in your browser')
    app.run(debug=True, host=host, port=port)
