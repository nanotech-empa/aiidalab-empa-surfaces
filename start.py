import ipywidgets as ipw


_ICON_STROKE = "currentColor"


ICONS = {
    "geometry": """
        <svg viewBox="0 0 64 64" aria-hidden="true">
            <path d="M9 49c10-23 18-23 26-8s13 11 20-17" />
            <path d="M42 20h13v13" />
            <circle cx="20" cy="23" r="5" />
            <circle cx="35" cy="29" r="4" />
            <circle cx="27" cy="39" r="3" />
            <path d="M24 25l7 3M33 32l-4 5" />
        </svg>
    """,
    "scf": """
        <svg viewBox="0 0 64 64" aria-hidden="true">
            <path d="M13 46h38" />
            <path d="M18 38c6-18 12-18 18 0s11 10 16-8" />
            <circle cx="20" cy="24" r="5" />
            <circle cx="34" cy="20" r="4" />
            <circle cx="45" cy="30" r="5" />
            <path d="M25 23l5-2M38 23l4 4" />
            <path d="M22 52h20" opacity=".55" />
        </svg>
    """,
    "adsorption": """
        <svg viewBox="0 0 64 64" aria-hidden="true">
            <path d="M10 45h44M15 51h34M18 39h28" />
            <circle cx="29" cy="19" r="5" />
            <circle cx="20" cy="27" r="4" />
            <circle cx="38" cy="27" r="4" />
            <path d="M25 22l-3 3M33 22l3 3" />
            <path d="M32 34v9" />
            <path d="M27 38l5 5 5-5" />
        </svg>
    """,
    "phonons": """
        <svg viewBox="0 0 64 64" aria-hidden="true">
            <path d="M8 32c6-14 12-14 18 0s12 14 18 0 8-10 12-6" />
            <circle cx="13" cy="42" r="5" />
            <circle cx="32" cy="42" r="5" />
            <circle cx="51" cy="42" r="5" />
            <path d="M18 42h9M37 42h9" />
            <path d="M13 22v-7M32 18v-7M51 22v-7" />
        </svg>
    """,
    "replica": """
        <svg viewBox="0 0 64 64" aria-hidden="true">
            <circle cx="12" cy="34" r="6" />
            <circle cx="32" cy="24" r="6" />
            <circle cx="52" cy="34" r="6" />
            <path d="M18 32l8-5M38 27l8 5" />
            <path d="M10 48c10 6 33 6 44 0" />
            <path d="M49 45l5 3-5 3" />
        </svg>
    """,
    "neb": """
        <svg viewBox="0 0 64 64" aria-hidden="true">
            <path d="M8 48c9 0 12-30 24-30s15 30 24 30" />
            <circle cx="10" cy="48" r="3" />
            <circle cx="20" cy="34" r="3" />
            <circle cx="32" cy="18" r="3" />
            <circle cx="44" cy="34" r="3" />
            <circle cx="54" cy="48" r="3" />
            <path d="M14 54h36" />
        </svg>
    """,
    "search": """
        <svg viewBox="0 0 64 64" aria-hidden="true">
            <circle cx="27" cy="27" r="14" />
            <path d="M38 38l13 13" />
            <path d="M22 22h10v10H22z" />
            <path d="M22 22l5-5 5 5M32 32l-5 5-5-5" />
        </svg>
    """,
    "spm": """
        <svg viewBox="0 0 64 64" aria-hidden="true">
            <path d="M18 7h28" />
            <path d="M22 12h20" opacity=".75" />
            <path d="M20 8l7 13h10l7-13" />
            <circle cx="27" cy="21" r="4" />
            <circle cx="37" cy="21" r="4" />
            <circle cx="32" cy="30" r="4" />
            <path d="M32 34v5" />
            <path d="M29 39h6" />
            <circle cx="17" cy="50" r="6" />
            <circle cx="29" cy="50" r="6" />
            <circle cx="41" cy="50" r="6" />
            <circle cx="53" cy="50" r="6" />
            <circle cx="23" cy="40" r="5" />
            <circle cx="35" cy="40" r="5" />
            <circle cx="47" cy="40" r="5" />
            <path d="M32 41c-3 2-6 2-9 0" opacity=".45" />
            <path d="M32 41c3 2 6 2 9 0" opacity=".45" />
        </svg>
    """,
    "pdos": """
        <svg viewBox="0 0 64 64" aria-hidden="true">
            <path d="M12 52h42M12 52V12" />
            <path d="M19 48V32M28 48V18M37 48V26M46 48V38" />
            <path d="M18 24c7 6 12 6 18 0s10-6 15 1" />
        </svg>
    """,
}


ITEMS = [
    (
        "Density functional theory",
        [
            ("geometry", "Geometry optimization", "Relax a structure to a local minimum.", "submit_geometry_optimization.ipynb"),
            ("scf", "SCF energy", "Single-point CP2K energy and optional remote overlap matrix.", "submit_scf.ipynb"),
            ("adsorption", "Adsorption energy", "Compare adsorbed and reference systems.", "submit_adsorption_energy.ipynb"),
            ("phonons", "Phonons", "Vibrational modes and finite-difference displacements.", "submit_phonons.ipynb"),
            ("replica", "Replica chain", "Constrained replicas along a reaction coordinate.", "submit_replica_chain.ipynb"),
            ("neb", "Nudged elastic band", "Images on a minimum-energy reaction path.", "submit_neb.ipynb"),
            ("search", "Search", "Find and reopen previous calculations.", "search.ipynb"),
        ],
    ),
    (
        "Post-processing",
        [
            ("spm", "Scanning probe microscopy", "STM, AFM, and orbital imaging workflows.", "submit_spm.ipynb"),
            ("pdos", "Projected density of states", "Orbital-resolved electronic spectra.", "submit_pdos.ipynb"),
        ],
    ),
]


def _card(appbase, icon, title, description, notebook):
    return f"""
        <a class="surface-card" href="{appbase}/{notebook}" target="_blank">
            <span class="surface-icon surface-icon-{icon}">{ICONS[icon]}</span>
            <span class="surface-card-text">
                <span class="surface-card-title">{title}</span>
                <span class="surface-card-description">{description}</span>
            </span>
        </a>
    """


def get_start_widget(appbase, jupbase):  # noqa: ARG001
    sections = []
    for title, items in ITEMS:
        cards = "".join(_card(appbase, *item) for item in items)
        sections.append(
            f"""
            <section class="surface-section">
                <h2>{title}</h2>
                <div class="surface-grid">{cards}</div>
            </section>
            """
        )

    return ipw.HTML(
        f"""
        <style>
            .surface-launcher {{
                --surface-ink: #1f2933;
                --surface-muted: #5b6673;
                --surface-line: #d8dee6;
                --surface-panel: #ffffff;
                --surface-hover: #f4f8fb;
                --surface-blue: #0c7bb3;
                --surface-green: #2a8c55;
                --surface-orange: #c66a1f;
                color: var(--surface-ink);
                max-width: 1120px;
                margin: 0 auto;
                padding: 12px 6px 20px;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }}
            .surface-launcher h1 {{
                font-size: 26px;
                font-weight: 650;
                margin: 0 0 18px;
            }}
            .surface-sections {{
                display: grid;
                grid-template-columns: minmax(0, 2fr) minmax(280px, 1fr);
                gap: 18px;
                align-items: start;
            }}
            .surface-section h2 {{
                font-size: 16px;
                font-weight: 650;
                margin: 0 0 10px;
                padding-bottom: 7px;
                border-bottom: 1px solid var(--surface-line);
            }}
            .surface-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 10px;
            }}
            .surface-card {{
                display: grid;
                grid-template-columns: 54px minmax(0, 1fr);
                gap: 12px;
                min-height: 86px;
                padding: 12px;
                border: 1px solid var(--surface-line);
                border-radius: 7px;
                background: var(--surface-panel);
                color: inherit;
                text-decoration: none;
                box-sizing: border-box;
                transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
            }}
            .surface-card:hover,
            .surface-card:focus {{
                background: var(--surface-hover);
                border-color: #98b7cc;
                color: inherit;
                text-decoration: none;
                transform: translateY(-1px);
            }}
            .surface-icon {{
                width: 52px;
                height: 52px;
                border-radius: 7px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background: #edf4f8;
                color: var(--surface-blue);
            }}
            .surface-icon svg {{
                width: 42px;
                height: 42px;
                fill: none;
                stroke: {_ICON_STROKE};
                stroke-width: 3;
                stroke-linecap: round;
                stroke-linejoin: round;
            }}
            .surface-icon-adsorption,
            .surface-icon-replica {{
                color: var(--surface-green);
                background: #eef7f1;
            }}
            .surface-icon-phonons,
            .surface-icon-neb,
            .surface-icon-spm {{
                color: var(--surface-orange);
                background: #fff3e8;
            }}
            .surface-card-text {{
                display: flex;
                flex-direction: column;
                justify-content: center;
                min-width: 0;
            }}
            .surface-card-title {{
                font-size: 15px;
                font-weight: 650;
                line-height: 1.2;
                margin-bottom: 5px;
            }}
            .surface-card-description {{
                font-size: 13px;
                line-height: 1.35;
                color: var(--surface-muted);
            }}
            @media (max-width: 860px) {{
                .surface-sections {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
        <div class="surface-launcher">
            <h1>Surfaces workflows</h1>
            <div class="surface-sections">
                {''.join(sections)}
            </div>
        </div>
        """
    )
