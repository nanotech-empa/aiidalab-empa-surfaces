"""Optional chat bindings for host-declared notebook forms and actions."""

FIELD_NAMES = (
    "protocol",
    "workflow_description",
    "text_calc_description",
    "tot_charge",
    "spin_u",
    "spin_d",
    "cell_opt",
    "max_nodes",
    "mem_node",
    "max_hours",
    "pseudo_family_dropdown",
    "pw_code_dropdown",
    "pp_code_dropdown",
    "projwfc_code_dropdown",
    "code_input_widget",
    "cubehandler_code",
    "parent_calc_folder",
    "skip_wfn_restart_search",
)


def make_structure_manager(*args, **kwargs):
    """Construct the unchanged GUI and retain explicit importer references."""
    import aiidalab_widgets_base as awb
    import aiidalab_widgets_empa as awe

    manager = awb.StructureManagerWidget(*args, **kwargs)
    kinds = (
        ("file", awb.StructureUploadWidget),
        ("database", awb.StructureBrowserWidget),
        ("cdxml", awe.CdxmlUploadWidget),
    )
    manager.assistant_importers = {
        name: importer
        for importer in kwargs.get("importers", ())
        for name, cls in kinds
        if isinstance(importer, cls)
    }
    manager.assistant_editors = tuple(kwargs.get("editors", ()))
    return manager


def create_panel(page, namespace):
    try:
        from aiidalab_ai_assistant.forms import panel
        from aiidalab_ai_assistant.guided import (
            GuidedFormAdapter,
            form_controls,
            messages,
        )
        from aiidalab_ai_assistant.structures import StructureTools
    except ModuleNotFoundError as exc:
        if not exc.name.startswith("aiidalab_ai_assistant"):
            raise
        import ipywidgets as ipw

        return ipw.HTML(
            "AI assistant is optional; install aiidalab-ai-assistant to enable it."
        )
    from .assistant_bindings import DRAFT_BUTTONS, EXECUTION_BUTTONS, FORM_ROOTS

    roots = {
        name: namespace[name]
        for name in FORM_ROOTS.get(page, FIELD_NAMES)
        if name in namespace
    }
    draft = {
        name: namespace[name]
        for name in DRAFT_BUTTONS.get(page, ())
        if name in namespace
    }
    execution = {
        name: namespace[name]
        for name in EXECUTION_BUTTONS.get(page, ())
        if name in namespace
    }
    selector = namespace.get("structure_selector")
    if selector is None and page == "cdxml_editor.ipynb":
        selector = namespace.get("editor")
    structure_tools = StructureTools(selector) if selector is not None else None
    if selector is not None:
        roots.update(
            {
                f"geometry_editor.{i}": widget
                for i, widget in enumerate(getattr(selector, "assistant_editors", ()))
            }
        )
    microscopy = (
        namespace.get("stm_microscopy_export") if page == "view_stm.ipynb" else None
    )
    plotter = namespace.get("series_plotter_inst") if page == "view_stm.ipynb" else None
    spm_buttons = namespace.get("submit_button") if page == "submit_spm.ipynb" else None

    diagnostics = [
        namespace[name]
        for name in (
            "output",
            "submit_out",
            "info",
            "unfolding_status",
            "submission_links",
        )
        if name in namespace
    ]

    electrostatics = getattr(roots.get("input_details"), "electrostatics", None)
    resources = roots.get("resources")
    dft = roots.get("dft_settings_widget")
    details = roots.get("input_details")

    def neb_section():
        if page == "submit_neb.ipynb":
            return next(
                (
                    section
                    for section in getattr(details, "displayed_sections", ())
                    if hasattr(section, "n_replica_per_group")
                ),
                None,
            )

    def fields():
        section = neb_section()
        named = {}
        if section is not None:
            named = {
                "neb.restart_from": section.restart_from,
                "neb.replica_count": section.n_replica,
                "neb.replicas_per_group": section.n_replica_per_group,
                "neb.band_type": section.band_type,
                "neb.k_spring": section.k_spring,
                "neb.align_frames": section.align_frames,
                "neb.rotate_frames": section.rotate_frames,
                "neb.optimize_endpoints": section.optimize_endpoints,
                "neb.steps_before_ci": section.nsteps_it,
            }
        if plotter is not None:
            named["stm.series"] = plotter.selections_vbox
        if microscopy is not None:
            named["stm.include_curve"] = microscopy.include_curve
            named.update(
                {
                    "stm.image_width_angstrom": microscopy.area_width,
                    "stm.image_height_angstrom": microscopy.area_height,
                    "stm.image_center_x_angstrom": microscopy.area_center_x,
                    "stm.image_center_y_angstrom": microscopy.area_center_y,
                }
            )
            if microscopy.export_widget is not None:
                named["stm.export_review"] = microscopy.export_widget
        values, _ = form_controls({**named, **roots})
        if electrostatics is not None:
            values["electrostatics.apply_selected"] = electrostatics.active
        if structure_tools:
            values.update(structure_tools.fields())
        return values

    def draft_buttons():
        _, values = form_controls(roots)
        values.update(draft)
        if microscopy is not None:
            values["prepare_microscopy_export"] = microscopy.prepare_button
            values["stm.reset_image_area"] = microscopy.reset_area_button
            if microscopy.export_widget is not None:
                _, review_buttons = form_controls(
                    {"stm.export_review": microscopy.export_widget}
                )
                values.update(review_buttons)
        return values

    def execution_buttons():
        active = dict(execution)
        if microscopy is not None and microscopy.export_widget is not None:
            active["export_stm_microscopy"] = (
                microscopy.export_widget.save_simulations_button
            )
        if spm_buttons is not None:
            active = {
                name: widget
                for name, widget in execution.items()
                if widget in spm_buttons.children
            }
        return {
            name: getattr(widget, "btn_submit", widget)
            for name, widget in active.items()
        }

    def context():
        submitted = {}
        for name, widget in execution.items():
            process = getattr(widget, "process", None)
            if process is not None:
                submitted[name] = str(process.uuid)
            del process
        result = {
            "page": page,
            "messages": messages([*roots.values(), *execution.values(), *diagnostics]),
            "submitted_processes": submitted,
            "execution": "Review science and resources, then request the current execution action. Only the user's action button executes it.",
        }
        if microscopy is not None:
            result.update(microscopy.context())
            try:
                result["stm_selection"] = plotter.microscopy_selection()
            except ValueError as error:
                result["stm_selection"] = {"error": str(error)}
        if electrostatics is not None:
            result["electrostatics"] = electrostatics.submission_state()
        section = neb_section()
        if section is not None:
            from .utils.neb_resources import resource_layout

            result["neb"] = {
                "restart_from": section.restart_from.value,
                "saved_replicas": section._restart_replica_count,
                "restart_message": section.restart_info.value,
                "layout": resource_layout(
                    section.n_replica.value,
                    section.n_replica_per_group.value,
                    resources.nodes,
                    resources.tasks_per_node,
                    resources.threads_per_task,
                ),
            }
            if section._restart_replica_count is not None:
                ready, message, draft_params = details.return_final_dictionary()
                result["neb"]["input_validation"] = {"valid": ready, "message": message}
                if ready:
                    from aiida import orm

                    parent = orm.load_node(draft_params["restart_from"])
                    changed = {}
                    for port in ("neb_params", "sys_params", "dft_params"):
                        previous = parent.inputs[port].get_dict()
                        current = draft_params.get(port, {})
                        changed[port] = sorted(
                            key
                            for key in previous.keys() | current.keys()
                            if key not in previous
                            or key not in current
                            or previous[key] != current[key]
                        )
                    result["neb"]["changed_parent_settings"] = changed
        if electrostatics is not None and hasattr(resources, "tasks_per_node"):
            result["cp2k_resources"] = {
                "nodes": resources.nodes,
                "ranks_per_node": resources.tasks_per_node,
                "cpus_per_rank": resources.threads_per_task,
                "wallclock_seconds": resources.walltime_seconds,
            }
            if dft is not None and dft.xc_functional.value == "PBE0":
                result["cp2k_resources"]["hfx_memory_mib_per_rank"] = (
                    dft.hfx_max_memory.value
                )
        if structure_tools:
            result.update(structure_tools.context())
        return result

    # Do not retain the notebook namespace, credential widgets or ORM objects.
    return panel(
        GuidedFormAdapter(
            "surfaces",
            fields,
            context=context,
            draft_buttons=draft_buttons,
            execution_buttons=execution_buttons,
            structure_tools=structure_tools,
            prepare_export=microscopy.prepare if microscopy is not None else None,
        )
    )
