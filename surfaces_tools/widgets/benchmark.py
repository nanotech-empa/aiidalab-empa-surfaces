"""Find benchmark CalcJobs from their scheduler job IDs."""

from html import escape

import ipywidgets as ipw
from aiida import orm
from aiida.common.links import LinkType


def find_benchmark_calcjobs(workflow_uuid, job_id):
    """Return plain details for matching CP2K jobs called by this benchmark."""
    job_id = job_id.strip()
    if not job_id:
        return []
    query = (
        orm.QueryBuilder()
        .append(
            orm.WorkChainNode,
            filters={"uuid": workflow_uuid},
            tag="benchmark",
        )
        .append(
            orm.CalcJobNode,
            with_incoming="benchmark",
            edge_filters={"type": LinkType.CALL_CALC.value},
            filters={
                "attributes.job_id": job_id,
                "process_type": "aiida.calculations:cp2k",
            },
            project=["uuid"],
        )
    )
    matches = []
    for (calc_uuid,) in query.all():
        calc = orm.load_node(calc_uuid)
        matches.append(
            {
                "pk": calc.pk,
                "uuid": str(calc.uuid),
                "job_id": calc.get_job_id(),
                "computer": calc.computer.label if calc.computer else "Unknown",
                "remote_path": calc.get_remote_workdir(),
                "state": calc.process_state.value if calc.process_state else "unknown",
                "exit_status": calc.exit_status,
            }
        )
        del calc
    return sorted(matches, key=lambda item: item["pk"])


class BenchmarkJobLookup(ipw.VBox):
    """Look up a job on demand, retaining only the benchmark UUID in state."""

    def __init__(self, workflow_uuid):
        self.workflow_uuid = str(workflow_uuid)
        self.job_id = ipw.Text(
            description="Scheduler job ID:",
            placeholder="e.g. 123456",
            style={"description_width": "initial"},
        )
        self.find_button = ipw.Button(description="Find CalcJob", icon="search")
        self.result = ipw.HTML()
        super().__init__(
            [
                ipw.HTML(
                    "<b>Find CP2K CalcJob</b><br>"
                    "Search this benchmark by scheduler job ID, including failed jobs."
                ),
                ipw.HBox([self.job_id, self.find_button]),
                self.result,
            ]
        )
        self.find_button.on_click(self.find_job)
        self.job_id.observe(self.clear_result, names="value")

    def clear_result(self, _):
        self.result.value = ""

    def find_job(self, _=None):
        job_id = self.job_id.value.strip()
        if not job_id:
            self.result.value = "Enter a scheduler job ID."
            return
        try:
            matches = find_benchmark_calcjobs(self.workflow_uuid, job_id)
        except Exception as exc:
            self.result.value = f"Cannot look up CalcJob: {escape(str(exc))}"
            return
        if not matches:
            self.result.value = (
                f"No CP2K CalcJob with job ID {escape(job_id)} "
                "was found in this benchmark."
            )
            return
        blocks = []
        for item in matches:
            pk = item["pk"]
            state = escape(item["state"])
            if item["exit_status"] is not None:
                state += f" (exit status {item['exit_status']})"
            remote_path = item["remote_path"]
            directory = (
                f'<code style="overflow-wrap: anywhere">{escape(remote_path)}</code>'
                f"<br>Open workdir from a terminal: "
                f"<code>verdi calcjob gotocomputer {pk}</code>"
                if remote_path
                else "No remote work directory is recorded yet."
            )
            blocks.append(
                f"<b>CP2K CalcJob PK: {pk}</b> &nbsp; "
                f'<a href="../home/process.ipynb?id={item["uuid"]}" '
                'target="_blank" rel="noopener noreferrer">Inspect CalcJob</a>'
                f"<br>Job ID: {escape(str(item['job_id']))} · "
                f"Computer: {escape(item['computer'])} · State: {state}"
                f"<br>Workdir: {directory}"
            )
        self.result.value = "<br><br>".join(blocks)
