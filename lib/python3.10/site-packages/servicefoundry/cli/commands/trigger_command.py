from typing import Sequence

import rich_click as click

from servicefoundry.cli.const import COMMAND_CLS, GROUP_CLS
from servicefoundry.cli.util import handle_exception_wrapper
from servicefoundry.lib.dao import application


@click.group(name="trigger", cls=GROUP_CLS)
def trigger_command():
    """
    Trigger a deployed job asynchronously
    """
    pass


@click.command(
    name="job",
    cls=COMMAND_CLS,
    context_settings=dict(ignore_unknown_options=True),
    help="Trigger a Job asynchronously",
)
@click.option(
    "--application-fqn",
    "--application_fqn",
    type=click.STRING,
    required=True,
    help="FQN of the deployment of the Job. This can be found on the Job details page.",
)
@click.argument("job_command", nargs=-1, type=click.UNPROCESSED)
@handle_exception_wrapper
def trigger_job(application_fqn: str, job_command: Sequence[str]):
    application.trigger_job(application_fqn=application_fqn, command=job_command)


def get_trigger_command():
    trigger_command.add_command(trigger_job)
    return trigger_command
