# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Run all scenarios in the tool sandbox."""

import argparse
import datetime
import json
import multiprocessing
import random
import subprocess
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any, Optional, Union

import polars as pl
from tqdm import tqdm

from tool_sandbox.cli.utils import (
    AGENT_TYPE_TO_FACTORY,
    TEST_SCENARIO_NAMES,
    USER_TYPE_TO_FACTORY,
    RoleImplType,
    get_category_summary,
    get_category_to_scenario_count,
    get_necessary_tool_name_to_scenario_count,
    resolve_scenarios,
    run_scenario,
    run_scenario_with_config,
)
from tool_sandbox.common.execution_context import ScenarioCategories
from tool_sandbox.common.scenario import Scenario
from tool_sandbox.common.tool_discovery import ToolBackend

DEFAULT_USER_TYPE = RoleImplType.GPT_4_o_2024_05_13


def get_git_sha() -> Optional[str]:
    """Get the git SHA of the `HEAD` branch."""
    # From https://stackoverflow.com/a/21901260
    # Note that there are some 3rd party Python modules for interacting with git. I have
    # tried `pygit2` and `GitPython`, but both failed to get the commit associated with
    # `HEAD` for me.
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError:
        # The tool sandbox script was not executed from within the git repository so we
        # cannot figure out the git SHA.
        return None


def has_local_changes() -> bool:
    # From https://stackoverflow.com/a/3878934 . `git diff --exit-code` will return 0 if
    # there are no local changes. The `--quiet` suppresses printing to stdout. Note that
    # this approach does not detect untracked files, but this should be fine for our
    # purposes.
    completed_proc = subprocess.run(["git", "diff", "--exit-code", "--quiet"])
    return completed_proc.returncode == 1


def write_result_summary(
    result_summary: list[dict[str, Any]],
    category_summary: dict[str, dict[str, list[float]]],
    output_directory: Path,
) -> None:
    # Try to get the current git SHA so that there is some provenance on with which
    # version of the code results have been generated with.
    git_sha = get_git_sha()
    if git_sha is not None and has_local_changes():
        git_sha += " + local changes"

    with open(output_directory / "result_summary.json", "w") as f:
        json.dump(
            {
                "per_scenario_results": result_summary,
                "category_aggregated_results": {
                    category: {k: sum(v) / len(v) for k, v in aggregation.items()}
                    for category, aggregation in category_summary.items()
                },
                "git_sha": git_sha,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )


def run_sandbox(
    *,
    agent_type: RoleImplType,
    user_type: RoleImplType,
    name_to_scenario: dict[str, Scenario],
    processes: int,
    output_base_dir: Path,
    agent_config: Optional[dict[str, Any]] = None,
) -> None:
    """Entry point for Tool Sandbox

    Args:
        agent_type:       The agent type to use (ignored when *agent_config* is set).
        user_type:        The user type to use.
        name_to_scenario: Dictionary from scenario name to scenario definition.
        processes:        Number of processes to run in parallel.
        output_base_dir:  Base directory for model outputs.
        agent_config:     Optional JSON config dict for complex agent topologies
                          (tool-filtered, multi-agent, etc.).  When provided,
                          *agent_type* is ignored.
    """
    # Show all rows and all columns when converting polars dataframes to strings.
    pl.Config.set_tbl_rows(-1).set_tbl_cols(-1).set_fmt_str_lengths(10000)
    pl.Config.set_tbl_formatting("ASCII_FULL")

    if agent_config is not None:
        from tool_sandbox.cli.agent_config import build_agent_from_config

        agent = build_agent_from_config(agent_config)
    else:
        agent = AGENT_TYPE_TO_FACTORY[agent_type]()
    user = USER_TYPE_TO_FACTORY[user_type]()
    # Sanitize model names for use in directory paths (e.g. ':' is invalid on Windows)
    agent_label = str(getattr(agent, 'model_name', agent_type)).replace(':', '-')
    user_label = str(getattr(user, 'model_name', user_type)).replace(':', '-')
    output_directory = (
        Path(output_base_dir) / f"agent_{agent_label}_"
        f"user_{user_label}_"
        f"{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
    )
    print(f"Storing outputs to '{output_directory}'.")

    # Print a category-wise count before playing scenarios
    category_counter: Counter[Union[ScenarioCategories, str]] = (
        get_category_to_scenario_count(name_to_scenario)
    )
    print(
        "Number of test cases per category:",
        json.dumps(
            {str(k): v for k, v in category_counter.most_common(len(category_counter))},
            indent=4,
            ensure_ascii=False,
        ),
    )
    # Print a necessary tool-wise count before playing scenarios
    necessary_tool_counter: Counter[str] = get_necessary_tool_name_to_scenario_count(
        name_to_scenario
    )
    print(
        "Number of test cases per necessary tool name:",
        json.dumps(
            {
                str(k): v
                for k, v in necessary_tool_counter.most_common(
                    len(necessary_tool_counter)
                )
            },
            indent=4,
            ensure_ascii=False,
        ),
    )
    # Shuffle scenarios for load balancing
    name_and_scenario_list = list(name_to_scenario.items())
    random.shuffle(name_and_scenario_list)
    num_scenarios = len(name_and_scenario_list)

    # Choose the correct run_scenario variant depending on whether a config
    # was provided.  For the config path we use ``run_scenario_with_config``
    # which rebuilds the agent from the serialisable config dict inside each
    # worker (required for multiprocessing which pickles arguments).
    if agent_config is not None:
        _run_fn = partial(
            run_scenario_with_config,
            agent_config=agent_config,
            user_type=user_type,
            output_directory=output_directory,
        )
    else:
        _run_fn = partial(
            run_scenario,
            agent_type=agent_type,
            user_type=user_type,
            output_directory=output_directory,
        )

    if processes > 1 and num_scenarios > 1:
        mpctx = multiprocessing.get_context("spawn")
        with mpctx.Pool(min(processes, num_scenarios)) as pool:
            result_summary = pool.map(_run_fn, name_and_scenario_list)
    else:
        result_summary = []
        tqdm_iterator = tqdm(name_and_scenario_list, desc="Scenarios")
        for name_and_scenario in tqdm_iterator:
            result_summary.append(_run_fn(name_and_scenario))

    # Aggregate results by category
    category_summary = get_category_summary(result_summary)
    write_result_summary(
        result_summary=result_summary,
        category_summary=category_summary,
        output_directory=output_directory,
    )


def main() -> None:
    random.seed(42)
    parser = argparse.ArgumentParser(description=__doc__)
    agent_selection_group = parser.add_mutually_exclusive_group()
    agent_selection_group.add_argument(
        "--agent",
        help="Agent type (simple enum-based selection).",
        default=None,
        choices=[str(t) for t in AGENT_TYPE_TO_FACTORY.keys()],
    )
    agent_selection_group.add_argument(
        "--agent_config",
        help=(
            "Path to a JSON file describing the agent configuration. "
            "Supports tool-filtered agents, multi-agent ensembles, and "
            "nested compositions. See tool_sandbox/cli/agent_config.py for "
            "the configuration format. Mutually exclusive with --agent."
        ),
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--user",
        help="User type.",
        default=str(DEFAULT_USER_TYPE),
        choices=[str(t) for t in USER_TYPE_TO_FACTORY.keys()],
    )
    parser.add_argument(
        "--preferred_tool_backend",
        help="Preferred tool backend to use.",
        default="DEFAULT",
        choices=[str(t) for t in ToolBackend],
    )
    scenario_selection_group = parser.add_mutually_exclusive_group()
    scenario_selection_group.add_argument(
        "-t",
        "--test_mode",
        action="store_true",
        help="Only run a few scenarios rather than the full suite.",
    )
    scenario_selection_group.add_argument(
        "-s",
        "--scenarios",
        nargs="*",
        help="Name of scenarios to run.",
        required=False,
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=16,
        help="Max number of processes for running scenarios in parallel.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("data"),
        help="Output base directory.",
    )
    args = parser.parse_args()

    # The parser for `--test_mode` and `--scenarios` are in a mutually exclusive group
    # so we can safely ignore the value of `args.scenarios` when `args.test_mode` is
    # true.
    scenario_names = TEST_SCENARIO_NAMES if args.test_mode else args.scenarios

    name_to_scenario = resolve_scenarios(
        desired_scenario_names=scenario_names,
        preferred_tool_backend=args.preferred_tool_backend,
    )

    # Resolve agent configuration: either from --agent_config JSON file or
    # the --agent enum (defaulting to GPT_4_o_2024_05_13 when neither given).
    agent_config: Optional[dict[str, Any]] = None
    if args.agent_config is not None:
        with open(args.agent_config, "r", encoding="utf-8") as f:
            agent_config = json.load(f)
        # agent_type is unused when agent_config is set, but run_sandbox still
        # requires it as a parameter for the non-config path.
        agent_type = RoleImplType.GPT_4_o_2024_05_13
    else:
        agent_type_str = args.agent or "GPT_4_o_2024_05_13"
        agent_type = RoleImplType(agent_type_str)

    user_type = RoleImplType(args.user)
    run_sandbox(
        agent_type=agent_type,
        user_type=user_type,
        name_to_scenario=name_to_scenario,
        processes=args.parallel,
        output_base_dir=args.output_dir,
        agent_config=agent_config,
    )


if __name__ == "__main__":
    main()
