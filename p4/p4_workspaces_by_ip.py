#!/usr/bin/env python3
"""Look up Perforce workspaces by machine IP address."""

from __future__ import annotations

import argparse
import ipaddress
import locale
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence


DEFAULT_COMMAND_TIMEOUT = 3


@dataclass(frozen=True)
class WorkspaceInfo:
    """A Perforce workspace entry."""

    name: str
    host: str
    owner: str
    root: str


@dataclass(frozen=True)
class LookupResult:
    """The outcome of resolving an IP to Perforce workspaces."""

    ip_address: str
    resolved_host_names: tuple[str, ...]
    workspaces: tuple[WorkspaceInfo, ...]


def normalize_host_name(name: str) -> str:
    """Normalize a machine name for case-insensitive matching."""
    return name.strip().rstrip(".").split(".", 1)[0].upper()


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    """Return unique non-empty values in their first-seen order."""
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned:
            continue
        key = normalize_host_name(cleaned)
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


class CommandRunner:
    """Run subprocess commands with the local text encoding."""

    def __init__(self, timeout: int = DEFAULT_COMMAND_TIMEOUT) -> None:
        self.timeout = timeout
        self.encoding = locale.getpreferredencoding(False) or "utf-8"

    def run(
        self,
        args: Sequence[str],
        *,
        check: bool = False,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            encoding=self.encoding,
            errors="replace",
            timeout=timeout or self.timeout,
            check=check,
        )


class HostResolver:
    """Resolve likely machine names for an IP using several Windows-friendly strategies."""

    def __init__(self, runner: CommandRunner | None = None) -> None:
        self.runner = runner or CommandRunner()

    def resolve_host_names(self, ip_address: str) -> List[str]:
        candidates: List[str] = []
        for strategy in (
            self._resolve_via_dns,
            self._resolve_via_ping,
            self._resolve_via_nbtstat,
        ):
            candidates.extend(strategy(ip_address))
        return dedupe_preserve_order(candidates)

    def _resolve_via_dns(self, ip_address: str) -> List[str]:
        command = [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            (
                f"Resolve-DnsName -Name '{ip_address}' -Type PTR -ErrorAction Stop | "
                "Select-Object -ExpandProperty NameHost"
            ),
        ]
        try:
            completed = self.runner.run(command)
        except (FileNotFoundError, subprocess.SubprocessError):
            return []
        if completed.returncode != 0:
            return []
        return [line.strip() for line in completed.stdout.splitlines()]

    def _resolve_via_ping(self, ip_address: str) -> List[str]:
        try:
            completed = self.runner.run(["ping", "-a", ip_address, "-n", "1", "-w", "1000"])
        except (FileNotFoundError, subprocess.SubprocessError):
            return []
        if completed.returncode != 0:
            return []

        target_marker = f"[{ip_address}]"
        for line in completed.stdout.splitlines():
            if target_marker not in line:
                continue
            prefix = line.split(target_marker, 1)[0].strip()
            if not prefix:
                continue
            host = prefix.split()[-1]
            if host and host != ip_address:
                return [host]
        return []

    def _resolve_via_nbtstat(self, ip_address: str) -> List[str]:
        try:
            completed = self.runner.run(["nbtstat", "-A", ip_address])
        except (FileNotFoundError, subprocess.SubprocessError):
            return []
        if completed.returncode != 0:
            return []

        pattern = re.compile(r"^\s*([^\s<]+)\s*<(20|00)>\s+UNIQUE\b", re.IGNORECASE)
        preferred: List[str] = []
        fallback: List[str] = []
        for line in completed.stdout.splitlines():
            match = pattern.match(line)
            if not match:
                continue
            host = match.group(1)
            suffix = match.group(2)
            if suffix == "20":
                preferred.append(host)
            else:
                fallback.append(host)
        return preferred or fallback


class PerforceClient:
    """Read Perforce workspace metadata from the local p4 CLI."""

    def __init__(self, runner: CommandRunner | None = None) -> None:
        self.runner = runner or CommandRunner(timeout=15)
        self._workspace_cache: List[WorkspaceInfo] | None = None

    def list_workspaces(self) -> List[WorkspaceInfo]:
        if self._workspace_cache is not None:
            return list(self._workspace_cache)

        try:
            completed = self.runner.run(["p4", "-ztag", "clients", "-a"])
        except FileNotFoundError as exc:
            raise RuntimeError("Unable to execute 'p4'. Ensure the Perforce CLI is installed.") from exc
        except subprocess.SubprocessError as exc:
            raise RuntimeError(f"Failed to execute 'p4 clients -a': {exc}") from exc

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise RuntimeError(stderr or "Failed to query Perforce workspaces.")

        self._workspace_cache = list(self._parse_tagged_clients(completed.stdout))
        return list(self._workspace_cache)

    def find_workspaces_by_host(self, host_name: str) -> List[WorkspaceInfo]:
        expected_host = normalize_host_name(host_name)
        return [
            workspace
            for workspace in self.list_workspaces()
            if workspace.host and normalize_host_name(workspace.host) == expected_host
        ]

    def _parse_tagged_clients(self, output: str) -> Iterable[WorkspaceInfo]:
        current: dict[str, str] = {}
        for raw_line in output.splitlines():
            line = raw_line.rstrip()
            if not line:
                if current:
                    workspace = self._workspace_from_record(current)
                    if workspace is not None:
                        yield workspace
                    current = {}
                continue

            match = re.match(r"^\.\.\.\s+(\S+)\s?(.*)$", line)
            if not match:
                continue
            key, value = match.groups()
            current[key] = value.strip()

        if current:
            workspace = self._workspace_from_record(current)
            if workspace is not None:
                yield workspace

    @staticmethod
    def _workspace_from_record(record: dict[str, str]) -> WorkspaceInfo | None:
        name = record.get("client", "")
        if not name:
            return None
        return WorkspaceInfo(
            name=name,
            host=record.get("Host", ""),
            owner=record.get("Owner", ""),
            root=record.get("Root", ""),
        )


def find_workspaces_by_ip(
    ip_address: str,
    *,
    resolver: HostResolver | None = None,
    perforce: PerforceClient | None = None,
) -> List[WorkspaceInfo]:
    """Resolve an IP to machine names and return matching Perforce workspaces."""
    return list(lookup_workspaces_by_ip(ip_address, resolver=resolver, perforce=perforce).workspaces)


def lookup_workspaces_by_ip(
    ip_address: str,
    *,
    resolver: HostResolver | None = None,
    perforce: PerforceClient | None = None,
) -> LookupResult:
    """Resolve an IP to machine names and return structured lookup details."""
    resolver = resolver or HostResolver()
    perforce = perforce or PerforceClient()

    host_names = resolver.resolve_host_names(ip_address)
    workspaces_by_host: dict[str, List[WorkspaceInfo]] = {}
    for workspace in perforce.list_workspaces():
        if not workspace.host:
            continue
        key = normalize_host_name(workspace.host)
        workspaces_by_host.setdefault(key, []).append(workspace)

    results: List[WorkspaceInfo] = []
    seen: set[str] = set()
    for host_name in host_names:
        for workspace in workspaces_by_host.get(normalize_host_name(host_name), []):
            if workspace.name in seen:
                continue
            seen.add(workspace.name)
            results.append(workspace)
    return LookupResult(
        ip_address=ip_address,
        resolved_host_names=tuple(host_names),
        workspaces=tuple(results),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find Perforce workspaces whose Host matches the machine name resolved from an IP address."
    )
    parser.add_argument("ip_address", help="IPv4 or IPv6 address to look up")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="write lookup diagnostics to stderr when no workspaces are returned",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        ipaddress.ip_address(args.ip_address)
    except ValueError:
        parser.error(f"invalid IP address: {args.ip_address}")

    try:
        result = lookup_workspaces_by_ip(args.ip_address)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.debug:
        if result.resolved_host_names:
            print(
                f"Resolved host names: {', '.join(result.resolved_host_names)}",
                file=sys.stderr,
            )
        else:
            print(f"No host name could be resolved for {result.ip_address}.", file=sys.stderr)

        if not result.workspaces and result.resolved_host_names:
            print("No Perforce workspace matched the resolved host name(s).", file=sys.stderr)

    for workspace in result.workspaces:
        print(workspace.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
