#!/usr/bin/env python3
"""
Resolve IP addresses for Feishu wiki/upload URLs.

Examples:
  python WinScript/ip_resolve.py https://qv2cxb5my2y.feishu.cn/wiki/HiUQwaFKciBUAHkqsnycUXM2nie
  python WinScript/ip_resolve.py --format list https://qv2cxb5my2y.feishu.cn/wiki/HiUQwaFKciBUAHkqsnycUXM2nie
  python WinScript/ip_resolve.py --input-file feishu-upload-urls.json --format list
  python WinScript/ip_resolve.py --fetch https://qv2cxb5my2y.feishu.cn/wiki/HiUQwaFKciBUAHkqsnycUXM2nie

The old feishu-upload-url-sniffer.js records browser-only upload URLs.  This
script can resolve a normal Feishu URL directly, and it can also consume the
JSON/text copied from that sniffer to resolve the actual upload hosts.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import socket
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


URL_PREFIXES = ("http://", "https://")
DEFAULT_TIMEOUT_SECONDS = 8
DEFAULT_FETCH_LIMIT_BYTES = 2 * 1024 * 1024

DOH_ENDPOINTS = {
    "alidns": "https://dns.alidns.com/resolve",
    "cloudflare": "https://cloudflare-dns.com/dns-query",
    "google": "https://dns.google/resolve",
}

RELATED_SUFFIXES = (
    ".feishu.cn",
    ".larksuite.com",
    ".larkoffice.com",
    ".bytecdn.cn",
    ".byteimg.com",
    ".bytedance.com",
    ".bytedance.net",
    ".volces.com",
    ".ivolces.com",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract hostnames from Feishu URLs/sniffer output and resolve their IP addresses."
    )
    parser.add_argument(
        "items",
        nargs="*",
        help="URL, hostname, or text file path. If omitted, stdin is read when data is piped in.",
    )
    parser.add_argument(
        "-i",
        "--input-file",
        action="append",
        default=[],
        help="Read URLs/hosts from a file, including JSON copied by __feishuUploadUrls.copy().",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch input URLs and scan returned text for Feishu-related hosts. Login-only upload URLs may still require sniffer JSON.",
    )
    parser.add_argument(
        "--all-fetched-hosts",
        action="store_true",
        help="When --fetch is used, keep every hostname found in fetched pages instead of only Feishu-related domains.",
    )
    parser.add_argument(
        "--dns",
        choices=["system", "alidns", "cloudflare", "google", "all"],
        default="system",
        help="Resolver to use. 'all' combines system DNS with public DoH resolvers.",
    )
    parser.add_argument(
        "--ipv6",
        action="store_true",
        help="Also resolve AAAA records. By default only IPv4 A records are printed.",
    )
    parser.add_argument(
        "--format",
        choices=["hosts", "list", "json"],
        default="hosts",
        help="Output format. 'list' prints unique IPs only, one per line.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Network timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS}.",
    )
    parser.add_argument(
        "--fetch-limit",
        type=int,
        default=DEFAULT_FETCH_LIMIT_BYTES,
        help=f"Maximum bytes to read from each fetched URL. Default: {DEFAULT_FETCH_LIMIT_BYTES}.",
    )
    return parser.parse_args()


def ordered_add(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def normalize_host(value: str) -> str | None:
    raw = value.strip().strip("[](){}<>,;'\"")
    if not raw:
        return None

    if raw.startswith("//"):
        raw = "https:" + raw

    parsed = urllib.parse.urlparse(raw)
    if not parsed.hostname and "/" in raw and not raw.startswith(URL_PREFIXES):
        parsed = urllib.parse.urlparse("//" + raw)

    host = parsed.hostname if parsed.hostname else raw
    host = host.strip().strip("[](){}<>,;'\"").rstrip(".").lower()
    if not host:
        return None

    if "/" in host or "\\" in host or "@" in host:
        return None

    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        pass

    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return None

    labels = host.split(".")
    if len(labels) < 2:
        return None
    if any(not label or len(label) > 63 for label in labels):
        return None
    if any(label.startswith("-") or label.endswith("-") for label in labels):
        return None
    allowed_chars = set("abcdefghijklmnopqrstuvwxyz0123456789-.")
    if any(ch not in allowed_chars for ch in host):
        return None
    return host


def is_related_host(host: str) -> bool:
    return any(host == suffix[1:] or host.endswith(suffix) for suffix in RELATED_SUFFIXES)


def extract_urls(text: str) -> list[str]:
    urls: list[str] = []
    for prefix in URL_PREFIXES:
        start = 0
        while True:
            idx = text.find(prefix, start)
            if idx < 0:
                break
            end = idx
            while end < len(text) and text[end] not in " \t\r\n'\"<>[]{}":
                end += 1
            ordered_add(urls, text[idx:end].rstrip("),;"))
            start = end
    return urls


def walk_json_strings(value: Any, strings: list[str]) -> None:
    if isinstance(value, str):
        ordered_add(strings, value)
    elif isinstance(value, list):
        for item in value:
            walk_json_strings(item, strings)
    elif isinstance(value, dict):
        for item in value.values():
            walk_json_strings(item, strings)


def extract_hosts_from_text(text: str, *, related_only: bool) -> tuple[list[str], list[str]]:
    hosts: list[str] = []
    urls: list[str] = []
    candidates: list[str] = []

    try:
        parsed_json = json.loads(text)
    except json.JSONDecodeError:
        parsed_json = None
    if parsed_json is not None:
        walk_json_strings(parsed_json, candidates)

    candidates.extend(extract_urls(text))
    candidates.extend(text.replace(",", " ").replace(";", " ").split())

    for candidate in candidates:
        for url in extract_urls(candidate):
            ordered_add(urls, url)
            host = normalize_host(url)
            if host and (not related_only or is_related_host(host)):
                ordered_add(hosts, host)

        host = normalize_host(candidate)
        if host and (not related_only or is_related_host(host)):
            ordered_add(hosts, host)

    return hosts, urls


def read_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8-sig", errors="replace")


def fetch_text(url: str, *, timeout: float, limit: int) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "ip_resolve.py/1.0 (+https://feishu.cn)",
            "Accept": "text/html,application/json,text/plain,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(limit + 1)
        if len(data) > limit:
            data = data[:limit]
        charset = resp.headers.get_content_charset() or "utf-8"
        return data.decode(charset, errors="replace")


def resolve_system(host: str, *, include_ipv6: bool) -> list[str]:
    try:
        return [str(ipaddress.ip_address(host))]
    except ValueError:
        pass

    families = [socket.AF_INET]
    if include_ipv6:
        families.append(socket.AF_INET6)

    ips: list[str] = []
    for family in families:
        try:
            results = socket.getaddrinfo(host, None, family, socket.SOCK_STREAM)
        except socket.gaierror:
            continue
        for result in results:
            ordered_add(ips, result[4][0])
    return sort_ips(ips)


def doh_query(resolver: str, name: str, record_type: str, *, timeout: float) -> dict[str, Any]:
    params = urllib.parse.urlencode({"name": name, "type": record_type})
    url = f"{DOH_ENDPOINTS[resolver]}?{params}"
    req = urllib.request.Request(url, headers={"Accept": "application/dns-json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def resolve_doh(host: str, resolver: str, *, include_ipv6: bool, timeout: float) -> list[str]:
    try:
        return [str(ipaddress.ip_address(host))]
    except ValueError:
        pass

    record_types = ["A"]
    if include_ipv6:
        record_types.append("AAAA")

    ips: list[str] = []
    queue = [host]
    seen: set[str] = set()

    while queue:
        name = queue.pop(0).rstrip(".").lower()
        if name in seen:
            continue
        seen.add(name)

        for record_type in record_types:
            try:
                payload = doh_query(resolver, name, record_type, timeout=timeout)
            except (OSError, json.JSONDecodeError):
                continue

            for answer in payload.get("Answer", []):
                answer_type = answer.get("type")
                data = str(answer.get("data", "")).rstrip(".")
                if answer_type == 5 and data:
                    ordered_add(queue, data.lower())
                    continue
                if (record_type == "A" and answer_type == 1) or (
                    record_type == "AAAA" and answer_type == 28
                ):
                    try:
                        ordered_add(ips, str(ipaddress.ip_address(data)))
                    except ValueError:
                        continue

    return sort_ips(ips)


def sort_ips(ips: list[str]) -> list[str]:
    return sorted(ips, key=lambda ip: (ipaddress.ip_address(ip).version, int(ipaddress.ip_address(ip))))


def collect_inputs(args: argparse.Namespace) -> tuple[list[str], list[str], list[str]]:
    hosts: list[str] = []
    urls: list[str] = []
    notes: list[str] = []

    sources: list[tuple[str, str]] = []
    for item in args.items:
        path = Path(item)
        if path.is_file():
            sources.append((str(path), read_text_file(str(path))))
        else:
            sources.append(("<arg>", item))

    for input_file in args.input_file:
        sources.append((input_file, read_text_file(input_file)))

    if not sources and not sys.stdin.isatty():
        sources.append(("<stdin>", sys.stdin.read()))

    for source_name, text in sources:
        source_hosts, source_urls = extract_hosts_from_text(text, related_only=False)
        for host in source_hosts:
            ordered_add(hosts, host)
        for url in source_urls:
            ordered_add(urls, url)

        if not source_hosts and source_name != "<arg>":
            notes.append(f"No hostnames found in {source_name}.")

    if args.fetch:
        fetch_related_only = not args.all_fetched_hosts
        for url in list(urls):
            try:
                fetched = fetch_text(url, timeout=args.timeout, limit=args.fetch_limit)
            except OSError as exc:
                notes.append(f"Fetch failed for {url}: {exc}")
                continue
            fetched_hosts, fetched_urls = extract_hosts_from_text(fetched, related_only=fetch_related_only)
            for host in fetched_hosts:
                ordered_add(hosts, host)
            for fetched_url in fetched_urls:
                ordered_add(urls, fetched_url)

    return hosts, urls, notes


def resolve_hosts(hosts: list[str], args: argparse.Namespace) -> dict[str, list[str]]:
    resolved: dict[str, list[str]] = {}
    resolvers = ["system"] if args.dns != "all" else ["system", "alidns", "cloudflare", "google"]
    if args.dns != "all":
        resolvers = [args.dns]

    for host in hosts:
        ips: list[str] = []
        for resolver in resolvers:
            if resolver == "system":
                resolver_ips = resolve_system(host, include_ipv6=args.ipv6)
            else:
                resolver_ips = resolve_doh(
                    host,
                    resolver,
                    include_ipv6=args.ipv6,
                    timeout=args.timeout,
                )
            for ip in resolver_ips:
                ordered_add(ips, ip)
        resolved[host] = sort_ips(ips)
    return resolved


def unique_ips(resolved: dict[str, list[str]]) -> list[str]:
    ips: list[str] = []
    for host_ips in resolved.values():
        for ip in host_ips:
            ordered_add(ips, ip)
    return sort_ips(ips)


def print_output(resolved: dict[str, list[str]], notes: list[str], fmt: str) -> None:
    ips = unique_ips(resolved)

    if fmt == "json":
        print(json.dumps({"hosts": resolved, "ips": ips, "notes": notes}, ensure_ascii=False, indent=2))
        return

    if fmt == "list":
        for ip in ips:
            print(ip)
        for note in notes:
            print(f"# {note}", file=sys.stderr)
        return

    for host, host_ips in resolved.items():
        print(host)
        if host_ips:
            for ip in host_ips:
                print(f"  {ip}")
        else:
            print("  <unresolved>")

    if len(resolved) > 1:
        print()
        print("unique IPs")
        for ip in ips:
            print(f"  {ip}")

    for note in notes:
        print(f"note: {note}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    socket.setdefaulttimeout(args.timeout)

    hosts, _urls, notes = collect_inputs(args)
    if not hosts:
        print("No hostnames found. Pass a Feishu URL, hostname, text file, or sniffer JSON.", file=sys.stderr)
        return 2

    resolved = resolve_hosts(hosts, args)
    print_output(resolved, notes, args.format)

    return 0 if any(resolved.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
