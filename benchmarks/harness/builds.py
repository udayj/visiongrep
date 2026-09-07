"""One controlled build path for local archives and cloud source bundles."""

import hashlib
import json
import os
import platform
import shutil
import sys
import tarfile
import tomllib
from pathlib import Path

from .storage import ROOT, command, digest, read_json, write_json


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_environment(cargo_home):
    # Allow transport and tool discovery, but no inherited compiler/profile overrides.
    allowed = (
        "HOME",
        "PATH",
        "TMPDIR",
        "TMP",
        "TEMP",
        "RUSTUP_HOME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    )
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env.update(CARGO_HOME=str(cargo_home), LC_ALL="C", TZ="UTC")
    return env


def reject_cargo_configs(source, cargo_home):
    # Cargo merges ancestor/global configs. Disallow them until we explicitly support
    # their linker, runner, rustflags and source-replacement semantics.
    directories = [cargo_home] + [p / ".cargo" for p in (source, *source.parents)]
    for directory in directories:
        for name in ("config", "config.toml"):
            if (directory / name).exists():
                raise ValueError(
                    f"benchmark builds do not support Cargo config: {directory / name}"
                )


def prepare(source, cargo_home, invoke):
    env = build_environment(cargo_home)
    reject_cargo_configs(source, cargo_home)
    toolchain = tomllib.loads((source / "rust-toolchain.toml").read_text())["toolchain"]
    channel = toolchain["channel"]
    if not channel or not channel[0].isdigit():
        raise ValueError("benchmark builds require a pinned Rust toolchain")
    rust_tools = {}
    for name in ("rustc", "cargo"):
        path = invoke(
            ["rustup", "which", "--toolchain", channel, name], cwd=source, env=env
        ).strip()
        rust_tools[name] = Path(path)
    env["RUSTC"] = str(rust_tools["rustc"])
    rustc = invoke([env["RUSTC"], "-vV"], cwd=source, env=env).strip()
    target = next(
        line.removeprefix("host: ")
        for line in rustc.splitlines()
        if line.startswith("host: ")
    )
    native = {}
    for name in ("cc", "c++", "ar"):
        found = shutil.which(name, path=env.get("PATH"))
        if not found:
            raise ValueError(f"required build tool missing: {name}")
        native[name] = str(Path(found).resolve())
    linker = invoke([native["cc"], "-print-prog-name=ld"], cwd=source, env=env).strip()
    linker_path = shutil.which(linker, path=env.get("PATH"))
    if not linker_path:
        raise ValueError("cannot identify the native linker")
    native["ld"] = str(Path(linker_path).resolve())
    env.update(CC=native["cc"], CXX=native["c++"], AR=native["ar"])
    env["CARGO_TARGET_" + target.upper().replace("-", "_") + "_LINKER"] = native["cc"]
    # Merge stderr because Apple's ld emits its version there.
    probe = "import subprocess,sys; p=subprocess.run(sys.argv[1:],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True); print(p.stdout,end=''); sys.exit(p.returncode)"
    versions = {
        "rustc": rustc,
        "cargo": invoke([str(rust_tools["cargo"]), "-vV"], cwd=source, env=env).strip(),
        "cc": invoke([native["cc"], "--version"], cwd=source, env=env).strip(),
        "c++": invoke([native["c++"], "--version"], cwd=source, env=env).strip(),
        "ld": invoke(
            [
                sys.executable,
                "-c",
                probe,
                native["ld"],
                "-v" if platform.system() == "Darwin" else "--version",
            ],
            cwd=source,
            env=env,
        ).strip(),
    }
    # Rust may select its bundled LLD instead of the system ld on Linux. Include
    # that executable too, while retaining the native linker used by C build steps.
    sysroot = Path(
        invoke([env["RUSTC"], "--print", "sysroot"], cwd=source, env=env).strip()
    )
    bundled_linker = sysroot / "lib/rustlib" / target / "bin/rust-lld"
    if bundled_linker.is_file():
        native["rust-lld"] = str(bundled_linker)
        versions["rust-lld"] = invoke(
            [str(bundled_linker), "-flavor", "gnu", "--version"], cwd=source, env=env
        ).strip()
    sdk = None
    if platform.system() == "Darwin":
        env["SDKROOT"] = invoke(
            ["xcrun", "--show-sdk-path"], cwd=source, env=env
        ).strip()
        sdk = invoke(["xcrun", "--show-sdk-version"], cwd=source, env=env).strip()
    manifest = tomllib.loads((source / "Cargo.toml").read_text())
    settings = {
        "schema": 1,
        "toolchain": toolchain,
        "versions": versions,
        "tool_sha256": {
            name: digest(Path(path)) for name, path in (rust_tools | native).items()
        },
        "target": target,
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "sdk_version": sdk,
        "profiles": manifest.get("profile", {}),
        "features": {
            "selection": "default",
            "definitions": manifest.get("features", {}),
        },
        "command": ["build", "--release", "--locked", "--target", target],
        "environment_policy": "allowlisted-v1; explicit rustc, CC, CXX, AR and target linker; no Cargo config",
    }
    return env, rust_tools["cargo"], settings


def build(revision, cache, invoke, source=None):
    source_record = None
    extracted = False
    if source is None:
        sha = command(
            ["git", "rev-parse", "--verify", revision + "^{commit}"], cwd=ROOT
        ).strip()
        source = cache / "sources" / sha
        source_record = cache / "sources" / (sha + ".json")
        if not source.exists():
            source.mkdir(parents=True)
            archive = source / "source.tar"
            command(
                ["git", "archive", "--format=tar", "--output", str(archive), sha],
                cwd=ROOT,
            )
            with tarfile.open(archive) as archive_file:
                archive_file.extractall(source, filter="data")
            archive.unlink()
            extracted = True
    else:
        sha = revision
    # Source trees contain only git-archive inputs. Include their contents so an
    # accidental edit cannot silently reuse a binary from the original extraction.
    source_hash = fingerprint(
        {
            str(p.relative_to(source)): digest(p)
            for p in sorted(source.rglob("*"))
            if p.is_file()
        }
    )
    if source_record is not None:
        if extracted:
            write_json(source_record, {"sha256": source_hash})
        elif (
            not source_record.exists()
            or read_json(source_record)["sha256"] != source_hash
        ):
            raise ValueError(f"cached source is incomplete or changed: {source}")
    env, cargo, settings = prepare(source, cache / "cargo-home", invoke)
    identity = {
        "commit": sha,
        "source_sha256": source_hash,
        "build_environment": settings,
    }
    destination = cache / "binaries" / fingerprint(identity)
    binary = destination / "target" / settings["target"] / "release/visiongrep"
    record_path = destination / "build.json"
    if record_path.exists():
        record = read_json(record_path)
        if (
            record["identity"] != identity
            or not binary.is_file()
            or digest(binary) != record["sha256"]
        ):
            raise ValueError(f"cached build is corrupt: {destination}")
        return binary, sha, settings
    destination.mkdir(parents=True, exist_ok=True)
    env.update(
        VISIONGREP_BUILD_COMMIT=sha, CARGO_TARGET_DIR=str(destination / "target")
    )
    invoke([str(cargo), *settings["command"]], cwd=source, env=env)
    write_json(record_path, {"identity": identity, "sha256": digest(binary)})
    return binary, sha, settings
