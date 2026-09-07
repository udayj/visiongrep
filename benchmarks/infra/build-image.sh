#!/bin/bash
# EC2 user data for a one-time Ubuntu 24.04 image build. The launcher must also
# schedule termination after 60 minutes and use shutdown behavior "stop" so a
# successful stopped builder can be captured as an AMI before it is terminated.
set -euo pipefail
exec > >(tee -a /var/log/visiongrep-image-build.log) 2>&1

finish() {
    result=$?
    trap - EXIT
    if command -v aws >/dev/null && [[ -n "${BENCH_AMI_REPORT_URI:-}" ]]; then
        export AWS_MAX_ATTEMPTS=2
        aws s3 cp /var/log/visiongrep-image-build.log "$BENCH_AMI_REPORT_URI/build.log" \
            --cli-connect-timeout 10 --cli-read-timeout 20 --only-show-errors || true
        if [[ "$result" == 0 ]]; then
            aws s3 cp /usr/local/share/visiongrep-image.json "$BENCH_AMI_REPORT_URI/ready.json" \
                --cli-connect-timeout 10 --cli-read-timeout 20 --only-show-errors || result=1
        fi
    fi
    echo "VISIONGREP_IMAGE_BUILD_EXIT=$result"
    shutdown -h now
    exit "$result"
}
trap finish EXIT
systemd-run --unit=visiongrep-image-deadline --on-active=45m /sbin/shutdown -h now
: "${BENCH_AMI_REPORT_URI:?launcher must supply the S3 report destination}"
export AWS_DEFAULT_REGION=us-east-1

timeout --signal=TERM --kill-after=30s 35m bash -euo pipefail <<'INSTALL'
export DEBIAN_FRONTEND=noninteractive
apt-get -o DPkg::Lock::Timeout=120 update
apt-get -o DPkg::Lock::Timeout=120 install -y --no-install-recommends \
    build-essential pkg-config libssl-dev python3 python3-venv git curl unzip ca-certificates

workspace=$(mktemp -d)
trap 'rm -rf "$workspace"' EXIT
cd "$workspace"

# Official HTTPS downloads, with fixed versions and checksums verified before execution.
curl --fail --location --retry 2 --connect-timeout 20 --max-time 300 \
    https://awscli.amazonaws.com/awscli-exe-linux-x86_64-2.35.2.zip -o awscli.zip
echo '757a038404f95dbb716a830565602078fb96b7281135920c91f6e8dc5c578a4f  awscli.zip' | sha256sum -c -
unzip -q awscli.zip
./aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli

curl --fail --location --retry 2 --connect-timeout 20 --max-time 300 \
    https://static.rust-lang.org/rustup/archive/1.28.2/x86_64-unknown-linux-gnu/rustup-init -o rustup-init
echo '20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c  rustup-init' | sha256sum -c -
chmod +x rustup-init
export RUSTUP_HOME=/opt/rustup
export CARGO_HOME=/opt/cargo
./rustup-init -y --no-modify-path --profile default --default-toolchain 1.98.1

useradd --create-home --shell /bin/bash bench
ln -s /opt/rustup /home/bench/.rustup
ln -s /opt/cargo /home/bench/.cargo
chown -R bench:bench /opt/rustup /opt/cargo /home/bench
for tool in rustup cargo rustc rustdoc rustfmt; do
    ln -s "/opt/cargo/bin/$tool" "/usr/local/bin/$tool"
done

# Verify the same user and default environment that the benchmark worker will use.
unset RUSTUP_HOME CARGO_HOME
runuser -u bench -- rustup which --toolchain 1.98.1 rustc
runuser -u bench -- rustc +1.98.1 -vV
runuser -u bench -- cargo +1.98.1 -vV
runuser -u bench -- aws --version
runuser -u bench -- python3 -c 'import platform, sqlite3, tomllib; assert platform.python_version_tuple()[:2] == ("3", "12")'

# Package updates happen when making a new image, not while measuring a candidate.
systemctl disable --now apt-daily.timer apt-daily-upgrade.timer
systemctl mask apt-daily.service apt-daily-upgrade.service
apt-get clean

python3 - <<'MANIFEST'
import datetime, json, platform, subprocess
from pathlib import Path
def output(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
record = {
    "status": "ready",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "python": platform.python_version(),
    "rustc": output(["runuser", "-u", "bench", "--", "rustc", "+1.98.1", "-vV"]),
    "cargo": output(["runuser", "-u", "bench", "--", "cargo", "+1.98.1", "-vV"]),
    "aws": output(["aws", "--version"]),
    "cc": output(["cc", "--version"]),
    "ld": output(["ld", "--version"]),
    "packages": output(["dpkg-query", "-W"]),
}
Path("/usr/local/share/visiongrep-image.json").write_text(json.dumps(record, indent=2) + "\n")
MANIFEST
INSTALL

# New instances must run their own cloud-init user data and get a fresh machine ID.
cloud-init clean --logs --machine-id
