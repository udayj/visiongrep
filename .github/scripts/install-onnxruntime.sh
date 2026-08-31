#!/usr/bin/env bash

set -euo pipefail

readonly version="1.24.2"
readonly destination="${1:?usage: install-onnxruntime.sh DESTINATION}"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)
    readonly package="onnxruntime-linux-x64-${version}"
    readonly checksum="43725474ba5663642e17684717946693850e2005efbd724ac72da278fead25e6"
    readonly library_variable="LD_LIBRARY_PATH"
    ;;
  Darwin-arm64)
    readonly package="onnxruntime-osx-arm64-${version}"
    readonly checksum="0af4fa503e8ea285245b47ee42d0a7461b8156a81270857da0c1d4ecf858abde"
    readonly library_variable="DYLD_LIBRARY_PATH"
    ;;
  *)
    echo "unsupported ONNX Runtime host: $(uname -s)-$(uname -m)" >&2
    exit 1
    ;;
esac

readonly archive="${destination}/${package}.tgz"
readonly extracted="${destination}/${package}"
readonly url="https://github.com/microsoft/onnxruntime/releases/download/v${version}/${package}.tgz"

mkdir -p "${destination}"
if [[ ! -f "${archive}" ]]; then
  curl --fail --location --retry 5 --retry-all-errors --output "${archive}.partial" "${url}"
  mv "${archive}.partial" "${archive}"
fi

if command -v sha256sum >/dev/null 2>&1; then
  readonly actual_checksum="$(sha256sum "${archive}" | awk '{print $1}')"
else
  readonly actual_checksum="$(shasum -a 256 "${archive}" | awk '{print $1}')"
fi
if [[ "${actual_checksum}" != "${checksum}" ]]; then
  echo "ONNX Runtime checksum mismatch: expected ${checksum}, got ${actual_checksum}" >&2
  exit 1
fi

if [[ ! -d "${extracted}" ]]; then
  tar -xzf "${archive}" -C "${destination}"
fi

readonly library_directory="${extracted}/lib"
{
  printf 'ORT_LIB_LOCATION=%s\n' "${library_directory}"
  printf 'ORT_PREFER_DYNAMIC_LINK=1\n'
  printf '%s=%s\n' "${library_variable}" "${library_directory}"
} >> "${GITHUB_ENV:?GITHUB_ENV must be set by GitHub Actions}"
