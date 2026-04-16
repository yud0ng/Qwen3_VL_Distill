#!/usr/bin/env bash
# 将 rclone 安装到仓库旁 tools/（无需改系统目录）；安装后把 tools/rclone 加入 PATH 或直接用绝对路径。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/tools"
ZIP="${DEST}/rclone-current-linux-amd64.zip"
ARCH_DIR="${DEST}/rclone-current-linux-amd64"
BIN="${DEST}/rclone"

mkdir -p "${DEST}"
echo "Downloading rclone to ${ZIP} ..."
curl -fL --retry 3 --retry-delay 2 \
  -o "${ZIP}" \
  "https://downloads.rclone.org/rclone-current-linux-amd64.zip"
rm -rf "${ARCH_DIR}"
unzip -q -o "${ZIP}" -d "${DEST}"
# 解压目录名带版本号，取匹配到的唯一目录
EXTRACTED="$(find "${DEST}" -maxdepth 1 -type d -name 'rclone-*-linux-amd64' | head -1)"
if [[ -z "${EXTRACTED}" ]]; then
  echo "error: expected rclone-*-linux-amd64 under ${DEST}" >&2
  exit 1
fi
mv -f "${EXTRACTED}" "${ARCH_DIR}"
cp -f "${ARCH_DIR}/rclone" "${BIN}"
chmod +x "${BIN}"
"${BIN}" version
echo
echo "Done. Binary: ${BIN}"
echo "Use: export PATH=\"${DEST}:\$PATH\""
echo "Or:  ${BIN} config"
