#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/.build"
APP_NAME="Fusion ComfyUI"
APP_BUNDLE="$BUILD_DIR/$APP_NAME.app"
CONFIGURATION="${CONFIGURATION:-release}"
VERSION="0.2.2"
BUILD_NUM=$(date +%Y%m%d%H%M)

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
step()  { echo -e "${CYAN}━━━ $1 ━━━${NC}"; }

build_app() {
    step "swift build -c $CONFIGURATION"
    (cd "$PROJECT_DIR" && swift build -c "$CONFIGURATION" 2>&1 | tail -5)
    info "SPM build done"
}

package_app() {
    step "package $APP_NAME.app"
    local app_dir="$APP_BUNDLE/Contents"
    mkdir -p "$app_dir/MacOS" "$app_dir/Resources"

    local binary_path
    binary_path=$(cd "$PROJECT_DIR" && swift build -c "$CONFIGURATION" --show-bin-path 2>/dev/null || echo "")
    if [ -z "$binary_path" ] || [ ! -f "$binary_path/FusionComfyUI" ]; then
        error "binary not found at $binary_path/FusionComfyUI"
        return 1
    fi
    cp "$binary_path/FusionComfyUI" "$app_dir/MacOS/"
    info "copied binary"

    cat > "$app_dir/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>FusionComfyUI</string>
    <key>CFBundleIdentifier</key>
    <string>com.fusion-mlx.comfyui</string>
    <key>CFBundleName</key>
    <string>Fusion ComfyUI</string>
    <key>CFBundleVersion</key>
    <string>$BUILD_NUM</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>LSArchitecturePriority</key>
    <array><string>arm64</string></array>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSAppTransportSecurity</key>
    <dict>
        <key>NSAllowsLocalNetworking</key>
        <true/>
        <key>NSExceptionDomains</key>
        <dict>
            <key>127.0.0.1</key>
            <dict>
                <key>NSExceptionAllowsInsecureHTTPLoads</key>
                <true/>
            </dict>
            <key>localhost</key>
            <dict>
                <key>NSExceptionAllowsInsecureHTTPLoads</key>
                <true/>
            </dict>
        </dict>
    </dict>
    <key>NSNetworkUsageDescription</key>
    <string>Fusion ComfyUI connects to the local ComfyUI server on 127.0.0.1:8189 and downloads models from hf-mirror.com.</string>
</dict>
</plist>
PLIST
    info "generated Info.plist"
    info "app bundle ready: $APP_BUNDLE"
}

main() {
    local action="${1:-all}"
    case "$action" in
        app)
            build_app
            ;;
        package)
            build_app
            package_app
            ;;
        all)
            build_app
            package_app
            info "done: $APP_BUNDLE"
            ;;
        clean)
            rm -rf "$BUILD_DIR" 2>/dev/null || true
            (cd "$PROJECT_DIR" && swift package clean 2>/dev/null) || true
            info "cleaned"
            ;;
        *)
            echo "Usage: $0 {all|app|package|clean}"
            echo "  CONFIGURATION=debug|release (default release)"
            exit 1
            ;;
    esac
}

main "$@"
