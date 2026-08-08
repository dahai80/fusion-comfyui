#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/.build"
APP_NAME="Fusion ComfyUI"
APP_BUNDLE="$BUILD_DIR/$APP_NAME.app"
CONFIGURATION="${CONFIGURATION:-release}"
VERSION="0.2.4"
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

    if [ -f "$SCRIPT_DIR/AppIcon.icns" ]; then
        cp "$SCRIPT_DIR/AppIcon.icns" "$app_dir/Resources/AppIcon.icns"
        info "copied app icon"
    else
        info "no AppIcon.icns found (run 'python Scripts/make_icon.py' to generate)"
    fi

    cat > "$app_dir/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>FusionComfyUI</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
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
    <string>Fusion ComfyUI connects to the local ComfyUI server on 127.0.0.1:11445 and downloads models from hf-mirror.com.</string>
</dict>
</plist>
PLIST
    info "generated Info.plist"
    info "app bundle ready: $APP_BUNDLE"
}

build_dmg() {
    step "build DMG for $APP_NAME"
    if [ ! -d "$APP_BUNDLE" ]; then
        error "app bundle missing: $APP_BUNDLE — run 'package' first"
        return 1
    fi
    local dmg_name="$APP_NAME-$VERSION-arm64.dmg"
    local dmg_path="$BUILD_DIR/$dmg_name"
    local staging="$BUILD_DIR/dmg-staging"
    rm -rf "$staging" "$dmg_path" 2>/dev/null || true
    mkdir -p "$staging"
    cp -R "$APP_BUNDLE" "$staging/"
    ln -s /Applications "$staging/Applications"
    info "staged app + /Applications link"
    local vol_name="${APP_NAME// /_}_$VERSION"
    hdiutil create -volname "$vol_name" -srcfolder "$staging" \
        -ov -format UDBZ "$dmg_path" >/dev/null 2>&1 || {
        error "hdiutil failed"
        rm -rf "$staging"
        return 1
    }
    rm -rf "$staging"
    info "DMG ready: $dmg_path ($(du -h "$dmg_path" | cut -f1))"
}

main() {
    local action="${1:-all}"
    case "$action" in
        app)
            build_app
            ;;
        icon)
            (cd "$PROJECT_DIR" && python "$SCRIPT_DIR/make_icon.py")
            ;;
        package)
            build_app
            package_app
            ;;
        dmg)
            build_app
            package_app
            build_dmg
            info "done: $BUILD_DIR/$APP_NAME-$VERSION-arm64.dmg"
            ;;
        all)
            build_app
            package_app
            build_dmg
            info "done: $APP_BUNDLE + DMG"
            ;;
        clean)
            rm -rf "$BUILD_DIR" 2>/dev/null || true
            (cd "$PROJECT_DIR" && swift package clean 2>/dev/null) || true
            info "cleaned"
            ;;
        *)
            echo "Usage: $0 {all|app|package|dmg|icon|clean}"
            echo "  CONFIGURATION=debug|release (default release)"
            exit 1
            ;;
    esac
}

main "$@"
