import Foundation
import os.log
import SwiftUI

private let logger = Logger(subsystem: "com.fusion.comfyui", category: "SetupManager")

enum SetupPhase: Equatable {
    case idle
    case checking
    case creatingVenv
    case installing
    case done
    case failed
}

@MainActor
final class SetupManager: ObservableObject {
    @Published var phase: SetupPhase = .idle
    @Published var progress: Double = 0.0
    @Published var logText: String = ""
    @Published var venvPath: String

    private let venvRoot: String

    init() {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        venvRoot = ProcessInfo.processInfo.environment["FUSION_COMFYUI_VENV_ROOT"]
            ?? "\(home)/.fusion-comfyui/venv"
        venvPath = venvRoot
        if FileManager.default.fileExists(atPath: "\(venvRoot)/bin/python") {
            phase = .done
            progress = 1.0
        }
    }

    var isReady: Bool { phase == .done }

    func resetLog() {
        logText = ""
    }

    func appendLog(_ line: String) {
        logText.append(line + "\n")
        logger.info("setup: \(line, privacy: .public)")
    }

    func setup() {
        guard phase != .checking, phase != .creatingVenv, phase != .installing else { return }
        phase = .checking
        progress = 0.05
        appendLog("checking for Python >=3.11 ...")

        Task {
            let py = await findPython()
            guard let python = py else {
                await MainActor.run {
                    phase = .failed
                    appendLog("ERROR: no Python >=3.11 found. Install via: brew install python@3.12")
                    appendLog("or set FUSION_COMFYUI_VENV_ROOT to an existing venv.")
                }
                return
            }
            await MainActor.run { appendLog("found python: \(python)") }

            await MainActor.run {
                phase = .creatingVenv
                progress = 0.15
                appendLog("creating venv at \(venvRoot) ...")
            }
            let created = await runShell("\(python) -m venv \"\(venvRoot)\"")
            if !created {
                await MainActor.run {
                    phase = .failed
                    appendLog("ERROR: venv creation failed")
                }
                return
            }
            let venvPy = "\(venvRoot)/bin/python"
            await MainActor.run {
                appendLog("venv ready: \(venvPy)")
                phase = .installing
                progress = 0.25
                appendLog("upgrading pip ...")
            }
            _ = await runShell("\(venvPy) -m pip install --upgrade pip -i https://hf-mirror.com/pypi/simple")

            await MainActor.run { appendLog("installing ComfyUI requirements (this takes a few minutes) ...") }
            let projectDir = ProcessInfo.processInfo.environment["FUSION_COMFYUI_PROJECT_DIR"]
                ?? "/Users/dahai/fusion/fusion-comfyui"
            let reqPath = "\(projectDir)/ComfyUI/requirements.txt"
            _ = await runShell("\(venvPy) -m pip install -r \"\(reqPath)\" -i https://hf-mirror.com/pypi/simple")
            await MainActor.run { progress = 0.75; appendLog("installing fusion-mlx ...") }
            _ = await runShell("\(venvPy) -m pip install fusion-mlx -i https://hf-mirror.com/pypi/simple")

            await MainActor.run {
                progress = 1.0
                phase = .done
                appendLog("setup complete. venv at \(venvRoot)")
            }
        }
    }

    private func findPython() async -> String? {
        let candidates = [
            "/opt/homebrew/bin/python3.14",
            "/opt/homebrew/bin/python3.12",
            "/opt/homebrew/bin/python3.11",
            "/usr/local/bin/python3.12",
            "/usr/local/bin/python3.11",
        ]
        for path in candidates {
            let ok = await runShell("\(path) -c \"import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)\"")
            if ok { return path }
        }
        return nil
    }

    private func runShell(_ command: String) async -> Bool {
        await withCheckedContinuation { continuation in
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/bin/bash")
            proc.arguments = ["-lc", command]
            let pipe = Pipe()
            proc.standardOutput = pipe
            proc.standardError = pipe
            do {
                try proc.run()
                let data = try pipe.fileHandleForReading.readToEnd() ?? Data()
                if let out = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines), !out.isEmpty {
                    Task { @MainActor in self.appendLog(out) }
                }
                proc.waitUntilExit()
                continuation.resume(returning: proc.terminationStatus == 0)
            } catch {
                Task { @MainActor in self.appendLog("shell error: \(error.localizedDescription)") }
                continuation.resume(returning: false)
            }
        }
    }
}

struct SetupView: View {
    @ObservedObject var setupManager: SetupManager
    var onDone: () -> Void

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "wrench.and.screwdriver")
                .font(.system(size: 48))
                .foregroundColor(.accentColor)
            Text("Setting up Fusion ComfyUI")
                .font(.title2.bold())
            Text("Installing Python dependencies on first launch. This happens once.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            if setupManager.phase == .installing || setupManager.phase == .creatingVenv {
                ProgressView(value: setupManager.progress)
                    .progressViewStyle(.linear)
                    .frame(maxWidth: 360)
            } else if setupManager.phase == .failed {
                Image(systemName: "exclamationmark.triangle")
                    .foregroundColor(.red)
                    .font(.title3)
            }

            ScrollView {
                Text(setupManager.logText)
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .frame(maxWidth: 480, maxHeight: 200)
            .border(Color.secondary.opacity(0.2))

            HStack {
                if setupManager.phase == .failed {
                    Button("Retry") { setupManager.setup() }
                }
                if setupManager.phase == .done {
                    Button("Continue") { onDone() }
                        .buttonStyle(.borderedProminent)
                }
                if setupManager.phase == .idle {
                    Button("Start Setup") { setupManager.setup() }
                        .buttonStyle(.borderedProminent)
                }
            }
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            if setupManager.phase == .idle {
                setupManager.setup()
            }
        }
    }
}
