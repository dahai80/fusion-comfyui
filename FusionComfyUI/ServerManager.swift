import Foundation
import os.log

private let logger = Logger(subsystem: "com.fusion.comfyui", category: "ServerManager")

enum ServerStatus: Equatable {
    case stopped
    case starting
    case running
    case failed
}

final class ServerManager: ObservableObject {
    @Published var status: ServerStatus = .stopped
    @Published var lastError: String?

    let host: String = "127.0.0.1"
    let port: Int = 8189

    var isRunning: Bool { status == .running }
    var baseURL: URL { URL(string: "http://\(host):\(port)")! }
    var healthURL: URL { URL(string: "http://\(host):\(port)/system_stats")! }

    var startShPath: String {
        ProcessInfo.processInfo.environment["FUSION_COMFYUI_START_SH"]
            ?? "/Users/dahai/fusion/fusion-comfyui/start.sh"
    }

    private var process: Process?
    private var healthTask: Task<Void, Never>?

    func start() {
        guard status != .running, status != .starting else { return }
        status = .starting
        lastError = nil
        logger.info("starting server via \(self.startShPath) start")

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/bin/bash")
        proc.arguments = [startShPath, "start"]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        do {
            try proc.run()
            process = proc
            logger.info("start.sh launched PID=\(proc.processIdentifier)")
            beginHealthPoll()
        } catch {
            status = .failed
            lastError = "failed to launch start.sh: \(error.localizedDescription)"
            logger.error("failed to launch start.sh: \(error.localizedDescription)")
        }
    }

    private func beginHealthPoll() {
        let timeout = ProcessInfo.processInfo.environment["FUSION_HEALTH_TIMEOUT"]
            .flatMap(Int.init) ?? 30
        let deadline = Date(timeIntervalSinceNow: TimeInterval(timeout))
        healthTask?.cancel()
        healthTask = Task { [weak self] in
            guard let self = self else { return }
            while !Task.isCancelled {
                if await self.probeHealth() {
                    await MainActor.run {
                        self.status = .running
                        logger.info("server healthy at \(self.healthURL.absoluteString)")
                    }
                    return
                }
                if Date() > deadline {
                    await MainActor.run {
                        self.status = .failed
                        self.lastError = "health probe timed out after \(timeout)s"
                        logger.error("health probe timed out after \(timeout)s")
                    }
                    return
                }
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        }
    }

    func probeHealth() async -> Bool {
        var req = URLRequest(url: healthURL)
        req.timeoutInterval = 2.0
        do {
            let (_, resp) = try await URLSession.shared.data(for: req)
            if let http = resp as? HTTPURLResponse, (200..<400).contains(http.statusCode) {
                return true
            }
            return false
        } catch {
            return false
        }
    }

    func stop() {
        healthTask?.cancel()
        healthTask = nil
        logger.info("stopping server via start.sh stop")
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/bin/bash")
        proc.arguments = [startShPath, "stop"]
        do {
            try proc.run()
            proc.waitUntilExit()
        } catch {
            logger.error("failed to run start.sh stop: \(error.localizedDescription)")
        }
        process?.terminate()
        process = nil
        status = .stopped
    }

    deinit {
        healthTask?.cancel()
        process?.terminate()
    }
}
