import Foundation
import Logging

private let logger = Logger(label: "FusionComfyUI.ServerManager")

final class ServerManager: ObservableObject {
    @Published var isRunning = false
    @Published var port: Int = 11443
    @Published var host: String = "127.0.0.1"

    private var process: Process?

    var baseURL: URL {
        URL(string: "http://\(host):\(port)")!
    }

    func start() {
        guard !isRunning else { return }
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        proc.arguments = [
            "fusion-comfyui", "serve",
            "--host", host,
            "--port", String(port),
        ]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        do {
            try proc.run()
            process = proc
            isRunning = true
            logger.info("server started PID=\(proc.processIdentifier) on \(host):\(port)")
        } catch {
            logger.error("failed to start server: \(error)")
        }
    }

    func stop() {
        guard isRunning, let proc = process else { return }
        proc.terminate()
        isRunning = false
        process = nil
        logger.info("server stopped")
    }

    deinit {
        stop()
    }
}
