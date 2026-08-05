import Foundation
import os.log

private let logger = Logger(subsystem: "com.fusion.comfyui", category: "ModelManager")

struct ModelInfo: Identifiable, Hashable {
    let id: String
    let name: String
    let type: String
    let sizeBytes: Int64?

    var displayName: String {
        "\(name) (\(type))"
    }
}

@MainActor
final class ModelManager: ObservableObject {
    @Published var availableModels: [ModelInfo] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var isDownloading = false
    @Published var downloadLog: String = ""

    private let baseURL: URL
    private let modelsCacheURL: URL = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".fusion-mlx/models")
    }()

    init(serverURL: URL) {
        self.baseURL = serverURL
    }

    func refreshModels() async {
        isLoading = true
        errorMessage = nil
        let url = baseURL.appendingPathComponent("object_info")
        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
                errorMessage = "server returned \( (response as? HTTPURLResponse)?.statusCode ?? -1)"
                isLoading = false
                return
            }
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                errorMessage = "invalid JSON response"
                isLoading = false
                return
            }
            var models: [ModelInfo] = []
            if let loader = json["FusionModelLoader"] as? [String: Any],
               let input = loader["input"] as? [String: Any],
               let required = input["required"] as? [String: Any],
               let modelField = required["model_name"] as? [Any],
               modelField.count > 0,
               let names = modelField[0] as? [String] {
                for (idx, name) in names.enumerated() {
                    let type = name.contains("wan") || name.contains("skyreels") || name.contains("ltx") ? "video" : "image"
                    models.append(ModelInfo(id: "\(idx)", name: name, type: type, sizeBytes: nil))
                }
            }
            availableModels = models
            logger.info("discovered \(models.count) models via object_info")
        } catch {
            errorMessage = error.localizedDescription
            logger.error("model refresh failed: \(error.localizedDescription)")
        }
        isLoading = false
    }

    func listLocalModels() -> [String] {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: modelsCacheURL.path) else {
            logger.info("local models cache not present at \(self.modelsCacheURL.path)")
            return []
        }
        let dirs = entries.filter { name in
            var isDir: ObjCBool = false
            let exists = fm.fileExists(atPath: modelsCacheURL.appendingPathComponent(name).path, isDirectory: &isDir)
            return exists && isDir.boolValue
        }.sorted()
        logger.info("local models cache: \(dirs.count) entries")
        return dirs
    }

    func pullModel(repoId: String) async {
        guard !repoId.isEmpty else { return }
        guard !isDownloading else {
            logger.warning("pull already in progress, ignoring \(repoId)")
            return
        }
        isDownloading = true
        downloadLog = "Pulling \(repoId) via fusion-mlx (HF_MIRROR=hf-mirror.com) ...\n"
        logger.info("pulling model \(repoId) via fusion-mlx pull")

        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            DispatchQueue.global(qos: .userInitiated).async {
                let proc = Process()
                proc.executableURL = URL(fileURLWithPath: "/bin/bash")
                proc.arguments = ["-lc", "fusion-mlx pull \(shellQuote(repoId))"]
                var env = ProcessInfo.processInfo.environment
                env["HF_MIRROR"] = "https://hf-mirror.com"
                proc.environment = env

                let pipe = Pipe()
                proc.standardOutput = pipe
                proc.standardError = pipe
                let outHandle = pipe.fileHandleForReading
                outHandle.readabilityHandler = { handle in
                    let chunk = handle.availableData
                    guard !chunk.isEmpty, let line = String(data: chunk, encoding: .utf8) else { return }
                    Task { @MainActor in
                        self.downloadLog += line
                    }
                }

                proc.terminationHandler = { p in
                    outHandle.readabilityHandler = nil
                    let tail = String(data: outHandle.readDataToEndOfFile(), encoding: .utf8) ?? ""
                    let code = p.terminationStatus
                    Task { @MainActor in
                        if !tail.isEmpty { self.downloadLog += tail }
                        if code == 0 {
                            self.downloadLog += "\n[fusion-mlx pull] done.\n"
                            logger.info("pull succeeded for \(repoId)")
                            await self.refreshModels()
                        } else {
                            self.downloadLog += "\n[fusion-mlx pull] failed exit=\(code)\n"
                            logger.error("pull failed for \(repoId) exit=\(code)")
                        }
                        self.isDownloading = false
                        continuation.resume()
                    }
                }

                do {
                    try proc.run()
                } catch {
                    Task { @MainActor in
                        self.downloadLog += "failed to launch fusion-mlx: \(error.localizedDescription)\n"
                        self.isDownloading = false
                        continuation.resume()
                    }
                }
            }
        }
    }
}

private func shellQuote(_ s: String) -> String {
    "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'"
}
