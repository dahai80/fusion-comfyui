import Foundation
import Logging

private let logger = Logger(label: "FusionComfyUI.ModelManager")

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

    private let baseURL: URL

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
            logger.info("discovered \(models.count) models")
        } catch {
            errorMessage = error.localizedDescription
            logger.error("model refresh failed: \(error)")
        }
        isLoading = false
    }
}
