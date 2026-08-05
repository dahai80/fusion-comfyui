import SwiftUI

@main
struct FusionComfyUIApp: App {
    @StateObject private var serverManager = ServerManager()

    var body: some Scene {
        WindowGroup {
            ContentView(serverManager: serverManager)
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 1280, height: 900)
    }
}

struct ContentView: View {
    @ObservedObject var serverManager: ServerManager
    @StateObject private var modelManager: ModelManager
    @State private var showModelsPanel = false
    @State private var repoInput = ""

    init(serverManager: ServerManager) {
        self.serverManager = serverManager
        _modelManager = StateObject(wrappedValue: ModelManager(serverURL: serverManager.baseURL))
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Circle()
                    .fill(statusColor)
                    .frame(width: 8, height: 8)
                Text(statusText)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Button("Refresh Models") {
                    Task { await modelManager.refreshModels() }
                }
                .disabled(!serverManager.isRunning)
                Button("Models") { showModelsPanel.toggle() }
            }
            .padding(6)
            .background(Color(nsColor: .controlBackgroundColor))

            switch serverManager.status {
            case .running:
                WebViewWrapper(url: serverManager.baseURL, serverManager: serverManager)
                    .frame(minWidth: 800, minHeight: 600)
            case .starting:
                VStack(spacing: 12) {
                    ProgressView().scaleEffect(1.2)
                    Text("Starting ComfyUI server on :8189 ...")
                        .font(.headline)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            case .failed:
                VStack(spacing: 12) {
                    Text("Server failed to start")
                        .font(.headline)
                        .foregroundColor(.red)
                    if let err = serverManager.lastError {
                        Text(err).font(.caption).foregroundColor(.secondary).padding()
                    }
                    Button("Retry") { serverManager.start() }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            case .stopped:
                VStack(spacing: 12) {
                    Text("Server not running")
                        .font(.headline)
                    Button("Start Server") { serverManager.start() }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .sheet(isPresented: $showModelsPanel) {
            ModelsPanel(
                modelManager: modelManager,
                repoInput: $repoInput,
                isPresented: $showModelsPanel
            )
            .frame(minWidth: 520, minHeight: 460)
        }
        .onAppear {
            serverManager.start()
            Task { await modelManager.refreshModels() }
        }
        .onDisappear {
            serverManager.stop()
        }
    }

    private var statusColor: Color {
        switch serverManager.status {
        case .running: return .green
        case .starting: return .yellow
        case .failed: return .red
        case .stopped: return .gray
        }
    }

    private var statusText: String {
        switch serverManager.status {
        case .running: return "Connected :\(serverManager.port)"
        case .starting: return "Starting ..."
        case .failed: return "Failed"
        case .stopped: return "Disconnected"
        }
    }
}

struct ModelsPanel: View {
    @ObservedObject var modelManager: ModelManager
    @Binding var repoInput: String
    @Binding var isPresented: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Models").font(.headline)
            HStack {
                TextField("repo id, e.g. Wan-AI/Wan2.1-T2V-1.3B", text: $repoInput)
                    .textFieldStyle(.roundedBorder)
                Button("Pull") {
                    let repo = repoInput.trimmingCharacters(in: .whitespaces)
                    guard !repo.isEmpty else { return }
                    Task { await modelManager.pullModel(repoId: repo) }
                }
                .disabled(modelManager.isDownloading)
            }
            if modelManager.isDownloading {
                ProgressView().scaleEffect(0.8)
            }
            ScrollView {
                Text(modelManager.downloadLog)
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .frame(maxHeight: 160)
            .border(Color.secondary.opacity(0.2))

            Divider()
            Text("Available models (\(modelManager.availableModels.count))").font(.subheadline)
            List(modelManager.availableModels) { m in
                Text(m.displayName)
            }
            HStack {
                Spacer()
                Button("Close") { isPresented = false }
            }
        }
        .padding()
    }
}
