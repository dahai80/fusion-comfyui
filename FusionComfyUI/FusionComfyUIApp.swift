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

    init(serverManager: ServerManager) {
        self.serverManager = serverManager
        _modelManager = StateObject(wrappedValue: ModelManager(serverURL: serverManager.baseURL))
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Circle()
                    .fill(serverManager.isRunning ? Color.green : Color.red)
                    .frame(width: 8, height: 8)
                Text(serverManager.isRunning ? "Connected" : "Disconnected")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Spacer()
                Button("Refresh Models") {
                    Task { await modelManager.refreshModels() }
                }
                .disabled(!serverManager.isRunning)
            }
            .padding(6)
            .background(Color(nsColor: .controlBackgroundColor))

            if serverManager.isRunning {
                WebViewWrapper(url: serverManager.baseURL, serverManager: serverManager)
                    .frame(minWidth: 800, minHeight: 600)
            } else {
                VStack(spacing: 12) {
                    Text("Server not running")
                        .font(.headline)
                    Button("Start Server") {
                        serverManager.start()
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .onAppear {
            serverManager.start()
            Task { await modelManager.refreshModels() }
        }
    }
}
