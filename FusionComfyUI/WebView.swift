import SwiftUI
import WebKit

struct WebViewWrapper: NSViewRepresentable {
    let url: URL
    @ObservedObject var serverManager: ServerManager

    func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.preferences.isElementFullscreenEnabled = true
        let webView = WKWebView(frame: .zero, configuration: config)
        return webView
    }

    func updateNSView(_ nsView: WKWebView, context: Context) {
        if serverManager.isRunning {
            let request = URLRequest(url: url)
            if nsView.url != url {
                nsView.load(request)
            }
        }
    }
}
