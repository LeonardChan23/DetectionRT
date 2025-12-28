import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            CameraScreen()
                .tabItem { Label("Camera", systemImage: "camera") }

            ImageDetectScreen()
                .tabItem { Label("Photo", systemImage: "photo") }
        }
    }
}

#Preview {
    ContentView()
}
