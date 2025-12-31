//
//  CameraScreen.swift
//  LeonardCNNtest
//
//  Created by 陳暄暢 on 24/12/2025.
//

import SwiftUI
import AVFoundation
import Combine
import CoreVideo

// MARK: - Data Model
struct Detection: Identifiable, Equatable {
    let id : String
    /// 归一化坐标：x/y/width/height 均在 0~1（相对预览视图）
    var rect: CGRect
    var label: String
    var score: Float
}

func createEmptyPixelBuffer(width: Int, height: Int,pixelFormat: OSType = kCVPixelFormatType_32BGRA) -> CVPixelBuffer {
    let attrs: [CFString: Any] = [
        kCVPixelBufferCGImageCompatibilityKey: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        kCVPixelBufferMetalCompatibilityKey: true
    ]

    var pb: CVPixelBuffer?
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        width,
        height,
        pixelFormat,
        attrs as CFDictionary,
        &pb
    )
    precondition(status == kCVReturnSuccess && pb != nil, "CVPixelBufferCreate failed: \(status)")

    let pixelBuffer = pb!
    CVPixelBufferLockBaseAddress(pixelBuffer, [])
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

    // 全部清零：BGRA 下就是黑色且 Alpha=0/或取决于你后续解释方式
    if let base = CVPixelBufferGetBaseAddress(pixelBuffer) {
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let totalBytes = bytesPerRow * height
        memset(base, 0, totalBytes)
    }
    return pixelBuffer
}

// MARK: - Camera Session Manager
final class CameraSessionManager: NSObject {
    let session = AVCaptureSession()

    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "camera.video.queue")

    // 串行控制 session（避免 start/stop 交错造成闪烁、状态抖动）
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")

    private var isConfigured = false
    
    private var videoDevice: AVCaptureDevice?
    private(set) var minZoomFactor: CGFloat = 1.0
    private(set) var maxZoomFactor: CGFloat = 1.0
    @Published var zoomFactor: CGFloat = 1.0

    /// 每帧回调（在 videoQueue 上触发）
    /// 注意：这里传出的 pixelBuffer 是“已 retain”的；接收方必须在用完后 CVPixelBufferRelease
    var onFrame: ((CVPixelBuffer) -> Void)?
    
    func rampZoom(_ factor: CGFloat, rate: Float = 30.0) {
        sessionQueue.async { [weak self] in
            guard let self, let device = self.videoDevice else { return }

            let clamped = max(self.minZoomFactor, min(factor, self.maxZoomFactor))

            do {
                try device.lockForConfiguration()
                device.ramp(toVideoZoomFactor: clamped, withRate: rate)
                device.unlockForConfiguration()

                Task { @MainActor in
                    self.zoomFactor = clamped
                }
            } catch {
                print("Zoom lock error: \(error)")
            }
        }
    }

    func setZoom(_ factor: CGFloat) {
        sessionQueue.async { [weak self] in
            guard let self, let device = self.videoDevice else { return }

            let clamped = max(self.minZoomFactor, min(factor, self.maxZoomFactor))

            do {
                try device.lockForConfiguration()
                device.videoZoomFactor = clamped
                device.unlockForConfiguration()

                Task { @MainActor in
                    self.zoomFactor = clamped
                }
            } catch {
                print("Zoom lock error: \(error)")
            }
        }
    }

    func cancelZoomRamp() {
        sessionQueue.async { [weak self] in
            guard let device = self?.videoDevice else { return }
            do {
                try device.lockForConfiguration()
                device.cancelVideoZoomRamp()
                device.unlockForConfiguration()
                self?.zoomFactor = device.videoZoomFactor
            } catch { }
        }
    }

    func configureIfNeeded() throws {
        try sessionQueue.sync {
            guard !isConfigured else { return }
            isConfigured = true

            session.beginConfiguration()
            session.sessionPreset = .high

            let device =
                AVCaptureDevice.default(.builtInTripleCamera, for: .video, position: .back) ??
                AVCaptureDevice.default(.builtInDualCamera, for: .video, position: .back) ??
                AVCaptureDevice.default(.builtInDualWideCamera, for: .video, position: .back) ??
                AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)

            guard let device else {
                session.commitConfiguration()
                throw NSError(domain: "Camera", code: -1, userInfo: [NSLocalizedDescriptionKey: "未找到后摄"])
            }
            
            self.videoDevice = device
            self.minZoomFactor = device.minAvailableVideoZoomFactor
            self.maxZoomFactor = min(device.maxAvailableVideoZoomFactor, 12.0) // 你可以调整上限
            self.zoomFactor = device.videoZoomFactor


            let input = try AVCaptureDeviceInput(device: device)
            guard session.canAddInput(input) else {
                session.commitConfiguration()
                throw NSError(domain: "Camera", code: -2, userInfo: [NSLocalizedDescriptionKey: "无法添加相机输入"])
            }
            session.addInput(input)
            
            let defaultZoom: CGFloat = 2.0
            let clamped = max(device.minAvailableVideoZoomFactor,
                              min(defaultZoom, device.maxAvailableVideoZoomFactor))

            do {
                try device.lockForConfiguration()
                device.videoZoomFactor = clamped
                device.unlockForConfiguration()
            } catch {
                print("set default zoom failed: \(error)")
            }
            
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: 640,
                kCVPixelBufferHeightKey as String: 360
            ]
            videoOutput.setSampleBufferDelegate(self, queue: videoQueue)

            guard session.canAddOutput(videoOutput) else {
                session.commitConfiguration()
                throw NSError(domain: "Camera", code: -3, userInfo: [NSLocalizedDescriptionKey: "无法添加相机输出"])
            }
            session.addOutput(videoOutput)

            if let conn = videoOutput.connection(with: .video) {
                if conn.isVideoRotationAngleSupported(90) {
                    conn.videoRotationAngle = 90
                }
            }

            session.commitConfiguration()
            print("deviceType=\(device.deviceType), isVirtual=\(device.isVirtualDevice), min=\(device.minAvailableVideoZoomFactor), max=\(device.maxAvailableVideoZoomFactor)")
            if device.isVirtualDevice {
                print("switchOver=\(device.virtualDeviceSwitchOverVideoZoomFactors)")
            }
        }
    }

    func startRunning() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            guard !self.session.isRunning else { return }
            self.session.startRunning()
        }
    }

    func stopRunning() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            guard self.session.isRunning else { return }
            self.session.stopRunning()
        }
    }
}

extension CameraSessionManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        onFrame?(pixelBuffer)
    }
}



// MARK: - Preview View (UIView -> AVCaptureVideoPreviewLayer)
final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }

    func attach(session: AVCaptureSession) {
        previewLayer.session = session
        previewLayer.videoGravity = .resizeAspect
        if let conn = previewLayer.connection {
            if #available(iOS 17.0, *) {
                if conn.isVideoRotationAngleSupported(90) {
                    conn.videoRotationAngle = 90
                }
            } else {
                // iOS 16 及以下（仅在需要兼容时才保留）
                if conn.isVideoOrientationSupported {
                    conn.videoOrientation = .portrait
                }
            }
        }

    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let v = PreviewView()
        v.attach(session: session)
        return v
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}
}

import SwiftUI
import AVFoundation
import Combine

final class CameraScreenViewModel: ObservableObject {
//    private var thermalObs: NSObjectProtocol?
//
//    func startThermalMonitor() {
//        thermalObs = NotificationCenter.default.addObserver(
//            forName: ProcessInfo.thermalStateDidChangeNotification,
//            object: nil,
//            queue: .main
//        ) { _ in
//            let s = ProcessInfo.processInfo.thermalState
//            print("[THERM] thermalState=\(s.rawValue) (0=nominal 1=fair 2=serious 3=critical)")
//        }
//    }
//
//    deinit {
//        if let thermalObs { NotificationCenter.default.removeObserver(thermalObs) }
//    }
    @Published var isModelReady: Bool = false
    // MARK: - FPS Meter
    private final class FPSMeter {
        private var lastT: CFTimeInterval = CACurrentMediaTime()
        private var n: Int = 0

        func reset() {
            lastT = CACurrentMediaTime()
            n = 0
        }

        /// 每次 tick 一帧；每 >=0.5s 输出一次平滑 FPS
        func tick(now: CFTimeInterval = CACurrentMediaTime()) -> Double? {
            n += 1
            let dt = now - lastT
            guard dt >= 0.5 else { return nil }
            let fps = Double(n) / dt
            lastT = now
            n = 0
            return fps
        }
    }

    // MARK: - Public UI State
    enum PermissionState { case unknown, granted, denied }

    @Published var permission: PermissionState = .unknown
    @Published var isRunning: Bool = false

    @Published var detections: [Detection] = []
    @Published var demoOverlay: Bool = true

    // ✅ 保留 FPS（去掉 frameCount）
    @Published var camFPS: Double = 0
    @Published var infFPS: Double = 0

    // 用于 overlay 的尺寸/旋转
    @Published var frameSize: CGSize = .zero
    @Published var rotate90: Bool = false

    // MARK: - Core
    let camera = CameraSessionManager()
    private var detector: MNNDetector?

    private let inferenceQueue = DispatchQueue(label: "mnn.inference.queue", qos: .userInitiated)

    // ✅ “门闩”：同一时间只跑一个推理；拿不到就丢帧（避免队列堆积）
    private let infGate = DispatchSemaphore(value: 1)

    // 推理节流：你要 20fps 左右就用 1/20；想尽可能快就设 0
    private let minInferInterval: CFTimeInterval = 1.0 / 20.0
    private var lastInferT: CFTimeInterval = 0

    // stop/start 后丢弃旧结果
    private var runToken = UUID()

    // FPS meters
    private let camMeter = FPSMeter()
    private let infMeter = FPSMeter()

    // 线程安全读写少量状态（避免 Swift 6 并发/数据竞争问题）
    private let stateLock = NSLock()
    private var runningFlag: Bool = false
    private var overlayFlag: Bool = true
    private var tokenFlag: UUID = UUID()

    // MARK: - Permission & Setup
    func requestPermissionAndSetup() {
        let status = AVCaptureDevice.authorizationStatus(for: .video)

        switch status {
        case .authorized:
            permission = .granted
            setupCamera()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.permission = granted ? .granted : .denied
                    if granted { self.setupCamera() }
                }
            }
        default:
            permission = .denied
        
        }
        loadModelAsync()
    }
    private func loadModelAsync() {
        // 放到后台线程去初始化，绝对不要卡主线程
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            print("[ViewModel] Start loading model...")
            let det = MNNDetector() // 耗时操作在这里发生
            
            // 还可以顺便做一次“预热”推理，消除第一次识别的卡顿
            let heatUp = createEmptyPixelBuffer(width: 640, height: 640)
            det.detect(with: heatUp)
            
            DispatchQueue.main.async {
                print("[ViewModel] Model loaded!")
                self?.detector = det
                self?.isModelReady = true
            }
        }
    }

    private func setupCamera() {
        do {
            try camera.configureIfNeeded()
        } catch {
            permission = .denied
            print("Camera configure failed:", error)
            return
        }

        // ✅ 关键：不要每帧都 Task @MainActor
        // 相机回调（videoQueue 上）直接进入轻量 handleFrameOnVideoQueue
        camera.onFrame = { [weak self] pixelBuffer in
            self?.handleFrameOnVideoQueue(pixelBuffer)
        }
    }

    // MARK: - Run Control
    func start() {
        guard permission == .granted else { return }

        let newToken = UUID()
        stateLock.lock()
        runningFlag = true
        overlayFlag = demoOverlay
        tokenFlag = newToken
        stateLock.unlock()

        runToken = newToken
        isRunning = true
        camMeter.reset()
        infMeter.reset()
        camFPS = 0
        infFPS = 0

        camera.startRunning()
    }

    func stop() {
        let newToken = UUID()
        stateLock.lock()
        runningFlag = false
        tokenFlag = newToken
        stateLock.unlock()

        runToken = newToken
        isRunning = false
        camera.stopRunning()
        detections = []
    }

    func setDemoOverlay(_ on: Bool) {
        demoOverlay = on
        stateLock.lock()
        overlayFlag = on
        stateLock.unlock()

        if !on { detections = [] }
    }

    // MARK: - Frame Handling (runs on camera.video.queue)
    private func handleFrameOnVideoQueue(_ pixelBuffer: CVPixelBuffer) {
        // 1) cam FPS：这里统计才是真正的 camera output 速率
        if let fps = camMeter.tick() {
            DispatchQueue.main.async { [weak self] in
                self?.camFPS = fps
            }
        }

        // 2) 尺寸/旋转信息（变化很少，变化时才回主线程）
        let w = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let h = CGFloat(CVPixelBufferGetHeight(pixelBuffer))

        if frameSize.width != w || frameSize.height != h {
            let rot = (w > h)
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.frameSize = CGSize(width: w, height: h)
                self.rotate90 = rot
            }
        }

        // 3) 读取状态快照
        stateLock.lock()
        let running = runningFlag
        let enabled = overlayFlag
        let token = tokenFlag
        stateLock.unlock()

        guard running, enabled else { return }

        // 4) 推理节流（目标 20fps 左右）
        let now = CACurrentMediaTime()
        if now - lastInferT < minInferInterval { return }
        lastInferT = now

        // 5) “门闩”：拿不到就丢帧（防止队列堆积 -> cam FPS 被拖死）
        if infGate.wait(timeout: .now()) != .success {
            return
        }

        guard let detector = self.detector, isModelReady else {
            return
        }

        inferenceQueue.async { [pixelBuffer, token, detector] in
            defer { self.infGate.signal() }

            // ✅ autoreleasepool：防止 ObjC 桥接对象在后台线程堆积导致“越跑越慢”
            let mapped: [Detection] = autoreleasepool {
                let raw = (detector.detect(with: pixelBuffer) as? [[AnyHashable: Any]]) ?? []

                // key 统一成 String
                let dicts: [[String: Any]] = raw.map { anyDict in
                    var out: [String: Any] = [:]
                    out.reserveCapacity(anyDict.count)
                    for (k, v) in anyDict {
                        if let ks = k as? String { out[ks] = v }
                        else if let kns = k as? NSString { out[kns as String] = v }
                        else { out[String(describing: k)] = v }
                    }
                    return out
                }

                var result: [Detection] = []
                result.reserveCapacity(dicts.count)

                for (i, d) in dicts.enumerated() {
                    guard
                        let x = d["x"] as? NSNumber,
                        let y = d["y"] as? NSNumber,
                        let w = d["w"] as? NSNumber,
                        let h = d["h"] as? NSNumber,
                        let label = d["label"] as? String,
                        let score = d["score"] as? NSNumber
                    else { continue }

                    result.append(
                        Detection(
                            id: "\(label)#\(i)",  // 避免重复 id
                            rect: CGRect(x: x.doubleValue, y: y.doubleValue,
                                         width: w.doubleValue, height: h.doubleValue),
                            label: label,
                            score: score.floatValue
                        )
                    )
                }
                return result
            }

            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                // stop/start 后丢弃旧结果
                guard self.isRunning, self.runToken == token else { return }

                if self.detections != mapped {
                    self.detections = mapped
                }

                // inf FPS：以“推理结果真正落到 UI 的频率”为准
                if let fps = self.infMeter.tick() {
                    self.infFPS = fps
                }
            }
        }
    }
}


// MARK: - Overlay (SwiftUI 画框)
struct DetectionOverlay: View {
    let detections: [Detection]
    let frameSize: CGSize
    let rotate90: Bool

    var body: some View {
        GeometryReader { geo in
            let viewW = geo.size.width
            let viewH = geo.size.height

            // 原始 pixelBuffer 尺寸
            let srcW0 = max(frameSize.width, 1)
            let srcH0 = max(frameSize.height, 1)

            // 如果预览相对 buffer 旋转 90°，绘制坐标系要交换宽高
            let srcW = rotate90 ? srcH0 : srcW0
            let srcH = rotate90 ? srcW0 : srcH0

            // resizeAspect 对应：AspectFit（完整显示，不裁切）
            let scale = min(viewW / srcW, viewH / srcH)
            let dispW = srcW * scale
            let dispH = srcH * scale
            let offX = (viewW - dispW) / 2
            let offY = (viewH - dispH) / 2

            ForEach(Array(detections.enumerated()), id: \.offset) { _, det in
                // det.rect：相对原始 pixelBuffer（srcW0/srcH0）的归一化坐标
                let r0 = CGRect(
                    x: det.rect.origin.x * srcW0,
                    y: det.rect.origin.y * srcH0,
                    width: det.rect.size.width * srcW0,
                    height: det.rect.size.height * srcH0
                )

                // 把 r0 从“buffer 坐标”转到“预览坐标”
                // 仍按你当前逻辑：90° CW
                let r = rotate90
                ? CGRect(
                    x: srcH0 - (r0.minY + r0.height),
                    y: r0.minX,
                    width: r0.height,
                    height: r0.width
                  )
                : r0

                // aspectFit 映射到 view 坐标（注意这里是 +offX/+offY，不是减）
                let viewRect = CGRect(
                    x: r.minX * scale + offX,
                    y: r.minY * scale + offY,
                    width: r.width * scale,
                    height: r.height * scale
                )

                ZStack(alignment: .topLeading) {
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(.green, lineWidth: 2)
                        .frame(width: viewRect.width, height: viewRect.height)
                        .position(x: viewRect.midX, y: viewRect.midY)

                    Text("\(det.label) \(Int(det.score * 100))%")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(.black.opacity(0.55))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .position(x: viewRect.minX + 80, y: max(viewRect.minY - 14, 14))
                }
            }
        }
        .allowsHitTesting(false)
    }
}



// MARK: - Screen (UI)
struct CameraScreen: View {
    @StateObject private var vm = CameraScreenViewModel()
    @State private var baseZoom: CGFloat = 1.5
    
    @State private var uiBasePhysicalZoom: CGFloat = 0          // UI 的“1.0x”对应的物理倍率（默认 2.x）
    @State private var showZoomHUD: Bool = false
    @State private var hideZoomWorkItem: DispatchWorkItem?
    @State private var didIgnoreFirstZoomChange: Bool = false   // 避免启动默认设2x时HUD闪一下
    @State private var isPinching: Bool = false                 // 防止 pinch 过程中 baseZoom 被外部 onChange 打断

    private func clamp(_ x: CGFloat, _ lo: CGFloat, _ hi: CGFloat) -> CGFloat {
        max(lo, min(x, hi))
    }

    private func revealZoomHUD() {
        withAnimation(.easeInOut(duration: 0.15)) {
            showZoomHUD = true
        }

        hideZoomWorkItem?.cancel()

        let workItem = DispatchWorkItem {
            withAnimation(.easeInOut(duration: 0.15)) {
                showZoomHUD = false
            }
        }

        hideZoomWorkItem = workItem
        DispatchQueue.main.asyncAfter(deadline: .now() + 3.0, execute: workItem)
    }

    var body: some View {
        let base = (uiBasePhysicalZoom > 0) ? uiBasePhysicalZoom : 2.0
        let displayZoom = vm.camera.zoomFactor / base

        ZStack {
            // Camera preview
            CameraPreview(session: vm.camera.session)
                .gesture(
                    MagnificationGesture()
                        .onChanged { value in
                            isPinching = true
                            vm.camera.rampZoom(baseZoom * value)   // 你的原逻辑：物理倍率缩放
                            revealZoomHUD()
                        }
                        .onEnded { value in
                            isPinching = false
                            let target = baseZoom * value
                            baseZoom = max(vm.camera.minZoomFactor, min(target, vm.camera.maxZoomFactor))
                            vm.camera.cancelZoomRamp()
                            vm.camera.setZoom(baseZoom)
//                            vm.camera.zoomFactor = AVCaptureDevice.
                        }
                )
                .simultaneousGesture(
                    TapGesture(count: 2).onEnded {
                        // 双击回“UI 1.0x”= 物理回到 uiBasePhysicalZoom（默认 2.x）
                        let base = (uiBasePhysicalZoom > 0) ? uiBasePhysicalZoom : 2.0
                        vm.camera.rampZoom(base)
                        baseZoom = base
                        revealZoomHUD()
                    }
                )
                .ignoresSafeArea()
            
            // Overlay boxes
            DetectionOverlay(detections: vm.detections, frameSize: vm.frameSize, rotate90: vm.rotate90)
                .ignoresSafeArea()
            
            // Top bar
            VStack(spacing: 10) {
                HStack {
                    Text(vm.isRunning ? "camera.Running" : "camera.Stopped")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(.black.opacity(0.55))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    
                    Text("Det: \(vm.detections.count)")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(.black.opacity(0.55))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    
                    Text(verbatim: String(format: NSLocalizedString("camera_fps", comment: "camera FPS label"), locale: .current, vm.camFPS))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(.black.opacity(0.55))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    
                    Text(verbatim: String(format: NSLocalizedString("inference_fps", comment: "inference FPS label"), locale: .current, vm.infFPS))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(.black.opacity(0.55))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    
                    Spacer()
                    
                    Toggle("Demo", isOn: Binding(
                        get: { vm.demoOverlay },
                        set: { vm.setDemoOverlay($0) }
                    ))
                    .labelsHidden()
                    .tint(.green)
                }
                .padding(.top, 14)
                .padding(.horizontal, 14)
                
                // Zoom HUD：仅在 zoomFactor 变化/交互时显示，正常隐藏
                if showZoomHUD, uiBasePhysicalZoom > 0 {
                    Text(String(format: "%.1fx", displayZoom))
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 10)
                        .background(.black.opacity(0.55))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                }
                
                Spacer()
                
                if showZoomHUD, uiBasePhysicalZoom > 0 {
                    HStack(spacing: 10) {
                        ForEach([0.5, 1.0, 3.0], id: \.self) { d in
                            let selected = abs(displayZoom - d) < 0.12

                            Button {
                                let targetPhysical = clamp(d * base, vm.camera.minZoomFactor, vm.camera.maxZoomFactor)
                                vm.camera.rampZoom(targetPhysical)
                                baseZoom = targetPhysical
                                revealZoomHUD()
                            } label: {
                                Text(String(format: "%.1fx", d))   // 你想要的 0.5 1.0 3.0
                                    .font(.system(size: 14, weight: .semibold))
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 10)
                                    .background(selected ? .white.opacity(0.22) : .black.opacity(0.35))
                                    .clipShape(Capsule())
                                    .foregroundStyle(.white)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.horizontal, 14)
                    .padding(.bottom, 10) // 距离开始按钮的间距，可调
                    .transition(.opacity)
                }
                
                // Bottom bar
                HStack(spacing: 12) {
                    Button {
                        vm.isRunning ? vm.stop() : vm.start()
                    } label: {
                        Text(vm.isRunning ? "camera.Stop" : "camera.Start")
                            .font(.system(size: 16, weight: .semibold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 14)
                            .background(.orange.opacity(0.65))
                            .foregroundStyle(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 14))
                    }
                }
                .padding(.horizontal, 14)
                .padding(.bottom, 18)
            }
            
            // Permission denied overlay
            if vm.permission == .denied {
                VStack(spacing: 10) {
                    Text("camera.unusable")
                        .font(.system(size: 18, weight: .bold))
                        .foregroundStyle(.white)
                    Text("allow.camera")
                        .font(.system(size: 14))
                        .foregroundStyle(.white.opacity(0.9))
                }
                .padding(18)
                .background(.black.opacity(0.75))
                .clipShape(RoundedRectangle(cornerRadius: 16))
                .padding(.horizontal, 24)
            }
        }
        .onAppear {
            vm.requestPermissionAndSetup()
        }
        .onChange(of: vm.camera.zoomFactor) { _, newZoom in
            // 1) UI 基准：只在第一次拿到“默认 2.x”时锁定
            //    你默认设为2x，所以这里通常会在 newZoom≈2.x 时锁定
            if uiBasePhysicalZoom <= 0, newZoom >= 1.5 {
                uiBasePhysicalZoom = newZoom
                baseZoom = newZoom
            }
            
            // 2) 非 pinch 期间，让 baseZoom 跟随真实倍率（避免下次捏合从旧值跳）
            if !isPinching {
                baseZoom = newZoom
            }
            
            // 3) 只在“真实倍率变化”时显示 HUD，且忽略首次初始化变化
            if !didIgnoreFirstZoomChange {
                didIgnoreFirstZoomChange = true
                return
            }
            revealZoomHUD()
        }
        .onDisappear {
            hideZoomWorkItem?.cancel()
            hideZoomWorkItem = nil
        }
    }
}

