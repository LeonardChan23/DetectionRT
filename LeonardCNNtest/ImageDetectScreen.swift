//
//  ImageDetectScreen.swift
//  LeonardCNNtest
//
//  Created by 陳暄暢 on 25/12/2025.
//

import SwiftUI
import PhotosUI
import AVFoundation
import Combine
import ImageIO


// effect_roi 对应 demo 里的 object_rect：在 dst×dst 坐标系中，真实内容区域（去掉 padding）
struct EffectROI {
    var x: CGFloat
    var y: CGFloat
    var width: CGFloat
    var height: CGFloat
}

enum ImageUtils {
    static func imagePixelSize(from data: Data) -> CGSize {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? CGFloat,
              let h = props[kCGImagePropertyPixelHeight] as? CGFloat else {
            return .init(width: 1, height: 1)
        }
        return .init(width: w, height: h)
    }

    static func downsample(data: Data, maxPixel: CGFloat) -> UIImage? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        let opts: CFDictionary = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixel,
            kCGImageSourceShouldCacheImmediately: true
        ] as CFDictionary
        guard let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, opts) else { return nil }
        return UIImage(cgImage: cg)
    }

    static func uiImageRaw(from pixelBuffer: CVPixelBuffer) -> UIImage? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let width  = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bpr    = CVPixelBufferGetBytesPerRow(pixelBuffer)

        let cs = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.union(
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)
        )

        // 这里用 Data 拷贝一份，避免 provider 生命周期问题
        let data = Data(bytes: base, count: bpr * height)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }

        guard let cg = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bpr,
            space: cs,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else { return nil }

        return UIImage(cgImage: cg, scale: 1.0, orientation: .up)
    }

    static func normalizedUp(_ image: UIImage, maxSide: CGFloat = 2048) -> UIImage {
        // 既修正 orientation，又避免按原图全尺寸重绘导致内存暴涨
        let srcSize = image.size
        let longest = max(srcSize.width, srcSize.height)
        let scaleDown = min(1.0, maxSide / max(longest, 1))

        let outSize = CGSize(width: floor(srcSize.width * scaleDown),
                             height: floor(srcSize.height * scaleDown))

        if image.imageOrientation == .up, scaleDown == 1.0 {
            return image
        }

        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        format.opaque = true

        let renderer = UIGraphicsImageRenderer(size: outSize, format: format)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: outSize))
        }
    }

    /// 复刻 resize_uniform：输出 dst×dst + effect_roi（roi 在 dst 坐标系里）
    /// - 关键：renderer.scale = 1，确保输出确实是 dst 像素，而不是 dst pt @3x
    static func resizeUniform(_ image: UIImage, dst: Int) -> (ui: UIImage, roi: EffectROI, srcPixelSize: CGSize)? {
        let img = normalizedUp(image)
        guard let cg = img.cgImage else { return nil }

        let srcW = CGFloat(cg.width)
        let srcH = CGFloat(cg.height)

        let dstW = CGFloat(dst)
        let dstH = CGFloat(dst)

        let ratioSrc = srcW / srcH
        let ratioDst = dstW / dstH // =1

        var tmpW: CGFloat = 0
        var tmpH: CGFloat = 0
        var padX: CGFloat = 0
        var padY: CGFloat = 0

        if ratioSrc > ratioDst {
            // 宽图：撑满宽，黑边上下
            tmpW = dstW
            tmpH = floor((dstW / srcW) * srcH)
            padX = 0
            padY = floor((dstH - tmpH) / 2.0)
        } else if ratioSrc < ratioDst {
            // 高图：撑满高，黑边左右
            tmpH = dstH
            tmpW = floor((dstH / srcH) * srcW)
            padY = 0
            padX = floor((dstW - tmpW) / 2.0)
        } else {
            tmpW = dstW
            tmpH = dstH
            padX = 0
            padY = 0
        }

        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        format.opaque = true

        let renderer = UIGraphicsImageRenderer(size: CGSize(width: dstW, height: dstH), format: format)
        let out = renderer.image { ctx in
            UIColor.black.setFill()
            ctx.fill(CGRect(x: 0, y: 0, width: dstW, height: dstH))
            img.draw(in: CGRect(x: padX, y: padY, width: tmpW, height: tmpH))
        }

        let roi = EffectROI(x: padX, y: padY, width: tmpW, height: tmpH)
        return (out, roi, CGSize(width: srcW, height: srcH))
    }

    /// UIImage(dst×dst) -> CVPixelBuffer(BGRA dst×dst)
    static func pixelBufferBGRA(from image: UIImage, dst: Int) -> CVPixelBuffer? {
        let img = normalizedUp(image)
        guard let cg = img.cgImage else { return nil }

        let w = dst, h = dst
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        guard CVPixelBufferCreate(kCFAllocatorDefault, w, h,
                                  kCVPixelFormatType_32BGRA,
                                  attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let pixelBuffer = pb else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let bpr = CVPixelBufferGetBytesPerRow(pixelBuffer)

        let cs = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.union(
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)
        )

        guard let ctx = CGContext(data: base,
                                  width: w, height: h,
                                  bitsPerComponent: 8,
                                  bytesPerRow: bpr,
                                  space: cs,
                                  bitmapInfo: bitmapInfo.rawValue) else { return nil }

        ctx.clear(CGRect(x: 0, y: 0, width: w, height: h))
        ctx.interpolationQuality = .high
        ctx.setBlendMode(.copy)

        // ✅ 不做 flipY
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: CGFloat(w), height: CGFloat(h)))
        return pixelBuffer
    }

    /// 把 dst 坐标系里的 box 映射回原图像素坐标（复刻 demo draw_bboxes 的数学）
    static func mapBoxFromDstToSource(boxDst: CGRect, roi: EffectROI, srcSize: CGSize) -> CGRect {
        let srcW = max(srcSize.width, 1)
        let srcH = max(srcSize.height, 1)

        let dstW = max(roi.width, 1)
        let dstH = max(roi.height, 1)

        let widthRatio  = srcW / dstW
        let heightRatio = srcH / dstH

        let x1 = (boxDst.minX - roi.x) * widthRatio
        let y1 = (boxDst.minY - roi.y) * heightRatio
        let x2 = (boxDst.maxX - roi.x) * widthRatio
        let y2 = (boxDst.maxY - roi.y) * heightRatio

        let cx1 = max(0, min(x1, srcW))
        let cy1 = max(0, min(y1, srcH))
        let cx2 = max(0, min(x2, srcW))
        let cy2 = max(0, min(y2, srcH))

        return CGRect(x: cx1, y: cy1, width: max(0, cx2 - cx1), height: max(0, cy2 - cy1))
    }
}

// 画在原图（scaledToFit）上
struct ImageOverlayFit: View {
    let detections: [(rectNorm: CGRect, label: String, score: Float)]
    let srcPixelSize: CGSize

    var body: some View {
        GeometryReader { geo in
            let viewW = geo.size.width
            let viewH = geo.size.height
            let srcW = max(srcPixelSize.width, 1)
            let srcH = max(srcPixelSize.height, 1)

            let scale = min(viewW / srcW, viewH / srcH)
            let drawnW = srcW * scale
            let drawnH = srcH * scale
            let offX = (viewW - drawnW) * 0.5
            let offY = (viewH - drawnH) * 0.5

            ForEach(Array(detections.enumerated()), id: \.offset) { _, det in
                let r = det.rectNorm
                let x = r.minX * srcW * scale + offX
                let y = r.minY * srcH * scale + offY
                let w = r.width * srcW * scale
                let h = r.height * srcH * scale

                ZStack(alignment: .topLeading) {
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(.green, lineWidth: 2)
                        .frame(width: w, height: h)
                        .position(x: x + w/2, y: y + h/2)

                    Text("\(det.label) \(Int(det.score * 100))%")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(.black.opacity(0.55))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .position(x: x + 90, y: max(y - 14, 14))
                }
            }
        }
        .allowsHitTesting(false)
    }
}

final class ImageDetectVM: ObservableObject {
    private enum BoxNormSpace {
        case dstSquare   // x/y/w/h 是相对 dst×dst（含 padding）
        case roiContent  // x/y/w/h 是相对 roi 内容区（不含 padding）
    }
    
    private var thumbTasks: [UUID: Task<Void, Never>] = [:]
    private var previewTasks: [UUID: Task<Void, Never>] = [:]
    private var detectTasks: [UUID: Task<Void, Never>] = [:]

    private let thumbMaxPixel: CGFloat = 256
    private let previewMaxPixel: CGFloat = 2048

    // ✅ 默认：相对 dst×dst（含 padding）
    private let boxSpace: BoxNormSpace = .dstSquare

    // MARK: - Multi select items

    struct Prepared {
        let ui: UIImage
        let roi: EffectROI
        let srcPixelSize: CGSize
    }

    struct DetectItem: Identifiable {
        let id = UUID()
        let pickerItem: PhotosPickerItem

        // 懒加载缓存
        var thumb: UIImage? = nil        // 例如 256px
        var preview: UIImage? = nil      // 例如 1536~2048px（用于预览/叠框）
        var previewPixelSize: CGSize = .init(width: 1, height: 1)

        var isLoadingThumb = false
        var isLoadingPreview = false

        // 检测结果与状态
        var dets: [(rectNorm: CGRect, label: String, score: Float)] = []
        var status: String = ""

        // 可选：缓存一次预处理结果（dst 输入 & roi 映射信息），重复检测更快
        var prepared: Prepared? = nil
    }

    @Published var items: [DetectItem] = []
    @Published var selectedID: UUID? = nil

    // 当前预览（复用你原来的 UI 结构）
    @Published var status: String = NSLocalizedString("single.status.pick", comment: "pick a image")
    @Published var picked: UIImage? = nil
    @Published var dets: [(rectNorm: CGRect, label: String, score: Float)] = []

    private var roi: EffectROI? = nil
    private var srcPixelSize: CGSize = .init(width: 1, height: 1)

    private let detector = NanoDetSingleDetector()
    private let q = DispatchQueue(label: "single.image.detect.queue", qos: .userInitiated)

    // MARK: - Batch control

    enum BatchState: Equatable {
        case idle
        case ready
        case running
        case paused
        case cancelled
        case completed
    }

    @Published var batchState: BatchState = .idle
    @Published var batchTotal: Int = 0
    @Published var batchDone: Int = 0
    @Published var batchCurrent: Int = 0   // 1-based

    private let batchQueue: OperationQueue = {
        let q = OperationQueue()
        q.name = "batch.detect.queue"
        q.maxConcurrentOperationCount = 1   // 串行：最稳
        q.qualityOfService = .utility
        q.isSuspended = true
        return q
    }()

    private var batchIDs: [UUID] = []
    private var batchRunID: UUID = UUID()

    
    func setPickerItems(_ pickerItems: [PhotosPickerItem]) {
        cancelBatchInternal(setState: false)

        items = pickerItems.map {
            var it = DetectItem(pickerItem: $0)
            it.status = NSLocalizedString("single.status.loading", comment: "loading")
            return it
        }

        selectedID = items.first?.id
        syncSelectedToPreview()

        batchIDs = items.map(\.id)
        batchTotal = items.count
        batchDone = 0
        batchCurrent = 0
        batchState = items.isEmpty ? .idle : .ready

        if let id = selectedID {
            ensureThumbLoaded(id: id)
            ensurePreviewLoaded(id: id)
            prefetchNeighbors(around: id, keepAround: 30)
        }
    }

    func ensureThumbLoaded(id: UUID) {
        guard let idx = items.firstIndex(where: { $0.id == id }) else { return }
        if items[idx].thumb != nil || items[idx].isLoadingThumb { return }

        items[idx].isLoadingThumb = true
        let picker = items[idx].pickerItem

        thumbTasks[id]?.cancel()
        thumbTasks[id] = Task.detached(priority: .utility) { [weak self] in
            guard let self else { return }
            let data = try? await picker.loadTransferable(type: Data.self)
            if Task.isCancelled { return }

            let img: UIImage? = {
                guard let data else { return nil }
                return ImageUtils.downsample(data: data, maxPixel: self.thumbMaxPixel)
            }()

            await MainActor.run {
                guard let i = self.items.firstIndex(where: { $0.id == id }) else { return }
                self.items[i].thumb = img
                self.items[i].isLoadingThumb = false
                if self.selectedID == id { self.syncSelectedToPreview() }
            }
        }
    }

    func ensurePreviewLoaded(id: UUID) {
        guard let idx = items.firstIndex(where: { $0.id == id }) else { return }
        if items[idx].preview != nil || items[idx].isLoadingPreview { return }

        items[idx].isLoadingPreview = true
        let picker = items[idx].pickerItem

        previewTasks[id]?.cancel()
        previewTasks[id] = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            let data = try? await picker.loadTransferable(type: Data.self)
            if Task.isCancelled { return }

            let img: UIImage?
            if let data {
                img = ImageUtils.downsample(data: data, maxPixel: self.previewMaxPixel)
            } else {
                img = nil
            }

            let pxSize = img.map { CGSize(width: $0.size.width, height: $0.size.height) }
                ?? .init(width: 1, height: 1)

            await MainActor.run {
                guard let i = self.items.firstIndex(where: { $0.id == id }) else { return }
                self.items[i].preview = img
                self.items[i].previewPixelSize = pxSize
                self.items[i].isLoadingPreview = false

                if self.items[i].status == NSLocalizedString("single.status.loading", comment: "") {
                    self.items[i].status = NSLocalizedString("single.status.ready.simple", comment: "ready")
                }
                if self.selectedID == id { self.syncSelectedToPreview() }
            }
        }
    }
    
    func prefetchNeighbors(around id: UUID, keepAround: Int) {
        guard let selIdx = items.firstIndex(where: { $0.id == id }) else { return }

        // 预取当前±keepAround
        let lo = max(0, selIdx - keepAround)
        let hi = min(items.count - 1, selIdx + keepAround)
        if lo <= hi {
            for j in lo...hi {
                let nid = items[j].id
                ensureThumbLoaded(id: nid)
                ensurePreviewLoaded(id: nid)
            }
        }

        // 释放远端 preview（thumb 保留）
        for k in items.indices {
            if abs(k - selIdx) > keepAround {
                let rid = items[k].id
                items[k].preview = nil
                items[k].isLoadingPreview = false
                previewTasks[rid]?.cancel()
                previewTasks[rid] = nil
            }
        }
    }

    // MARK: - Public APIs
    func selectItem(_ id: UUID) {
        selectedID = id
        syncSelectedToPreview()
        prefetchNeighbors(around: id, keepAround: 30)
    }

    func detectSelected() {
        guard let id = selectedID else { return }
        detectInternal(id: id) { }
    }

    func startBatch() {
        guard !items.isEmpty else { return }

        // 新一轮
        batchRunID = UUID()
        let runID = batchRunID

        batchQueue.cancelAllOperations()
        batchQueue.isSuspended = true

        // 本轮全部重跑（更直观）；如需“只跑未完成”，可把 dets.isEmpty 作为过滤条件
        batchIDs = items.map(\.id)
        batchTotal = batchIDs.count
        batchDone = 0
        batchCurrent = 0

        // 先把每张图状态重置为 ready（不在这里做重预处理；检测时再按需加载与 resizeUniform）
        for i in items.indices {
            items[i].dets = []
            items[i].prepared = nil

            // 如果预览尚未加载完成，就保持 loading；否则标记为可开始检测
            if items[i].status != NSLocalizedString("single.status.loading", comment: "") {
                items[i].status = NSLocalizedString("single.status.ready.simple", comment: "ready")
            }
        }
        syncSelectedToPreview()

        batchState = .running

        for (i, id) in batchIDs.enumerated() {
            let op = BlockOperation()
            op.addExecutionBlock { [weak self, weak op] in
                guard let self, let op else { return }
                if op.isCancelled { return }

                DispatchQueue.main.async {
                    // 如果这一轮已经不是当前轮（例如用户重新开始/取消），就不再更新
                    guard self.batchRunID == runID else { return }
                    self.batchCurrent = i + 1
                }

                self.detectOneBlocking(id: id)

                if op.isCancelled { return }

                DispatchQueue.main.async {
                    guard self.batchRunID == runID else { return }
                    self.batchDone += 1
                    if self.batchDone >= self.batchTotal {
                        self.batchState = .completed
                        self.batchQueue.isSuspended = true
                    }
                }
            }
            batchQueue.addOperation(op)
        }

        batchQueue.isSuspended = false
    }

    func pauseBatch() {
        guard batchState == .running else { return }
        batchQueue.isSuspended = true
        batchState = .paused
    }

    func resumeBatch() {
        guard batchState == .paused else { return }
        batchQueue.isSuspended = false
        batchState = .running
    }

    func cancelBatch() {
        cancelBatchInternal(setState: true)
    }
    
    func clearSelectedPhotos() {
        // 让旧回调失效 + 停掉队列
        batchRunID = UUID()
        batchQueue.cancelAllOperations()
        batchQueue.isSuspended = true

        // 取消所有懒加载 / 推理任务（避免无意义资源占用）
        thumbTasks.values.forEach { $0.cancel() }
        previewTasks.values.forEach { $0.cancel() }
        detectTasks.values.forEach { $0.cancel() }
        thumbTasks.removeAll()
        previewTasks.removeAll()
        detectTasks.removeAll()

        items = []
        selectedID = nil

        picked = nil
        dets = []
        status = NSLocalizedString("single.status.pick", comment: "pick a image")

        roi = nil
        srcPixelSize = .init(width: 1, height: 1)

        batchTotal = 0
        batchDone = 0
        batchCurrent = 0
        batchState = .idle
    }

    
    // MARK: - Unified batch primary button (Start/Pause/Resume)

    var batchPrimaryTitleKey: String {
        switch batchState {
        case .running:
            return NSLocalizedString("Batch.Pause", comment:"Pause batch")
        case .paused:
            return NSLocalizedString("Batch.Resume", comment: "Resume batch")
        case .idle, .ready, .cancelled, .completed:
            return NSLocalizedString("Batch.Start", comment: "Start batch")
        }
    }

    var batchPrimaryEnabled: Bool {
        // 没有图片时不可用
        guard !items.isEmpty else { return false }
        // 其余状态都允许点击（running=暂停；paused=继续；其它=开始/重新开始）
        return true
    }

    var batchCancelEnabled: Bool {
        batchState == .running || batchState == .paused
    }

    func batchPrimaryAction() {
        switch batchState {
        case .running:
            pauseBatch()
        case .paused:
            resumeBatch()
        case .idle, .ready, .cancelled, .completed:
            startBatch()
        }
    }
    
    var batchDangerTitleKey: String {
        if batchCancelEnabled {
            return NSLocalizedString("Batch.CancelClear" , comment: "Cancel and clear Batch")     // 运行/暂停时：取消（你也可以改成“取消并清空”）
        } else if !items.isEmpty {
            return NSLocalizedString("Photos.Clear" , comment: "Clear Photo")     // 非运行态但已选图：清除
        } else {
            return NSLocalizedString("Batch.Cancel" , comment: "Cancel Batch")     // 没图：保持文案（但会 disabled）
        }
    }

    var batchDangerEnabled: Bool {
        batchCancelEnabled || !items.isEmpty
    }

    func batchDangerAction() {
        if batchCancelEnabled {
            // 先取消批处理
            cancelBatch()
            // 再清空已选照片
            clearSelectedPhotos()
        } else if !items.isEmpty {
            // 未在跑批处理：直接清空
            clearSelectedPhotos()
        }
    }



    func currentSrcPixelSize() -> CGSize { srcPixelSize }

    // MARK: - Internal helpers

    private func cancelBatchInternal(setState: Bool) {
        batchRunID = UUID() // 让旧回调失效
        batchQueue.cancelAllOperations()
        batchQueue.isSuspended = true

        // 尽量取消仍在排队/等待的推理任务（正在调用 detector 的那一张不保证可中断）
        detectTasks.values.forEach { $0.cancel() }
        detectTasks.removeAll()
        if setState, !items.isEmpty {
            batchState = .cancelled
            // 可选：给未完成的标一下状态
            for i in items.indices where items[i].dets.isEmpty {
                // 需要你在本地化里新增 single.status.cancelled；没有也不影响运行（会显示 key）
                items[i].status = NSLocalizedString("single.status.cancelled", comment: "cancelled")
            }
            syncSelectedToPreview()
        }
        if items.isEmpty {
            batchState = .idle
        }
    }

    private func syncSelectedToPreview() {
        guard let id = selectedID, let item = items.first(where: { $0.id == id }) else {
            picked = nil
            dets = []
            status = NSLocalizedString("single.status.pick", comment: "pick a image")
            roi = nil
            srcPixelSize = .init(width: 1, height: 1)
            return
        }

        picked = item.preview ?? item.thumb
        dets = item.dets
        status = item.status
        srcPixelSize = item.previewPixelSize

        if let p = item.prepared {
            roi = p.roi
            srcPixelSize = p.srcPixelSize
        } else {
            roi = nil
            // 没有 prepared 时，至少使用已加载预览/缩略图的尺寸，保证叠框比例正确
            srcPixelSize = item.previewPixelSize
        }
    }

    /// 给 batch 用：等待单张完成（注意：无法中断“正在推理的这一张”，暂停/取消在两张之间生效）
    private func detectOneBlocking(id: UUID) {
        let sem = DispatchSemaphore(value: 0)
        detectInternal(id: id) {
            sem.signal()
        }
        _ = sem.wait(timeout: .distantFuture)
    }

    /// 单张推理核心：可被“手动检测”和“批量队列”共用
    /// 单张推理核心：可被“手动检测”和“批量队列”共用
    private func detectInternal(id: UUID, completion: @escaping () -> Void) {
        // 如果同一张图正在推理，先取消旧任务（避免结果回写顺序错乱）
        if Thread.isMainThread {
            detectTasks[id]?.cancel()
            detectTasks[id] = nil
        } else {
            DispatchQueue.main.sync {
                self.detectTasks[id]?.cancel()
                self.detectTasks[id] = nil
            }
        }

        let runID = batchRunID
        let dst = Int(detector.inputSize)

        // 先把 UI 状态更新成 running（在主线程）
        DispatchQueue.main.async {
            if let idx = self.items.firstIndex(where: { $0.id == id }) {
                self.items[idx].dets = []
                self.items[idx].status = NSLocalizedString("single.status.running", comment: "detecting")
                if self.selectedID == id { self.syncSelectedToPreview() }
            }
        }

        let task = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else {
                await MainActor.run { completion() }
                return
            }

            // 推理结束后清理任务引用
            defer {
                Task { @MainActor [weak self] in
                    self?.detectTasks[id] = nil
                }
            }

            // 读取必要快照（从主线程拷贝，避免数据竞争）
            let snap = await MainActor.run { () -> (pickerItem: PhotosPickerItem, preview: UIImage?, thumb: UIImage?, prepared: Prepared?)? in
                guard let item = self.items.first(where: { $0.id == id }) else { return nil }
                return (item.pickerItem, item.preview, item.thumb, item.prepared)
            }

            guard let snap else {
                await MainActor.run { completion() }
                return
            }

            if Task.isCancelled {
                await MainActor.run { completion() }
                return
            }

            // 选择推理用的源图：优先 preview；没有则从 PhotosPickerItem 拉 Data 并 downsample
            var srcImage: UIImage? = snap.preview
            var thumbImage: UIImage? = snap.thumb

            if srcImage == nil {
                let data = try? await snap.pickerItem.loadTransferable(type: Data.self)
                if Task.isCancelled {
                    await MainActor.run { completion() }
                    return
                }

                guard let data else {
                    await MainActor.run {
                        guard self.batchRunID == runID else { completion(); return }
                        if let idx = self.items.firstIndex(where: { $0.id == id }) {
                            self.items[idx].status = NSLocalizedString("single.status.failed", comment: "fail to load image")
                            if self.selectedID == id { self.syncSelectedToPreview() }
                        }
                        completion()
                    }
                    return
                }

                // 下采样生成 preview / thumb（都在后台完成，避免主线程卡顿）
                srcImage = ImageUtils.downsample(data: data, maxPixel: self.previewMaxPixel)
                if thumbImage == nil {
                    thumbImage = ImageUtils.downsample(data: data, maxPixel: self.thumbMaxPixel)
                }

                // 回写缓存（如果这一轮仍然有效）
                await MainActor.run {
                    guard self.batchRunID == runID else { return }
                    guard let idx = self.items.firstIndex(where: { $0.id == id }) else { return }

                    if self.items[idx].thumb == nil { self.items[idx].thumb = thumbImage }
                    if self.items[idx].preview == nil {
                        self.items[idx].preview = srcImage
                        if let srcImage {
                            self.items[idx].previewPixelSize = .init(width: srcImage.size.width, height: srcImage.size.height)
                        }
                    }
                    self.items[idx].isLoadingPreview = false

                    if self.selectedID == id { self.syncSelectedToPreview() }
                }
            }

            guard let srcImage else {
                await MainActor.run {
                    guard self.batchRunID == runID else { completion(); return }
                    if let idx = self.items.firstIndex(where: { $0.id == id }) {
                        self.items[idx].status = NSLocalizedString("single.status.failed", comment: "fail to load image")
                        if self.selectedID == id { self.syncSelectedToPreview() }
                    }
                    completion()
                }
                return
            }

            // 准备输入：优先复用 cached prepared（srcPixelSize 匹配时）；否则现算一次
            let currentSrcSize = CGSize(width: srcImage.size.width, height: srcImage.size.height)
            let prep: Prepared?

            if let p = snap.prepared,
               abs(p.srcPixelSize.width - currentSrcSize.width) < 0.5,
               abs(p.srcPixelSize.height - currentSrcSize.height) < 0.5 {
                prep = p
            } else {
                guard let r = ImageUtils.resizeUniform(srcImage, dst: dst) else {
                    await MainActor.run {
                        guard self.batchRunID == runID else { completion(); return }
                        if let idx = self.items.firstIndex(where: { $0.id == id }) {
                            self.items[idx].status = NSLocalizedString("single.status.failed.resize", comment: "fail to resize uniform")
                            if self.selectedID == id { self.syncSelectedToPreview() }
                        }
                        completion()
                    }
                    return
                }
                prep = Prepared(ui: r.ui, roi: r.roi, srcPixelSize: r.srcPixelSize)
            }

            guard let prep else {
                await MainActor.run { completion() }
                return
            }

            // CVPixelBuffer（可在后台线程创建）
            guard let pb = ImageUtils.pixelBufferBGRA(from: prep.ui, dst: dst) else {
                await MainActor.run {
                    guard self.batchRunID == runID else { completion(); return }
                    if let idx = self.items.firstIndex(where: { $0.id == id }) {
                        self.items[idx].status = NSLocalizedString("single.status.failed.pxbuffer", comment: "fail to create pixel buffer")
                        if self.selectedID == id { self.syncSelectedToPreview() }
                    }
                    completion()
                }
                return
            }

            // detector 非线程安全：必须串行调用
            let raw: [[AnyHashable: Any]] = await withCheckedContinuation { (cont: CheckedContinuation<[[AnyHashable: Any]], Never>) in
                self.q.async {
                    let out = (self.detector.detect(with: pb) as? [[AnyHashable: Any]]) ?? []
                    cont.resume(returning: out)
                }
            }

            // 结果映射
            var mapped: [(CGRect, String, Float)] = []
            mapped.reserveCapacity(raw.count)

            for d in raw {
                guard
                    let xN = d["x"] as? NSNumber,
                    let yN = d["y"] as? NSNumber,
                    let wN = d["w"] as? NSNumber,
                    let hN = d["h"] as? NSNumber,
                    let label = d["label"] as? String,
                    let score = d["score"] as? NSNumber
                else { continue }

                var x  = CGFloat(truncating: xN)   // 左上角
                var y  = CGFloat(truncating: yN)
                var ww = CGFloat(truncating: wN)
                var hh = CGFloat(truncating: hN)

                // clamp 0~1
                x  = max(0, min(x, 1))
                y  = max(0, min(y, 1))
                ww = max(0, min(ww, 1 - x))
                hh = max(0, min(hh, 1 - y))

                // 先构造 dst 坐标系（像素）下的框 boxDst
                let boxDst: CGRect
                switch self.boxSpace {
                case .dstSquare:
                    boxDst = CGRect(
                        x: x * CGFloat(dst),
                        y: y * CGFloat(dst),
                        width: ww * CGFloat(dst),
                        height: hh * CGFloat(dst)
                    )
                case .roiContent:
                    boxDst = CGRect(
                        x: prep.roi.x + x * prep.roi.width,
                        y: prep.roi.y + y * prep.roi.height,
                        width: ww * prep.roi.width,
                        height: hh * prep.roi.height
                    )
                }

                // dst -> 原图像素（去 padding 并映射回原图）
                let boxSrcPx = ImageUtils.mapBoxFromDstToSource(
                    boxDst: boxDst,
                    roi: prep.roi,
                    srcSize: prep.srcPixelSize
                )

                // 原图像素 -> 原图归一化（给 scaledToFit overlay 用）
                let rectNorm = CGRect(
                    x: boxSrcPx.minX / max(prep.srcPixelSize.width, 1),
                    y: boxSrcPx.minY / max(prep.srcPixelSize.height, 1),
                    width: boxSrcPx.width / max(prep.srcPixelSize.width, 1),
                    height: boxSrcPx.height / max(prep.srcPixelSize.height, 1)
                )

                mapped.append((rectNorm, label, score.floatValue))
            }

            await MainActor.run {
                defer { completion() }

                // 若用户已重新开始/取消，这一轮结果不再回写
                guard self.batchRunID == runID else { return }
                guard let idx = self.items.firstIndex(where: { $0.id == id }) else { return }

                self.items[idx].prepared = prep
                self.items[idx].dets = mapped
                self.items[idx].status = String.localizedStringWithFormat(
                    NSLocalizedString("single.status.done", comment: "Detection finished status"),
                    mapped.count, dst
                )

                if self.selectedID == id { self.syncSelectedToPreview() }
            }
        }

        // 记录任务，便于取消
        DispatchQueue.main.async {
            self.detectTasks[id] = task
        }
    }

}

struct ImageDetectScreen: View {
    @StateObject private var vm = ImageDetectVM()
    @State private var pickerItems: [PhotosPickerItem] = []

    var body: some View {
        VStack(spacing: 10) {

            // 当前选中图片状态
            Text(verbatim: vm.status)
                .font(.system(size: 14, weight: .semibold))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 14)
                .padding(.top, 10)

            // 缩略图条（多张才显示）
            if vm.items.count > 1 {
                ScrollViewReader { proxy in
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 10) {
                            ForEach(vm.items) { item in
                                ZStack {
                                    if let t = item.thumb {
                                        Image(uiImage: t).resizable().scaledToFill()
                                    } else {
                                        Color.black.opacity(0.12)
                                    }
                                }
                                .frame(width: 64, height: 64)
                                .clipped()
                                .overlay(
                                    RoundedRectangle(cornerRadius: 10)
                                        .stroke(item.id == vm.selectedID ? .orange : .clear, lineWidth: 3)
                                )
                                .clipShape(RoundedRectangle(cornerRadius: 10))
                                .id(item.id) // 关键：用于 scrollTo
                                .onTapGesture { vm.selectItem(item.id) }
                                .task { vm.ensureThumbLoaded(id: item.id) }
                            }
                        }
                        .padding(.horizontal, 14)
                    }
                    // 选中项变化（点击缩略图或左右滑动预览）=> 自动居中
                    .onChange(of: vm.selectedID) { _, newID in
                        guard let newID else { return }
                        // 让出一帧，避免和布局/图片异步加载抢时序导致 scrollTo 偶发失效
                        DispatchQueue.main.async {
                            withAnimation(.easeInOut(duration: 0.22)) {
                                proxy.scrollTo(newID, anchor: .center)
                            }
                        }
                    }
                }
            }


            ZStack {
                Color.black.opacity(0.06)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                if vm.items.isEmpty {
                    Text("No.photo")
                        .foregroundStyle(.secondary)
                } else {
                    TabView(selection: selectedIDBinding) {
                        ForEach(vm.items) { item in
                            GeometryReader { geo in
                                ZStack {
                                    if let img = item.preview ?? item.thumb {
                                        Image(uiImage: img)
                                            .resizable()
                                            .scaledToFit()
                                            .frame(width: geo.size.width, height: geo.size.height)

                                        ImageOverlayFit(
                                            detections: item.dets,
                                            srcPixelSize: (item.preview?.size ?? item.thumb?.size).map { .init(width: $0.width, height: $0.height) } ?? item.previewPixelSize
                                        )
                                    } else {
                                        ProgressView()
                                    }
                                }
                                .frame(width: geo.size.width, height: geo.size.height)
                            }
                            .task {
                                vm.ensureThumbLoaded(id: item.id)
                                vm.ensurePreviewLoaded(id: item.id)
                            }
                            .tag(Optional(item.id))
                        }
                    }
                    .tabViewStyle(.page(indexDisplayMode: .never)) // 不显示底部小圆点（你已有缩略图条）
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(.horizontal, 14)
            
            // 批量进度（有任务才显示）
            if vm.batchTotal > 0, vm.batchState != .idle {
                ProgressView(value: Double(vm.batchDone), total: Double(max(vm.batchTotal, 1)))
                    .padding(.horizontal, 14)
                    .padding(.top, 6)
            }

            
            // Row 1: 选择 + 单张检测
            HStack(spacing: 12) {
                PhotosPicker(selection: $pickerItems, maxSelectionCount: 30, matching: .images) {
                    Text("Choose.photo")
                        .font(.system(size: 16, weight: .semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(.orange.opacity(0.75))
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }

                Button {
                    vm.detectSelected()
                } label: {
                    Text("SingleDetect.Once") // 可保留原 key
                        .font(.system(size: 16, weight: .semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(.green)
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                .disabled(vm.picked == nil || vm.batchState == .running) // 批量运行时避免打乱队列
            }
            .padding(.horizontal, 14)

            // Row 2: 批量控制（合并 Start/Pause/Resume + Cancel）
            HStack(spacing: 12) {
                Button {
                    vm.batchPrimaryAction()
                } label: {
                    HStack(spacing: 8) {
                        Text(LocalizedStringKey(vm.batchPrimaryTitleKey))

                        if vm.batchTotal > 0 {
                            Text("(\(vm.batchDone)/\(vm.batchTotal))")
                                .foregroundStyle(.white.opacity(0.9))
                        }
                    }
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background(primaryBatchButtonColor(for: vm.batchState).opacity(0.85))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                .disabled(!vm.batchPrimaryEnabled)

                Button(role: .destructive) {
                    vm.batchDangerAction()
                    pickerItems = []   // ✅ 清空 PhotosPicker 当前选择（否则再次打开 picker 可能仍保留上次选择）
                } label: {
                    Text(LocalizedStringKey(vm.batchDangerTitleKey))
                        .font(.system(size: 14, weight: .semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                        .background(.red.opacity(0.85))
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                .disabled(!vm.batchDangerEnabled)
            }
            .padding(.horizontal, 14)

            Spacer(minLength: 10)
        }
        .onChange(of: pickerItems) { _, newItems in
            vm.setPickerItems(newItems)
        }


    }
    private func primaryBatchButtonColor(for state: ImageDetectVM.BatchState) -> Color {
        switch state {
        case .running:
            return .gray      // 暂停
        case .paused:
            return .purple    // 继续
        case .idle, .ready, .cancelled, .completed:
            return .blue      // 开始
        }
    }
    private var selectedIDBinding: Binding<UUID?> {
        Binding(
            get: { vm.selectedID },
            set: { newValue in
                if let id = newValue {
                    vm.selectItem(id)   // 这里会同步 status / picked / dets 等
                }
            }
        )
    }

}
