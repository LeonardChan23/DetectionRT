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

// effect_roi 对应 demo 里的 object_rect：在 dst×dst 坐标系中，真实内容区域（去掉 padding）
struct EffectROI {
    var x: CGFloat
    var y: CGFloat
    var width: CGFloat
    var height: CGFloat
}

enum ImageUtils {
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
        let image: UIImage
        var prepared: Prepared?
        var dets: [(rectNorm: CGRect, label: String, score: Float)] = []
        var status: String = ""
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

    // MARK: - Public APIs

    func setImage(_ img: UIImage) {
        setImages([img])
    }

    func setImages(_ images: [UIImage]) {
        // 选新图时，先终止旧批量
        cancelBatchInternal(setState: false)

        let dst = Int(detector.inputSize)

        var newItems: [DetectItem] = []
        newItems.reserveCapacity(images.count)

        for img in images {
            let up = ImageUtils.normalizedUp(img)
            var item = DetectItem(image: up)

            if let r = ImageUtils.resizeUniform(up, dst: dst) {
                let prepared = Prepared(ui: r.ui, roi: r.roi, srcPixelSize: r.srcPixelSize)
                item.prepared = prepared
                item.status = String(
                    format: NSLocalizedString("single.status.ready", comment: "ready for detection"),
                    locale: .current,
                    dst, Int(r.roi.x), Int(r.roi.y), Int(r.roi.width), Int(r.roi.height)
                )
            } else {
                item.prepared = nil
                item.status = NSLocalizedString("single.status.failed", comment: "fail to load image")
            }
            newItems.append(item)
        }

        items = newItems
        selectedID = items.first?.id
        syncSelectedToPreview()

        batchIDs = items.map(\.id)
        batchTotal = items.count
        batchDone = 0
        batchCurrent = 0
        batchState = items.isEmpty ? .idle : .ready
    }

    func selectItem(_ id: UUID) {
        selectedID = id
        syncSelectedToPreview()
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

        // 先把每张图状态重置为 ready（保留 prepared）
        let dst = Int(detector.inputSize)
        for i in items.indices {
            items[i].dets = []
            if let p = items[i].prepared {
                items[i].status = String(
                    format: NSLocalizedString("single.status.ready", comment: "ready for detection"),
                    locale: .current,
                    dst, Int(p.roi.x), Int(p.roi.y), Int(p.roi.width), Int(p.roi.height)
                )
            } else {
                items[i].status = NSLocalizedString("single.status.failed", comment: "fail to load image")
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

        picked = item.image
        dets = item.dets
        status = item.status

        if let p = item.prepared {
            roi = p.roi
            srcPixelSize = p.srcPixelSize
        } else {
            roi = nil
            srcPixelSize = .init(width: 1, height: 1)
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
    private func detectInternal(id: UUID, completion: @escaping () -> Void) {
        // 读取快照（线程安全：通过 main queue 读取 items）
        let snapshot: (image: UIImage, prepared: Prepared?)? = {
            if Thread.isMainThread {
                guard let item = items.first(where: { $0.id == id }) else { return nil }
                return (item.image, item.prepared)
            } else {
                return DispatchQueue.main.sync {
                    guard let item = self.items.first(where: { $0.id == id }) else { return nil }
                    return (item.image, item.prepared)
                }
            }
        }()

        guard let snapshot else {
            DispatchQueue.main.async { completion() }
            return
        }

        let dst = Int(detector.inputSize)

        // 先把 UI 状态更新成 running（在主线程）
        DispatchQueue.main.async {
            if let idx = self.items.firstIndex(where: { $0.id == id }) {
                self.items[idx].dets = []
                self.items[idx].status = NSLocalizedString("single.status.running", comment: "detecting")
                if self.selectedID == id { self.syncSelectedToPreview() }
            }
        }

        // 真正推理与映射放在 q 串行队列（避免 detector 线程不安全问题）
        q.async { [weak self] in
            guard let self else {
                DispatchQueue.main.async { completion() }
                return
            }

            // 准备输入：优先用缓存 prepared；没有就现算一次
            let prep: Prepared?
            if let p = snapshot.prepared {
                prep = p
            } else {
                guard let r = ImageUtils.resizeUniform(snapshot.image, dst: dst) else {
                    DispatchQueue.main.async {
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
                DispatchQueue.main.async { completion() }
                return
            }

            // CVPixelBuffer
            guard let pb = ImageUtils.pixelBufferBGRA(from: prep.ui, dst: dst) else {
                DispatchQueue.main.async {
                    if let idx = self.items.firstIndex(where: { $0.id == id }) {
                        self.items[idx].status = NSLocalizedString("single.status.failed.pxbuffer", comment: "fail to create pixel buffer")
                        if self.selectedID == id { self.syncSelectedToPreview() }
                    }
                    completion()
                }
                return
            }

            // ObjC: detectWithPixelBuffer: => Swift: detect(with:)
            let raw = (self.detector.detect(with: pb) as? [[AnyHashable: Any]]) ?? []

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

            DispatchQueue.main.async {
                // 可能已经换了一批图片：用 id 再找一次
                guard let idx = self.items.firstIndex(where: { $0.id == id }) else {
                    completion()
                    return
                }

                // 回写 prepared（用于后续复用）
                self.items[idx].prepared = prep
                self.items[idx].dets = mapped
                self.items[idx].status = String.localizedStringWithFormat(
                    NSLocalizedString("single.status.done", comment: "Detection finished status"),
                    mapped.count, dst
                )

                if self.selectedID == id {
                    self.syncSelectedToPreview()
                }
                completion()
            }
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
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 10) {
                        ForEach(vm.items) { item in
                            Image(uiImage: item.image)
                                .resizable()
                                .scaledToFill()
                                .frame(width: 64, height: 64)
                                .clipped()
                                .overlay(
                                    RoundedRectangle(cornerRadius: 10)
                                        .stroke(item.id == vm.selectedID ? .orange : .clear, lineWidth: 3)
                                )
                                .clipShape(RoundedRectangle(cornerRadius: 10))
                                .onTapGesture {
                                    vm.selectItem(item.id)
                                }
                        }
                    }
                    .padding(.horizontal, 14)
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
                                    Image(uiImage: item.image)
                                        .resizable()
                                        .scaledToFit()
                                        .frame(width: geo.size.width, height: geo.size.height)

                                    ImageOverlayFit(
                                        detections: item.dets,
                                        srcPixelSize: item.prepared?.srcPixelSize ?? .init(width: 1, height: 1)
                                    )
                                }
                                .frame(width: geo.size.width, height: geo.size.height)
                            }
                            .tag(Optional(item.id)) // 注意：selection 是 UUID?，tag 也要 Optional
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
            guard !newItems.isEmpty else { return }
            Task {
                var images: [UIImage] = []
                images.reserveCapacity(newItems.count)

                // 串行加载，避免一次性并发解码导致内存峰值
                for it in newItems {
                    if let data = try? await it.loadTransferable(type: Data.self),
                       let image = UIImage(data: data) {
                        images.append(image)
                    }
                }

                DispatchQueue.main.async {
                    if images.isEmpty {
                        vm.status = NSLocalizedString("single.status.failed", comment: "fail to load image")
                    } else {
                        vm.setImages(images)
                    }
                }
            }
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
