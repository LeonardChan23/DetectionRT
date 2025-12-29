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

@MainActor
final class ImageDetectVM: ObservableObject {
    private enum BoxNormSpace {
        case dstSquare   // x/y/w/h 是相对 dst×dst（含 padding）
        case roiContent  // x/y/w/h 是相对 roi 内容区（不含 padding）
    }
//    @Published var debugInput: UIImage? = nil
//    @Published var showDebugInput: Bool = false


    // ✅ 默认：中心点 + 相对 dst×dst
    private let boxSpace: BoxNormSpace = .dstSquare
    
    @Published var status: String = NSLocalizedString("single.status.pick", comment: "pick a image")
    @Published var picked: UIImage? = nil
    @Published var dets: [(rectNorm: CGRect, label: String, score: Float)] = []

    private var roi: EffectROI? = nil
    private var srcPixelSize: CGSize = .init(width: 1, height: 1)

    private let detector = NanoDetSingleDetector()
    private let q = DispatchQueue(label: "single.image.detect.queue", qos: .userInitiated)

    func setImage(_ img: UIImage) {
        dets = []
        let up = ImageUtils.normalizedUp(img)   // ✅ 统一坐标系
        picked = up

        let dst = Int(detector.inputSize)
        if let r = ImageUtils.resizeUniform(up, dst: dst) {
            roi = r.roi
            srcPixelSize = r.srcPixelSize
            status = String(format: NSLocalizedString("single.status.ready", comment: "ready for detection"), locale: .current,dst, Int(r.roi.x), Int(r.roi.y), Int(r.roi.width), Int(r.roi.height))
        } else {
            roi = nil
            status = NSLocalizedString("single.status.failed", comment: "fail to load image")
        }
    }



    func detectOnce() {
        guard let img = picked else { return }

        let dst = Int(detector.inputSize)   // ✅ 关键：跟随模型输入尺寸

        guard let r = ImageUtils.resizeUniform(img, dst: dst) else {
            status = NSLocalizedString("single.status.failed.resize", comment: "fail to resize uniform")
            return
        }

        print("[DBG] model dst=\(dst)")
        print("[DBG] ui.size(points)=\(r.ui.size), scale=\(r.ui.scale)")
        print("[DBG] ui.cgImage(px)=\(r.ui.cgImage?.width ?? -1)x\(r.ui.cgImage?.height ?? -1)")

        guard let pb = ImageUtils.pixelBufferBGRA(from: r.ui, dst: dst) else {
            status = NSLocalizedString("single.status.failed.pxbuffer", comment: "fail to create pixel buffer")
            return
        }

//        self.debugInput = ImageUtils.uiImageRaw(from: pb)

        
//        if let dbg = ImageUtils.uiImageRaw(from: pb) {
//            print("[DBG] pb->ui size(px)=\(dbg.cgImage?.width ?? -1)x\(dbg.cgImage?.height ?? -1)")
//        }


        let pw = CVPixelBufferGetWidth(pb)
        let ph = CVPixelBufferGetHeight(pb)
        print("[DBG] pb=\(pw)x\(ph)")

//        if pw != dst || ph != dst {
//            status = "内部错误：pb=\(pw)x\(ph)，不是 \(dst)x\(dst)"
//            return
//        }

        status = NSLocalizedString("single.status.running", comment: "detecting")
        dets = []

        q.async { [weak self] in
            guard let self else { return }

            // ObjC: detectWithPixelBuffer: => Swift: detect(with:)
            let raw = (self.detector.detect(with: pb) as? [[AnyHashable: Any]]) ?? []

            var mapped: [(CGRect, String, Float)] = []
            mapped.reserveCapacity(raw.count)
            
            print("[DBG] roi=\(r.roi)")
            print("[DBG] raw.first=\(raw.first ?? [:])")

            for d in raw {
                guard
                    let xN = d["x"] as? NSNumber,
                    let yN = d["y"] as? NSNumber,
                    let wN = d["w"] as? NSNumber,
                    let hN = d["h"] as? NSNumber,
                    let label = d["label"] as? String,
                    let score = d["score"] as? NSNumber
                else { continue }

                var x  = CGFloat(truncating: xN)   // ✅ 左上角
                var y  = CGFloat(truncating: yN)
                var ww = CGFloat(truncating: wN)
                var hh = CGFloat(truncating: hN)


                // clamp 0~1
                x  = max(0, min(x, 1))
                y  = max(0, min(y, 1))
                ww = max(0, min(ww, 1 - x))
                hh = max(0, min(hh, 1 - y))
//                y = 1 - y - hh

                // 先构造 dst 坐标系（像素）下的框 boxDst
                let boxDst: CGRect
                switch boxSpace {
                case .dstSquare:
                    // x/y/w/h 相对 dst×dst（含 padding）
                    boxDst = CGRect(
                        x: x * CGFloat(dst),
                        y: y * CGFloat(dst),
                        width: ww * CGFloat(dst),
                        height: hh * CGFloat(dst)
                    )

                case .roiContent:
                    // x/y/w/h 相对 roi 内容区（不含 padding）
                    boxDst = CGRect(
                        x: r.roi.x + x * r.roi.width,
                        y: r.roi.y + y * r.roi.height,
                        width: ww * r.roi.width,
                        height: hh * r.roi.height
                    )
                }

                // dst -> 原图像素（去 padding 并映射回原图）
                let boxSrcPx = ImageUtils.mapBoxFromDstToSource(
                    boxDst: boxDst,
                    roi: r.roi,
                    srcSize: r.srcPixelSize
                )

                // 原图像素 -> 原图归一化（给 scaledToFit overlay 用）
                let rectNorm = CGRect(
                    x: boxSrcPx.minX / max(r.srcPixelSize.width, 1),
                    y: boxSrcPx.minY / max(r.srcPixelSize.height, 1),
                    width: boxSrcPx.width / max(r.srcPixelSize.width, 1),
                    height: boxSrcPx.height / max(r.srcPixelSize.height, 1)
                )

                mapped.append((rectNorm, label, score.floatValue))
            }


            DispatchQueue.main.async {
                self.dets = mapped
                self.status = String.localizedStringWithFormat(           NSLocalizedString("single.status.done", comment: "Detection finished status"), mapped.count, dst)
            }
        }
    }

    func currentSrcPixelSize() -> CGSize { srcPixelSize }
}

struct ImageDetectScreen: View {
    @StateObject private var vm = ImageDetectVM()
    @State private var pickerItem: PhotosPickerItem? = nil

    var body: some View {
        VStack(spacing: 12) {
            Text(verbatim: vm.status)
                .font(.system(size: 14, weight: .semibold))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 14)
                .padding(.top, 10)

            ZStack {
                Color.black.opacity(0.06)
                    .clipShape(RoundedRectangle(cornerRadius: 16))

                if let img = vm.picked {
                    GeometryReader { geo in
                        ZStack {
//                            let showImg = (vm.showDebugInput ? (vm.debugInput ?? img) : img)

                            Image(uiImage: img)
                                .resizable()
                                .scaledToFit()
                                .frame(width: geo.size.width, height: geo.size.height)

                            ImageOverlayFit(detections: vm.dets, srcPixelSize: vm.currentSrcPixelSize())
                        }
                    }
                } else {
                    Text("No.photo")
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(.horizontal, 14)


            HStack(spacing: 12) {
//                Toggle("显示模型输入(pb)", isOn: $vm.showDebugInput)
//                    .padding(.horizontal, 14)
                PhotosPicker(selection: $pickerItem, matching: .images) {
                    Text("Choose.photo")
                        .font(.system(size: 16, weight: .semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(.orange.opacity(0.75))
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }

                Button {
                    vm.detectOnce()
                } label: {
                    Text("SingleDetect.Once")
                        .font(.system(size: 16, weight: .semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(.green)
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                .disabled(vm.picked == nil)
            }
            .padding(.horizontal, 14)

            Spacer(minLength: 10)
        }
        .onChange(of: pickerItem) { _, newItem in
            guard let newItem else { return }
            Task {
                if let data = try? await newItem.loadTransferable(type: Data.self),
                   let image = UIImage(data: data) {
                    vm.setImage(image)
                } else {
                    vm.status = NSLocalizedString("single.status.failed", comment: "fail to load image")
                }
            }
        }
    }
}
