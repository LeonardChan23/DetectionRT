//
//  ImageDetectScreen.swift
//  LeonardCNNtest
//
//  Created by 陳暄暢 on 25/12/2025.
//


import SwiftUI
import PhotosUI
import AVFoundation

// effect_roi 对应 main.cpp 里的 object_rect
struct EffectROI {
    var x: CGFloat
    var y: CGFloat
    var width: CGFloat
    var height: CGFloat
}

enum ImageUtils {

    static func normalizedUp(_ image: UIImage) -> UIImage {
        if image.imageOrientation == .up { return image }
        let renderer = UIGraphicsImageRenderer(size: image.size)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: image.size))
        }
    }

    /// 复刻 main.cpp resize_uniform：输出 416x416 + effect_roi（在 416 坐标系里）
    static func resizeUniform416(_ image: UIImage) -> (ui416: UIImage, roi: EffectROI, srcPixelSize: CGSize)? {
        let img = normalizedUp(image)
        guard let cg = img.cgImage else { return nil }

        let srcW = CGFloat(cg.width)
        let srcH = CGFloat(cg.height)
        let dstW: CGFloat = 416
        let dstH: CGFloat = 416

        let ratioSrc = srcW / srcH
        let ratioDst = dstW / dstH  // =1

        var tmpW: CGFloat = 0
        var tmpH: CGFloat = 0
        var padX: CGFloat = 0
        var padY: CGFloat = 0

        if ratioSrc > ratioDst {
            tmpW = dstW
            tmpH = floor((dstW / srcW) * srcH)
            padX = 0
            padY = floor((dstH - tmpH) / 2.0)
        } else if ratioSrc < ratioDst {
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

        let renderer = UIGraphicsImageRenderer(size: CGSize(width: dstW, height: dstH))
        let out = renderer.image { ctx in
            UIColor.black.setFill()
            ctx.fill(CGRect(x: 0, y: 0, width: dstW, height: dstH))
            img.draw(in: CGRect(x: padX, y: padY, width: tmpW, height: tmpH))
        }

        let roi = EffectROI(x: padX, y: padY, width: tmpW, height: tmpH)
        return (out, roi, CGSize(width: srcW, height: srcH))
    }

    /// UIImage(416x416) -> CVPixelBuffer(BGRA)
    static func pixelBufferBGRA(from image: UIImage) -> CVPixelBuffer? {
        let img = normalizedUp(image)
        guard let cg = img.cgImage else { return nil }

        let w = cg.width
        let h = cg.height

        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        let st = CVPixelBufferCreate(kCFAllocatorDefault, w, h,
                                     kCVPixelFormatType_32BGRA,
                                     attrs as CFDictionary, &pb)
        guard st == kCVReturnSuccess, let pixelBuffer = pb else { return nil }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let bpr = CVPixelBufferGetBytesPerRow(pixelBuffer)

        let cs = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.union(.init(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue))

        guard let ctx = CGContext(data: base, width: w, height: h,
                                  bitsPerComponent: 8, bytesPerRow: bpr,
                                  space: cs, bitmapInfo: bitmapInfo.rawValue) else { return nil }

        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return pixelBuffer
    }

    /// 把 416 坐标系里的 box 映射回原图坐标（复刻 main.cpp draw_bboxes 的数学）
    static func mapBoxFrom416ToSource(box416: CGRect, roi: EffectROI, srcSize: CGSize) -> CGRect {
        let srcW = max(srcSize.width, 1)
        let srcH = max(srcSize.height, 1)

        let dstW = max(roi.width, 1)
        let dstH = max(roi.height, 1)

        let widthRatio  = srcW / dstW
        let heightRatio = srcH / dstH

        let x1 = (box416.minX - roi.x) * widthRatio
        let y1 = (box416.minY - roi.y) * heightRatio
        let x2 = (box416.maxX - roi.x) * widthRatio
        let y2 = (box416.maxY - roi.y) * heightRatio

        // clamp
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
    @Published var status: String = "请选择一张图片"
    @Published var picked: UIImage? = nil
    @Published var dets: [(rectNorm: CGRect, label: String, score: Float)] = []

    private var roi: EffectROI? = nil
    private var srcPixelSize: CGSize = .init(width: 1, height: 1)

    private let detector = NanoDetSingleDetector()
    private let q = DispatchQueue(label: "single.image.detect.queue", qos: .userInitiated)

    func setImage(_ img: UIImage) {
        dets = []
        status = "已选择图片"
        picked = img

        if let r = ImageUtils.resizeUniform416(img) {
            roi = r.roi
            srcPixelSize = r.srcPixelSize
            status = "可开始检测（roi: x=\(Int(r.roi.x)) y=\(Int(r.roi.y)) w=\(Int(r.roi.width)) h=\(Int(r.roi.height)))"
        } else {
            roi = nil
            status = "图片处理失败"
        }
    }

    func detectOnce() {
        guard let img = picked else { return }
        guard let r = ImageUtils.resizeUniform416(img) else {
            status = "resize_uniform 失败"
            return
        }
        guard let pb = ImageUtils.pixelBufferBGRA(from: r.ui416) else {
            status = "PixelBuffer 创建失败"
            return
        }

        status = "检测中…"
        dets = []

        q.async { [weak self] in
            guard let self else { return }

            let raw = (self.detector.detect(with: pb) as? [[AnyHashable: Any]]) ?? []

            var mapped: [(CGRect, String, Float)] = []
            mapped.reserveCapacity(raw.count)

            for d in raw {
                guard
                    let x = d["x"] as? NSNumber,
                    let y = d["y"] as? NSNumber,
                    let w = d["w"] as? NSNumber,
                    let h = d["h"] as? NSNumber,
                    let label = d["label"] as? String,
                    let score = d["score"] as? NSNumber
                else { continue }

                // detector 输出是相对 416 的 0~1
                let x1 = CGFloat(truncating: x) * 416
                let y1 = CGFloat(truncating: y) * 416
                let ww = CGFloat(truncating: w) * 416
                let hh = CGFloat(truncating: h) * 416
                let box416 = CGRect(x: x1, y: y1, width: ww, height: hh)

                // 映射回原图像素坐标
                let boxSrcPx = ImageUtils.mapBoxFrom416ToSource(box416: box416, roi: r.roi, srcSize: r.srcPixelSize)

                // 转回 0~1（相对原图），用于 overlay
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
                self.status = "完成：\(mapped.count) 个目标"
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
            Text(vm.status)
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
                            Image(uiImage: img)
                                .resizable()
                                .scaledToFit()
                                .frame(width: geo.size.width, height: geo.size.height)

                            ImageOverlayFit(detections: vm.dets, srcPixelSize: vm.currentSrcPixelSize())
                        }
                    }
                } else {
                    Text("未选择图片")
                        .foregroundStyle(.secondary)
                }
            }
            .frame(height: 420)
            .padding(.horizontal, 14)

            HStack(spacing: 12) {
                PhotosPicker(selection: $pickerItem, matching: .images) {
                    Text("选择照片")
                        .font(.system(size: 16, weight: .semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(.black.opacity(0.75))
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                }

                Button {
                    vm.detectOnce()
                } label: {
                    Text("检测一次")
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
                    vm.status = "读取照片失败"
                }
            }
        }
    }
}
