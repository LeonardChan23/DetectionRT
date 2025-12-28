//
//  NanoDetSingleDetector.mm
//  LeonardCNNtest
//

#import "NanoDetSingleDetector.h"

#import <MNN/Interpreter.hpp>
#import <MNN/Tensor.hpp>
#import <MNN/ImageProcess.hpp>

#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>

// -----------------------------
// Tunables（先强收敛，防止“海量高分框”）
// -----------------------------
static constexpr float CONF_THR        = 0.45f; // 先 0.45 压住；后续可调 0.25~0.45
static constexpr float NMS_THR_CLASS   = 0.45f;
static constexpr float NMS_THR_AGN     = 0.50f;
static constexpr bool  ENABLE_AGN_NMS  = true;

static constexpr int   PRE_NMS_TOPK    = 300;
static constexpr int   PER_CLASS_TOPK  = 50;
static constexpr int   FINAL_TOPN      = 20;
static constexpr float MIN_BOX_PX      = 12.f;  // 在输入尺寸坐标系里过滤碎框（12/16 可试）

// -----------------------------
// Helpers
// -----------------------------
static inline float sigmoid_stable(float x) {
    if (x >= 0.f) { float z = std::exp(-x); return 1.f / (1.f + z); }
    else { float z = std::exp(x); return z / (1.f + z); }
}
static inline float sigmoid_safe(float x) {
    // 若已经是 0~1 概率，避免重复 sigmoid 导致 100%+ / 分布异常
    if (x >= 0.f && x <= 1.f) return x;
    return sigmoid_stable(x);
}

struct BoxInfo {
    float x1, y1, x2, y2;
    float score;
    int label;
};

static inline float iou(const BoxInfo& a, const BoxInfo& b) {
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);
    float w = std::max(0.f, xx2 - xx1);
    float h = std::max(0.f, yy2 - yy1);
    float inter = w * h;
    float areaA = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
    float areaB = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
    float uni = areaA + areaB - inter;
    return (uni <= 0.f) ? 0.f : (inter / uni);
}

// per-class NMS（同类抑制）
static void nms_per_class(std::vector<BoxInfo>& boxes, float thr) {
    std::sort(boxes.begin(), boxes.end(), [](const BoxInfo& a, const BoxInfo& b){ return a.score > b.score; });
    std::vector<bool> removed(boxes.size(), false);
    std::vector<BoxInfo> kept;
    kept.reserve(boxes.size());

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (removed[i]) continue;
        kept.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (removed[j]) continue;
            if (boxes[i].label != boxes[j].label) continue;
            if (iou(boxes[i], boxes[j]) > thr) removed[j] = true;
        }
    }
    boxes.swap(kept);
}

// agnostic NMS（跨类去重）
static void nms_agnostic(std::vector<BoxInfo>& boxes, float thr) {
    std::sort(boxes.begin(), boxes.end(), [](const BoxInfo& a, const BoxInfo& b){ return a.score > b.score; });
    std::vector<bool> removed(boxes.size(), false);
    std::vector<BoxInfo> kept;
    kept.reserve(boxes.size());

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (removed[i]) continue;
        kept.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (removed[j]) continue;
            if (iou(boxes[i], boxes[j]) > thr) removed[j] = true;
        }
    }
    boxes.swap(kept);
}

// COCO 80 labels（默认）
static NSArray<NSString *> *CocoLabels() {
    static NSArray<NSString *> *labels = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        labels = @[
            @"person",@"bicycle",@"car",@"motorcycle",@"airplane",@"bus",@"train",@"truck",@"boat",@"traffic light",
            @"fire hydrant",@"stop sign",@"parking meter",@"bench",@"bird",@"cat",@"dog",@"horse",@"sheep",@"cow",
            @"elephant",@"bear",@"zebra",@"giraffe",@"backpack",@"umbrella",@"handbag",@"tie",@"suitcase",@"frisbee",
            @"skis",@"snowboard",@"sports ball",@"kite",@"baseball bat",@"baseball glove",@"skateboard",@"surfboard",@"tennis racket",@"bottle",
            @"wine glass",@"cup",@"fork",@"knife",@"spoon",@"bowl",@"banana",@"apple",@"sandwich",@"orange",
            @"broccoli",@"carrot",@"hot dog",@"pizza",@"donut",@"cake",@"chair",@"couch",@"potted plant",@"bed",
            @"dining table",@"toilet",@"tv",@"laptop",@"mouse",@"remote",@"keyboard",@"cell phone",@"microwave",@"oven",
            @"toaster",@"sink",@"refrigerator",@"book",@"clock",@"vase",@"scissors",@"teddy bear",@"hair drier",@"toothbrush"
        ];
    });
    return labels;
}

// 输出张量视图：统一成 (N boxes, C channels)
struct OutView {
    int n = 0;
    int c = 0;
    bool boxMajor = true; // true: [n,c]; false: [c,n]
    const float* ptr = nullptr;

    inline float at(int i, int j) const {
        return boxMajor ? ptr[i * c + j] : ptr[j * n + i];
    }
};

// 支持：[1,n,c] / [1,c,n] / [n,c] / [c,n]
static inline bool makeOutView(const MNN::Tensor* t, OutView& v) {
    auto s = t->shape();
    if (s.size() == 3) {
        int a = s[1], b = s[2];
        // heuristic: channel 通常较小（84/85/…）
        if (b <= 512 && a > b) { v.n = a; v.c = b; v.boxMajor = true;  return true; }   // [n,c]
        if (a <= 512 && b > a) { v.n = b; v.c = a; v.boxMajor = false; return true; }   // [c,n]
        v.n = a; v.c = b; v.boxMajor = true; return true;
    }
    if (s.size() == 2) {
        int a = s[0], b = s[1];
        if (b <= 512 && a > b) { v.n = a; v.c = b; v.boxMajor = true;  return true; }
        if (a <= 512 && b > a) { v.n = b; v.c = a; v.boxMajor = false; return true; }
        v.n = a; v.c = b; v.boxMajor = true; return true;
    }
    return false;
}

@implementation NanoDetSingleDetector {
    std::shared_ptr<MNN::Interpreter> _net;
    MNN::Session* _session;
    MNN::Tensor* _input;
    MNN::Tensor* _output;
    std::shared_ptr<MNN::CV::ImageProcess> _pretreat;

    int _inW, _inH;
    BOOL _isNCHW;
    int _numClass;     // 默认 80（COCO）
    BOOL _ready;
}

- (int)inputSize {
    return _ready ? _inW : 416;
}

- (instancetype)init {
    self = [super init];
    if (!self) return nil;

    _ready = NO;
    _numClass = 80;

    // 1) 改成你的 YOLO 模型文件名（不带扩展名）
    //    例如：yolo11n_640.mnn / yolov5n_640.mnn
    NSString *path = [[NSBundle mainBundle] pathForResource:@"yolo11n_640" ofType:@"mnn"];
    if (!path) {
        NSLog(@"[YOLOSingleDetector] model not found in bundle");
        return self;
    }

    _net.reset(MNN::Interpreter::createFromFile(path.UTF8String));
    if (!_net) {
        NSLog(@"[YOLOSingleDetector] createFromFile failed");
        return self;
    }

    // 放在 init 里，createSession 之前

    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_Low;   // FP16，速度更快
    backendConfig.power     = MNN::BackendConfig::Power_High;
    backendConfig.memory    = MNN::BackendConfig::Memory_High;

    MNN::ScheduleConfig cfg;
    cfg.type = MNN_FORWARD_METAL;          // GPU: Metal
    cfg.backupType = MNN_FORWARD_CPU;      // Metal 不可用则回落 CPU
    cfg.numThread = 4;                     // 仅对 backup CPU 有意义
    cfg.backendConfig = &backendConfig;

    _session = _net->createSession(cfg);
    if (!_session) {
        NSLog(@"[MNNDetector] createSession METAL failed, fallback to CPU");
        MNN::ScheduleConfig cpuCfg;
        cpuCfg.type = MNN_FORWARD_CPU;
        cpuCfg.numThread = 4;
        _session = _net->createSession(cpuCfg);
    }

    if (!_session) {
        NSLog(@"[YOLOSingleDetector] createSession failed");
        return self;
    }

    _input = _net->getSessionInput(_session, nullptr);
    if (!_input) {
        NSLog(@"[YOLOSingleDetector] getSessionInput failed");
        return self;
    }

    // 输出名不一定叫 output：优先 "output"，拿不到就取第一个
    _output = _net->getSessionOutput(_session, "output");
    if (!_output) {
        auto outs = _net->getSessionOutputAll(_session);
        if (!outs.empty()) _output = outs.begin()->second;
    }
    if (!_output) {
        NSLog(@"[YOLOSingleDetector] getSessionOutput failed");
        return self;
    }

    // 2) 从 input shape 推断输入尺寸 + layout
    {
        auto s = _input->shape(); // 常见 [1,3,H,W] 或 [1,H,W,3]
        _isNCHW = YES;
        _inW = 640; _inH = 640;

        if (s.size() == 4) {
            if (s[1] == 3) { _isNCHW = YES;  _inH = s[2]; _inW = s[3]; }
            else if (s[3] == 3) { _isNCHW = NO; _inH = s[1]; _inW = s[2]; }
        }
    }

    // 3) preprocess：BGRA -> RGB，归一化 /255（Ultralytics YOLO 常用）
    MNN::CV::ImageProcess::Config icfg;
    ::memset(&icfg, 0, sizeof(icfg));
    icfg.filterType   = MNN::CV::BILINEAR;
    icfg.sourceFormat = MNN::CV::BGRA;
    icfg.destFormat   = MNN::CV::RGB;

    icfg.mean[0] = 0.f; icfg.mean[1] = 0.f; icfg.mean[2] = 0.f; icfg.mean[3] = 0.f;

    // 如果你的 MNN 头文件里没有 normal 字段而是 norm，请把 normal 改成 norm
    icfg.normal[0] = 1.f/255.f; icfg.normal[1] = 1.f/255.f; icfg.normal[2] = 1.f/255.f; icfg.normal[3] = 1.f;

    _pretreat.reset(MNN::CV::ImageProcess::create(icfg));
    if (!_pretreat) {
        NSLog(@"[YOLOSingleDetector] ImageProcess create failed");
        return self;
    }

    _net->releaseModel();
    _ready = YES;

    NSLog(@"[YOLOSingleDetector] ready=YES, input=%dx%d layout=%s",
          _inW, _inH, _isNCHW ? "NCHW" : "NHWC");
    return self;
}

- (NSArray<NSDictionary *> *)detectWithPixelBuffer:(CVPixelBufferRef)pixelBuffer {
    if (!_ready || !pixelBuffer) return @[];

    int srcW = (int)CVPixelBufferGetWidth(pixelBuffer);
    int srcH = (int)CVPixelBufferGetHeight(pixelBuffer);

    // Swift 侧应当先 resize_uniform 到 inputSize×inputSize；不一致也能跑，但会影响对齐
    if (srcW != _inW || srcH != _inH) {
        NSLog(@"[YOLOSingleDetector] warning: pixelBuffer=%dx%d, expect=%dx%d",
              srcW, srcH, _inW, _inH);
    }

    // 1) preprocess
    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    uint8_t* base = (uint8_t*)CVPixelBufferGetBaseAddress(pixelBuffer);
    int bpr = (int)CVPixelBufferGetBytesPerRow(pixelBuffer);

    MNN::CV::Matrix m; // identity（你已经在 Swift resize_uniform 到正确尺寸）
    _pretreat->setMatrix(m);
    _pretreat->convert(base, srcW, srcH, bpr, _input);

    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

    // 2) infer
    _net->runSession(_session);

    // 3) output -> host
    std::shared_ptr<MNN::Tensor> outHost(MNN::Tensor::createHostTensorFromDevice(_output, true));
    const float* outPtr = outHost->host<float>();
    if (!outPtr) return @[];

    OutView ov;
    if (!makeOutView(outHost.get(), ov)) {
        NSLog(@"[YOLOSingleDetector] unsupported output shape");
        return @[];
    }
    ov.ptr = outPtr;

    // 4) decode
    std::vector<BoxInfo> cands;
    cands.reserve(512);

    // 情况 A：导出时已带 NMS，常见 [N,6] => x1,y1,x2,y2,score,cls
    if (ov.c == 6) {
        for (int i = 0; i < ov.n; ++i) {
            float x1 = ov.at(i,0), y1 = ov.at(i,1), x2 = ov.at(i,2), y2 = ov.at(i,3);
            float sc = ov.at(i,4);
            int   cl = (int)ov.at(i,5);

            // 兼容归一化坐标
            if (x2 <= 1.5f && y2 <= 1.5f) {
                x1 *= _inW; x2 *= _inW;
                y1 *= _inH; y2 *= _inH;
            }

            float bw = x2 - x1, bh = y2 - y1;
            if (sc < CONF_THR) continue;
            if (bw < MIN_BOX_PX || bh < MIN_BOX_PX) continue;

            cands.push_back(BoxInfo{
                std::max(0.f,x1), std::max(0.f,y1),
                std::min((float)_inW,x2), std::min((float)_inH,y2),
                std::min(std::max(sc,0.f),1.f), cl
            });
        }
    } else {
        // 情况 B：raw 输出
        // YOLO11/8: C = 4 + numClass（常见 84）
        // YOLOv5:   C = 4 + 1 + numClass（常见 85）
        int C = ov.c;

        bool hasObj = false;
        int numClass = 0;

        if (C == 85) { hasObj = true; numClass = 80; }
        else if (C == 84) { hasObj = false; numClass = 80; }
        else {
            // 泛化：优先按 4+cls
            hasObj = false;
            numClass = C - 4;
            // 如果 4+1+cls 更合理（cls 在 1~300），可以改成 hasObj
            if (C - 5 >= 1 && C - 5 <= 300 && (C - 4 > 300)) {
                hasObj = true; numClass = C - 5;
            }
        }

        // 如果你不是 COCO 80 类，这里会变；Swift 端 label 显示也要换
        _numClass = numClass;

        for (int i = 0; i < ov.n; ++i) {
            float cx = ov.at(i, 0);
            float cy = ov.at(i, 1);
            float w  = ov.at(i, 2);
            float h  = ov.at(i, 3);

            // 兼容归一化输出
            if (cx <= 1.5f && cy <= 1.5f && w <= 1.5f && h <= 1.5f) {
                cx *= _inW; cy *= _inH;
                w  *= _inW; h  *= _inH;
            }

            float bestS = 0.f;
            int bestC = -1;

            if (hasObj) {
                float obj = sigmoid_safe(ov.at(i, 4));
                for (int c = 0; c < numClass; ++c) {
                    float cls = sigmoid_safe(ov.at(i, 5 + c));
                    float s = obj * cls;
                    if (s > bestS) { bestS = s; bestC = c; }
                }
            } else {
                for (int c = 0; c < numClass; ++c) {
                    float s = sigmoid_safe(ov.at(i, 4 + c));
                    if (s > bestS) { bestS = s; bestC = c; }
                }
            }

            if (bestC < 0 || bestS < CONF_THR) continue;

            float x1 = cx - w * 0.5f;
            float y1 = cy - h * 0.5f;
            float x2 = cx + w * 0.5f;
            float y2 = cy + h * 0.5f;

            x1 = std::max(0.f, x1); y1 = std::max(0.f, y1);
            x2 = std::min((float)_inW, x2); y2 = std::min((float)_inH, y2);

            float bw = x2 - x1, bh = y2 - y1;
            if (bw < MIN_BOX_PX || bh < MIN_BOX_PX) continue;

            cands.push_back(BoxInfo{x1,y1,x2,y2,bestS,bestC});
        }
    }

    if (cands.empty()) return @[];

    // 5) pre-nms topK
    std::sort(cands.begin(), cands.end(), [](const BoxInfo& a, const BoxInfo& b){ return a.score > b.score; });
    if ((int)cands.size() > PRE_NMS_TOPK) cands.resize(PRE_NMS_TOPK);

    // 6) per-class topK + per-class NMS
    int numClassForBucket = std::max(_numClass, 1);
    std::vector<std::vector<BoxInfo>> buckets(numClassForBucket);
    for (auto& b : cands) {
        if (b.label >= 0 && b.label < numClassForBucket) buckets[b.label].push_back(b);
    }

    std::vector<BoxInfo> merged;
    merged.reserve(256);

    for (int c = 0; c < numClassForBucket; ++c) {
        auto& v = buckets[c];
        if (v.empty()) continue;

        std::sort(v.begin(), v.end(), [](const BoxInfo& a, const BoxInfo& b){ return a.score > b.score; });
        if ((int)v.size() > PER_CLASS_TOPK) v.resize(PER_CLASS_TOPK);

        nms_per_class(v, NMS_THR_CLASS);
        merged.insert(merged.end(), v.begin(), v.end());
    }

    if (merged.empty()) return @[];

    // 7) optional agnostic NMS + final topN
    if (ENABLE_AGN_NMS) {
        nms_agnostic(merged, NMS_THR_AGN);
    }
    std::sort(merged.begin(), merged.end(), [](const BoxInfo& a, const BoxInfo& b){ return a.score > b.score; });
    if ((int)merged.size() > FINAL_TOPN) merged.resize(FINAL_TOPN);

    // 8) pack output (normalize to 0~1 in input space)
    NSArray<NSString*>* labels = CocoLabels();
    NSMutableArray* arr = [NSMutableArray arrayWithCapacity:merged.size()];

    for (const auto& b : merged) {
        float w = std::max(0.f, b.x2 - b.x1);
        float h = std::max(0.f, b.y2 - b.y1);

        NSString* name = (b.label >= 0 && b.label < (int)labels.count)
            ? labels[b.label]
            : [NSString stringWithFormat:@"cls_%d", b.label];

        [arr addObject:@{
            @"x": @(b.x1 / (float)_inW),
            @"y": @(b.y1 / (float)_inH),
            @"w": @(w    / (float)_inW),
            @"h": @(h    / (float)_inH),
            @"label": name,
            @"score": @(std::min(std::max(b.score, 0.f), 1.f))
        }];
    }

    return arr;
}

@end
