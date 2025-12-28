//
//  MNNDetector.mm  (YOLO version)
//  LeonardCNNtest
//

#import "MNNDetector.h"

#import <MNN/Interpreter.hpp>
#import <MNN/Tensor.hpp>
#import <MNN/ImageProcess.hpp>
#import <Accelerate/Accelerate.h>

#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>

std::shared_ptr<MNN::Tensor> _outHost;


// -------------------- COCO labels --------------------
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

// -------------------- math helpers --------------------
static inline float sigmoid_safe(float x) {
    // 如果已经在 0..1，直接返回（避免重复 sigmoid）
    if (x >= 0.0f && x <= 1.0f) return x;
    // stable sigmoid
    if (x >= 0.f) { float z = std::exp(-x); return 1.f / (1.f + z); }
    float z = std::exp(x); return z / (1.f + z);
}

struct YoloBox {
    float x1, y1, x2, y2;
    float score;
    int cls;
};

static inline float iou(const YoloBox& a, const YoloBox& b) {
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

static void nms_class_aware(std::vector<YoloBox>& boxes, float thr) {
    std::sort(boxes.begin(), boxes.end(), [](const YoloBox& a, const YoloBox& b){
        return a.score > b.score;
    });
    std::vector<bool> removed(boxes.size(), false);
    std::vector<YoloBox> kept;
    kept.reserve(boxes.size());

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (removed[i]) continue;
        kept.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (removed[j]) continue;
            if (boxes[i].cls != boxes[j].cls) continue;
            if (iou(boxes[i], boxes[j]) > thr) removed[j] = true;
        }
    }
    boxes.swap(kept);
}

// -------------------- detector --------------------
@implementation MNNDetector {
    std::shared_ptr<MNN::Interpreter> _net;
    MNN::Session* _session;
    MNN::Tensor* _input;

    // YOLO 通常只有 1 个输出（或你导出成 1 个输出）
    MNN::Tensor* _output;
    std::string _outputName;

    std::shared_ptr<MNN::CV::ImageProcess> _pretreat;

    BOOL _ready;
    int _inW, _inH;
    int _numClass;

    BOOL _dumpedOnce;
    std::vector<uint8_t> _resizedBuffer;
    std::vector<uint8_t> _canvasBuffer;
    std::vector<float>   _scoresBuffer; // 用于 NMS 排序复用 (可选，先优化大头)
}

// 你把模型文件名改成你自己的
static NSString* const kYoloModelName = @"yolo11n_640";
static NSString* const kYoloModelExt  = @"mnn";

- (instancetype)init {
    self = [super init];
    if (!self) return nil;

    _ready = NO;
    _dumpedOnce = NO;

    _numClass = 80;   // COCO
    _inW = 320;       // 兜底（最终以模型 input shape 为准）
    _inH = 320;

    NSString *path = [[NSBundle mainBundle] pathForResource:kYoloModelName ofType:kYoloModelExt];
    if (!path) {
        NSLog(@"[YOLO-MNN] model not found in bundle: %@.%@", kYoloModelName, kYoloModelExt);
        return self;
    }

    _net.reset(MNN::Interpreter::createFromFile(path.UTF8String));
    if (!_net) {
        NSLog(@"[YOLO-MNN] createFromFile failed");
        return self;
    }
    
    // 开启 Shader 缓存路径
    NSString *cacheDir = [NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES) firstObject];
    // 指定缓存文件名
    NSString *cachePath = [cacheDir stringByAppendingPathComponent:@"mnn_metal_cache.bin"];
    // 调用 API 设置缓存路径
    // 这样 MNN 会尝试从这个文件加载已编译的 Shader
    _net->setCacheFile(cachePath.UTF8String);
    
    MNN::ScheduleConfig cfg;
    cfg.type = MNN_FORWARD_METAL;
    cfg.numThread = 2;

    MNN::BackendConfig backendCfg;
    backendCfg.power = MNN::BackendConfig::Power_High;
    backendCfg.precision = MNN::BackendConfig::Precision_Low; // 或者 Precision_Low 换速度

    cfg.backendConfig = &backendCfg;

    _session = _net->createSession(cfg);
    if (!_session) {
        NSLog(@"[YOLO-MNN] createSession failed");
        return self;
    }

    _input = _net->getSessionInput(_session, nullptr);
    if (!_input) {
        NSLog(@"[YOLO-MNN] getSessionInput failed");
        return self;
    }

    // 尝试从模型读取输入尺寸
    {
        auto s = _input->shape(); // 期望 [1,3,H,W]
        if (s.size() == 4 && s[2] > 0 && s[3] > 0) {
            _inH = s[2];
            _inW = s[3];
        }
    }

    // 固定输入（避免动态形状带来的不确定）
    _net->resizeTensor(_input, {1, 3, _inH, _inW});
    _net->resizeSession(_session);
    
    _net->updateCacheFile(_session);

    // 自动选择输出（取 outputAll 的第一个）
    {
        auto outs = _net->getSessionOutputAll(_session);
        if (outs.empty()) {
            NSLog(@"[YOLO-MNN] getSessionOutputAll empty");
            return self;
        }
        // 记录第一个输出名
        _outputName = outs.begin()->first;
        _output = outs.begin()->second;
        _outHost.reset(new MNN::Tensor(_output, _output->getDimensionType()));
        NSLog(@"[YOLO-MNN] pick output name: %s", _outputName.c_str());
    }

    // 预处理：BGRA -> RGB，mean=0，norm=1/255（YOLO 常规）
    MNN::CV::ImageProcess::Config icfg;
    ::memset(&icfg, 0, sizeof(icfg));
    icfg.filterType = MNN::CV::BILINEAR;
    icfg.sourceFormat = MNN::CV::BGRA;
    icfg.destFormat   = MNN::CV::RGB;

    icfg.mean[0] = 0.f; icfg.mean[1] = 0.f; icfg.mean[2] = 0.f; icfg.mean[3] = 0.f;
    icfg.normal[0] = 1.f/255.f; icfg.normal[1] = 1.f/255.f; icfg.normal[2] = 1.f/255.f; icfg.normal[3] = 1.f;

    _pretreat.reset(MNN::CV::ImageProcess::create(icfg));
    if (!_pretreat) {
        NSLog(@"[YOLO-MNN] ImageProcess create failed");
        return self;
    }

    _net->releaseModel();
    _ready = YES;

    NSLog(@"[YOLO-MNN] ready=YES input=[1,3,%d,%d] output=%s", _inH, _inW, _outputName.c_str());
    return self;
}

- (NSArray<NSDictionary *> *)detectWithPixelBuffer:(CVPixelBufferRef)pixelBuffer {
    if (!_ready || !_session || !_input || !_output || !_pretreat) return @[];
    if (!pixelBuffer) return @[];
    // ------- YOLO decode params（先给你一套“观感更像产品”的默认）-------
    const float SCORE_THR = 0.35f;     // 0.25~0.5 之间调；越大框越少
    const float NMS_THR   = 0.5f;     // 0.4~0.6
    const int   TOPK      = 100;       // 进入 NMS 的候选上限
    const int   MAX_DET   = 15;        // 最终输出上限
//    const uint8_t PAD     = 114;       // letterbox padding

    // ------- 取源帧尺寸 -------
    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    uint8_t* base = (uint8_t*)CVPixelBufferGetBaseAddress(pixelBuffer);
    int srcW = (int)CVPixelBufferGetWidth(pixelBuffer);
    int srcH = (int)CVPixelBufferGetHeight(pixelBuffer);
    int srcStride = (int)CVPixelBufferGetBytesPerRow(pixelBuffer);

    // ------- letterbox 到 _inW x _inH（BGRA canvas）-------
    float scale = std::min((float)_inW / (float)srcW, (float)_inH / (float)srcH);
    int newW = (int)lroundf(srcW * scale);
    int newH = (int)lroundf(srcH * scale);
    int padX = (_inW - newW) / 2;
    int padY = (_inH - newH) / 2;

//    static std::vector<uint8_t> resized;
//    static std::vector<uint8_t> canvas;
//    resized.resize((size_t)newW * (size_t)newH * 4);
//    canvas.resize((size_t)_inW * (size_t)_inH * 4);
    size_t needResized = (size_t)newW * newH * 4;
    size_t needCanvas  = (size_t)_inW * _inH * 4;
        
    if (_resizedBuffer.size() < needResized) _resizedBuffer.resize(needResized);
    if (_canvasBuffer.size() < needCanvas)   _canvasBuffer.resize(needCanvas);

//    vImage_Buffer srcBuf{ base, (vImagePixelCount)srcH, (vImagePixelCount)srcW, (size_t)srcStride };
//    vImage_Buffer dstBuf{ resized.data(), (vImagePixelCount)newH, (vImagePixelCount)newW, (size_t)newW * 4 };
//    vImage_Error err = vImageScale_ARGB8888(&srcBuf, &dstBuf, nullptr, kvImageHighQualityResampling);
// 4. vImage 缩放 (使用 NoFlags 极速模式)
    vImage_Buffer srcBuf = { base, (vImagePixelCount)srcH, (vImagePixelCount)srcW, (size_t)srcStride };
    vImage_Buffer dstBuf = { _resizedBuffer.data(), (vImagePixelCount)newH, (vImagePixelCount)newW, (size_t)newW * 4 };
    vImage_Error err = vImageScale_ARGB8888(&srcBuf, &dstBuf, nullptr, kvImageNoFlags);
    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

    if (err != kvImageNoError) {
        NSLog(@"[YOLO-MNN] vImageScale error=%ld", (long)err);
        return @[];
    }

    // 填充 padding（BGRA）
//    for (size_t i = 0; i < canvas.size(); i += 4) {
//        canvas[i + 0] = PAD;  // B
//        canvas[i + 1] = PAD;  // G
//        canvas[i + 2] = PAD;  // R
//        canvas[i + 3] = 255;  // A
//    }
    ::memset(_canvasBuffer.data(), 114, needCanvas);
//    for (int y = 0; y < newH; ++y) {
//        uint8_t* dstRow = canvas.data() + ((size_t)(y + padY) * _inW + padX) * 4;
//        const uint8_t* srcRow2 = resized.data() + (size_t)y * newW * 4;
//        ::memcpy(dstRow, srcRow2, (size_t)newW * 4);
//    }
    for (int y = 0; y < newH; ++y) {
        uint8_t* dstRow = _canvasBuffer.data() + ((size_t)(y + padY) * _inW + padX) * 4;
        const uint8_t* srcRow = _resizedBuffer.data() + (size_t)y * newW * 4;
        ::memcpy(dstRow, srcRow, (size_t)newW * 4);
    }

    // ------- convert -> _input (BGRA->RGB + 1/255) -------
//    MNN::CV::Matrix m; // identity
//    _pretreat->setMatrix(m);
//    _pretreat->convert(canvas.data(), _inW, _inH, _inW * 4, _input);
    _pretreat->convert(_canvasBuffer.data(), _inW, _inH, _inW * 4, _input);

//    NSTimeInterval t1 = [[NSDate date] timeIntervalSince1970];
    // ------- infer -------
    _net->runSession(_session);
//    NSTimeInterval t2 = [[NSDate date] timeIntervalSince1970];
    
//    printf("Infer time: %.2f ms\n", (t2 - t1) * 1000.0);
    _output->copyToHostTensor(_outHost.get());
    const float* outPtr = _outHost->host<float>();
    auto shp = _outHost->shape();

    // ------- output host -------
    std::shared_ptr<MNN::Tensor> outHost(MNN::Tensor::createHostTensorFromDevice(_output, true));

    if (!_dumpedOnce) {
        _dumpedOnce = YES;
        NSMutableString *ss = [NSMutableString stringWithString:@"["];
        for (int i = 0; i < (int)shp.size(); i++) {
            [ss appendFormat:@"%d%@", shp[i], (i+1<(int)shp.size()?@",":@"")];
        }
        [ss appendString:@"]"];
        NSLog(@"[YOLO-MNN] output shape=%@", ss);
        NSLog(@"[YOLO-MNN] src=%dx%d in=%dx%d scale=%.4f padX=%d padY=%d", srcW, srcH, _inW, _inH, scale, padX, padY);
    }

    // ------- parse output shape -------
    // 支持：
    // A) [1, N, D]
    // B) [1, D, N]
    if (shp.size() != 3 || shp[0] != 1) {
        NSLog(@"[YOLO-MNN] unsupported output shape");
        return @[];
    }

    int dim1 = shp[1];
    int dim2 = shp[2];

    int N = 0;
    int D = 0;
    bool layoutN_D = false; // true: [1,N,D]; false: [1,D,N]

    // 常见 D = 84/85；N 常见 8400/25200
    if (dim2 == 84 || dim2 == 85) {
        // [1,N,D]
        N = dim1; D = dim2; layoutN_D = true;
    } else if (dim1 == 84 || dim1 == 85) {
        // [1,D,N]
        D = dim1; N = dim2; layoutN_D = false;
    } else {
        NSLog(@"[YOLO-MNN] unknown D=%d,%d (expect 84/85).", dim1, dim2);
        return @[];
    }

    auto getv = [&](int i, int k) -> float {
        // i: 0..N-1, k: 0..D-1
        if (layoutN_D) {
            return outPtr[i * D + k];
        } else {
            return outPtr[k * N + i];
        }
    };

    const bool hasObj = (D == 85);

    std::vector<YoloBox> cands;
    cands.reserve(256);

    // 先扫一遍统计一下坐标是否“看起来是归一化”
    // （只用前 64 个样本粗判，避免额外开销）
    bool coordLooksNormalized = true;
    {
        int probe = std::min(N, 64);
        float maxv = 0.f;
        for (int i = 0; i < probe; ++i) {
            float x = std::fabs(getv(i, 0));
            float y = std::fabs(getv(i, 1));
            float w = std::fabs(getv(i, 2));
            float h = std::fabs(getv(i, 3));
            maxv = std::max(maxv, std::max(std::max(x,y), std::max(w,h)));
        }
        // 若明显大于 2，认为是像素坐标
        coordLooksNormalized = (maxv <= 2.0f);
    }

    for (int i = 0; i < N; ++i) {
        float bx = getv(i, 0);
        float by = getv(i, 1);
        float bw = getv(i, 2);
        float bh = getv(i, 3);

        // 坐标归一化 -> 像素
        if (coordLooksNormalized) {
            bx *= (float)_inW;
            by *= (float)_inH;
            bw *= (float)_inW;
            bh *= (float)_inH;
        }

        // YOLO 常见：xywh(center)
        float x1 = bx - bw * 0.5f;
        float y1 = by - bh * 0.5f;
        float x2 = bx + bw * 0.5f;
        float y2 = by + bh * 0.5f;

        // objectness
        float obj = 1.f;
        int clsStart = 4;
        if (hasObj) {
            obj = sigmoid_safe(getv(i, 4));
            clsStart = 5;
        }

        // best class
        float bestC = 0.f;
        int bestId = -1;
        for (int c = 0; c < _numClass; ++c) {
            float sc = sigmoid_safe(getv(i, clsStart + c));
            if (sc > bestC) { bestC = sc; bestId = c; }
        }
        if (bestId < 0) continue;

        float conf = hasObj ? (obj * bestC) : bestC;
        if (conf < SCORE_THR) continue;

        // clamp in input space
        x1 = std::max(0.f, std::min(x1, (float)_inW));
        y1 = std::max(0.f, std::min(y1, (float)_inH));
        x2 = std::max(0.f, std::min(x2, (float)_inW));
        y2 = std::max(0.f, std::min(y2, (float)_inH));
        if (x2 <= x1 || y2 <= y1) continue;

        cands.push_back(YoloBox{x1,y1,x2,y2,conf,bestId});
    }

    if (cands.empty()) return @[];

    // TOPK 截断，加速 NMS
    std::sort(cands.begin(), cands.end(), [](const YoloBox& a, const YoloBox& b){
        return a.score > b.score;
    });
    if ((int)cands.size() > TOPK) cands.resize(TOPK);

    // NMS
    nms_class_aware(cands, NMS_THR);

    std::sort(cands.begin(), cands.end(), [](const YoloBox& a, const YoloBox& b){
        return a.score > b.score;
    });
    if ((int)cands.size() > MAX_DET) cands.resize(MAX_DET);

    // pack output: map back to srcW/srcH, return normalized (relative to original pixelBuffer)
    NSArray<NSString *> *labels = CocoLabels();
    NSMutableArray *arr = [NSMutableArray arrayWithCapacity:cands.size()];

    for (const auto& b : cands) {
        // b is in letterboxed input space (_inW/_inH)
        float x1s = (b.x1 - (float)padX) / scale;
        float y1s = (b.y1 - (float)padY) / scale;
        float x2s = (b.x2 - (float)padX) / scale;
        float y2s = (b.y2 - (float)padY) / scale;

        x1s = std::max(0.f, std::min(x1s, (float)srcW));
        y1s = std::max(0.f, std::min(y1s, (float)srcH));
        x2s = std::max(0.f, std::min(x2s, (float)srcW));
        y2s = std::max(0.f, std::min(y2s, (float)srcH));

        float ws = std::max(0.f, x2s - x1s);
        float hs = std::max(0.f, y2s - y1s);
        if (ws <= 1.f || hs <= 1.f) continue;

        NSString *name = (b.cls >= 0 && b.cls < (int)labels.count)
            ? labels[b.cls]
            : [NSString stringWithFormat:@"cls_%d", b.cls];

        NSDictionary *d = @{
            @"x": @(x1s / (float)srcW),
            @"y": @(y1s / (float)srcH),
            @"w": @(ws  / (float)srcW),
            @"h": @(hs  / (float)srcH),
            @"label": name,
            @"score": @(b.score)
        };
        [arr addObject:d];
    }

    return arr;
}

@end
