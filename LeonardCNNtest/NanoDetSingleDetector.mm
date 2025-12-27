//
//  NanoDetSingleDetector.m
//  LeonardCNNtest
//
//  Created by 陳暄暢 on 25/12/2025.
//


#import "NanoDetSingleDetector.h"

#import <MNN/Interpreter.hpp>
#import <MNN/Tensor.hpp>
#import <MNN/ImageProcess.hpp>

#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

struct BoxInfo {
    float x1, y1, x2, y2;
    float score;
    int label;
};

struct CenterPrior {
    int x;
    int y;
    int stride;
};

// ---- demo fast_exp/sigmoid/softmax（保持一致）----
static inline float fast_exp(float x) {
    union { uint32_t i; float f; } v{};
    v.i = (1u << 23) * (1.4426950409f * x + 126.93490512f);
    return v.f;
}

static inline void softmax(const float* src, float* dst, int length) {
    float alpha = src[0];
    for (int i = 1; i < length; ++i) alpha = std::max(alpha, src[i]);

    float denom = 0.f;
    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denom += dst[i];
    }
    float inv = denom > 0.f ? 1.f / denom : 0.f;
    for (int i = 0; i < length; ++i) dst[i] *= inv;
}

// demo 的 center priors 生成
static void generate_grid_center_priors(int input_h, int input_w,
                                        const std::vector<int>& strides,
                                        std::vector<CenterPrior>& out) {
    out.clear();
    for (int stride : strides) {
        int feat_w = (int)std::ceil((float)input_w / (float)stride);
        int feat_h = (int)std::ceil((float)input_h / (float)stride);
        for (int y = 0; y < feat_h; y++) {
            for (int x = 0; x < feat_w; x++) {
                out.push_back(CenterPrior{x, y, stride});
            }
        }
    }
}

// demo 的 NMS（带 +1 的面积/交并）
static void nms(std::vector<BoxInfo>& boxes, float nmsThr) {
    std::sort(boxes.begin(), boxes.end(), [](const BoxInfo& a, const BoxInfo& b){ return a.score > b.score; });
    std::vector<float> areas(boxes.size());
    for (int i = 0; i < (int)boxes.size(); ++i) {
        areas[i] = (boxes[i].x2 - boxes[i].x1 + 1.f) * (boxes[i].y2 - boxes[i].y1 + 1.f);
    }
    for (int i = 0; i < (int)boxes.size(); ++i) {
        for (int j = i + 1; j < (int)boxes.size(); ) {
            float xx1 = std::max(boxes[i].x1, boxes[j].x1);
            float yy1 = std::max(boxes[i].y1, boxes[j].y1);
            float xx2 = std::min(boxes[i].x2, boxes[j].x2);
            float yy2 = std::min(boxes[i].y2, boxes[j].y2);
            float w = std::max(0.f, xx2 - xx1 + 1.f);
            float h = std::max(0.f, yy2 - yy1 + 1.f);
            float inter = w * h;
            float ovr = inter / (areas[i] + areas[j] - inter);
            if (ovr >= nmsThr) {
                boxes.erase(boxes.begin() + j);
                areas.erase(areas.begin() + j);
            } else {
                j++;
            }
        }
    }
}

// COCO labels（同你 demo）
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

@implementation NanoDetSingleDetector {
    std::shared_ptr<MNN::Interpreter> _net;
    MNN::Session* _session;
    MNN::Tensor* _input;
    MNN::Tensor* _output;
    std::shared_ptr<MNN::CV::ImageProcess> _pretreat;

    std::vector<int> _strides;
    std::vector<CenterPrior> _centerPriors;

    int _inW, _inH;
    int _numClass;
    int _regMax;

    float _scoreThr;
    float _nmsThr;

    BOOL _ready;
}

- (instancetype)init {
    self = [super init];
    if (!self) return nil;

    _ready = NO;

    // 按 demo 配置
    _inW = 416; _inH = 416;
    _numClass = 80;
    _regMax = 7;
    _strides = {8,16,32,64};

    // demo main.cpp: score=0.45, nms=0.3
    _scoreThr = 0.45f;
    _nmsThr = 0.30f;

    NSString *path = [[NSBundle mainBundle] pathForResource:@"nanodet-plus-m_416_mnn" ofType:@"mnn"];
    if (!path) {
        NSLog(@"[NanoDetSingleDetector] model not found in bundle");
        return self;
    }

    _net.reset(MNN::Interpreter::createFromFile(path.UTF8String));
    if (!_net) {
        NSLog(@"[NanoDetSingleDetector] createFromFile failed");
        return self;
    }

    MNN::ScheduleConfig cfg;
    cfg.type = MNN_FORWARD_CPU;
    cfg.numThread = 4;

    _session = _net->createSession(cfg);
    if (!_session) {
        NSLog(@"[NanoDetSingleDetector] createSession failed");
        return self;
    }

    _input = _net->getSessionInput(_session, nullptr);
    if (!_input) {
        NSLog(@"[NanoDetSingleDetector] getSessionInput failed");
        return self;
    }

    // demo：input_name="data", output_name="output"；你目前模型输出名就是 output
    _output = _net->getSessionOutput(_session, "output");
    if (!_output) {
        NSLog(@"[NanoDetSingleDetector] getSessionOutput('output') failed");
        return self;
    }

    // 固定输入尺寸
    _net->resizeTensor(_input, {1, 3, _inH, _inW});
    _net->resizeSession(_session);

    // 预处理：BGRA -> BGR + mean/norm（和 nanodet_mnn.hpp 一致）
    MNN::CV::ImageProcess::Config icfg;
    ::memset(&icfg, 0, sizeof(icfg));
    icfg.filterType = MNN::CV::BILINEAR;
    icfg.sourceFormat = MNN::CV::BGRA;
    icfg.destFormat = MNN::CV::BGR;

    // mean_vals: {103.53,116.28,123.675}
    icfg.mean[0] = 103.53f;  icfg.mean[1] = 116.28f;  icfg.mean[2] = 123.675f; icfg.mean[3] = 0.f;
    // norm_vals: {0.017429,0.017507,0.017125}
    icfg.normal[0] = 0.017429f; icfg.normal[1] = 0.017507f; icfg.normal[2] = 0.017125f; icfg.normal[3] = 1.f;

    _pretreat.reset(MNN::CV::ImageProcess::create(icfg));
    if (!_pretreat) {
        NSLog(@"[NanoDetSingleDetector] ImageProcess create failed");
        return self;
    }

    // center priors 只生成一次
    generate_grid_center_priors(_inH, _inW, _strides, _centerPriors);
    NSLog(@"[NanoDetSingleDetector] priors=%d", (int)_centerPriors.size()); // 期望 3598

    _net->releaseModel();
    _ready = YES;
    return self;
}

- (NSArray<NSDictionary *> *)detectWithPixelBuffer:(CVPixelBufferRef)pixelBuffer {
    if (!_ready || !pixelBuffer) return @[];

    // 要求 Swift 输入已是 416x416；如果不是也能跑，但建议你按我给的 Swift resize_uniform
    const int srcW = (int)CVPixelBufferGetWidth(pixelBuffer);
    const int srcH = (int)CVPixelBufferGetHeight(pixelBuffer);
    if (srcW != _inW || srcH != _inH) {
        NSLog(@"[NanoDetSingleDetector] warning: pixelBuffer=%dx%d, expect=%dx%d", srcW, srcH, _inW, _inH);
    }

    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    uint8_t* base = (uint8_t*)CVPixelBufferGetBaseAddress(pixelBuffer);
    int stride = (int)CVPixelBufferGetBytesPerRow(pixelBuffer);

    // matrix identity（因为 Swift 已经做了 416x416）
    MNN::CV::Matrix m;
    _pretreat->setMatrix(m);
    _pretreat->convert(base, srcW, srcH, stride, _input);

    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

    // 推理
    _net->runSession(_session);

    // output -> host
    std::shared_ptr<MNN::Tensor> outHost(MNN::Tensor::createHostTensorFromDevice(_output, true));
    const float* pred = outHost->host<float>();
    auto shp = outHost->shape(); // [1,3598,112]

    const int bins = _regMax + 1;          // 8
    const int numCh = _numClass + 4*bins;  // 112

    if (shp.size() != 3 || shp[1] != (int)_centerPriors.size() || shp[2] != numCh) {
        NSLog(@"[NanoDetSingleDetector] unexpected output shape");
        return @[];
    }

    // per-class results
    std::vector<std::vector<BoxInfo>> results;
    results.resize(_numClass);

    float sm[8];

    // demo decode_infer：max score（不 sigmoid），center= x*stride/y*stride
    const int numPoints = (int)_centerPriors.size();
    for (int idx = 0; idx < numPoints; ++idx) {
        const CenterPrior& ct = _centerPriors[idx];
        const float* scores = pred + idx * numCh;

        float bestScore = 0.f;
        int bestLabel = 0;
        for (int c = 0; c < _numClass; ++c) {
            if (scores[c] > bestScore) {
                bestScore = scores[c];
                bestLabel = c;
            }
        }
        if (bestScore <= _scoreThr) continue;

        const float* bboxPred = scores + _numClass; // 32 floats
        float dis[4];

        for (int s = 0; s < 4; ++s) {
            softmax(bboxPred + s*bins, sm, bins);
            float expv = 0.f;
            for (int b = 0; b < bins; ++b) expv += sm[b] * (float)b;
            dis[s] = expv * (float)ct.stride;
        }

        float cx = (float)ct.x * (float)ct.stride;
        float cy = (float)ct.y * (float)ct.stride;

        float x1 = std::max(cx - dis[0], 0.f);
        float y1 = std::max(cy - dis[1], 0.f);
        float x2 = std::min(cx + dis[2], (float)_inW);
        float y2 = std::min(cy + dis[3], (float)_inH);

        results[bestLabel].push_back(BoxInfo{x1,y1,x2,y2,bestScore,bestLabel});
    }

    // NMS + pack output
    NSMutableArray* arr = [NSMutableArray array];
    NSArray<NSString*>* labels = CocoLabels();

    for (int c = 0; c < _numClass; ++c) {
        auto& vec = results[c];
        if (vec.empty()) continue;
        nms(vec, _nmsThr);

        for (const auto& b : vec) {
            float w = std::max(0.f, b.x2 - b.x1);
            float h = std::max(0.f, b.y2 - b.y1);

            NSString* name = (b.label >= 0 && b.label < (int)labels.count) ? labels[b.label] : [NSString stringWithFormat:@"cls_%d", b.label];

            // 输出归一化到 0~1（相对 416x416），方便 Swift 端复用
            NSDictionary* d = @{
                @"x": @(b.x1 / (float)_inW),
                @"y": @(b.y1 / (float)_inH),
                @"w": @(w    / (float)_inW),
                @"h": @(h    / (float)_inH),
                @"label": name,
                @"score": @(b.score)
            };
            [arr addObject:d];
        }
    }

    return arr;
}

@end
