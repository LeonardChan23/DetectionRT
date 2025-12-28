//
//  NanoDetSingleDetector.h
//  LeonardCNNtest
//
//  Created by 陳暄暢 on 25/12/2025.
//

#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface NanoDetSingleDetector : NSObject

/// 默认初始化：从 Bundle 加载 YOLO 的 .mnn 模型（具体文件名在 .mm 里配置）
/// - 建议：yolo11n_640.mnn / yolov5n_640.mnn 等
- (instancetype)init;

/// 模型期望输入尺寸（正方形）
/// - 例如 640 或 416
/// - 由 .mm 在加载模型后从 input tensor shape 推断得到
@property (nonatomic, readonly) int inputSize;

/// 输入：inputSize x inputSize 的 BGRA pixelBuffer（Swift 侧先做 resize_uniform 到这个尺寸）
/// 输出：NSArray<NSDictionary*>
/// 字段固定：
/// - x/y/w/h: 0~1（相对 inputSize×inputSize 输入坐标系）
/// - label: NSString*
/// - score: 0~1
- (NSArray<NSDictionary *> *)detectWithPixelBuffer:(CVPixelBufferRef)pixelBuffer;

@end

NS_ASSUME_NONNULL_END
