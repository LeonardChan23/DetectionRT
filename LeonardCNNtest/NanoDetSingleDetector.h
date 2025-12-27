#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface NanoDetSingleDetector : NSObject

/// 初始化会从 Bundle 加载模型：nanodet-plus-m_416_mnn.mnn（按你现有命名）
- (instancetype)init;

/// 输入：416x416 的 BGRA pixelBuffer（Swift 会先做 resize_uniform）
/// 输出：NSArray<NSDictionary*>，字段：x/y/w/h(0~1，相对 416x416), label, score
- (NSArray<NSDictionary *> *)detectWithPixelBuffer:(CVPixelBufferRef)pixelBuffer;

@end

NS_ASSUME_NONNULL_END
