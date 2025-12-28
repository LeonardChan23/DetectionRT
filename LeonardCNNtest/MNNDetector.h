//
//  MNNDetector.h
//  LeonardCNNtest
//
//  Created by 陳暄暢 on 24/12/2025.
//

#import <Foundation/Foundation.h>
#import <CoreVideo/CoreVideo.h>

NS_ASSUME_NONNULL_BEGIN

@interface MNNDetector : NSObject

- (instancetype)init;

/// 先做“假推理”：吃一帧 pixelBuffer，返回固定 bbox 列表
/// 返回数组元素：@{ @"x":@(0~1), @"y":@(0~1), @"w":@(0~1), @"h":@(0~1), @"label":@"", @"score":@(0~1) }
- (NSArray<NSDictionary *> *)detectWithPixelBuffer:(CVPixelBufferRef)pixelBuffer;

@end

NS_ASSUME_NONNULL_END

