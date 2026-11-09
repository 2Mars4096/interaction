#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import <QuartzCore/QuartzCore.h>
#import <Speech/Speech.h>
#include <math.h>

static void PrintJSONAndExit(NSDictionary *payload, int statusCode) {
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:payload options:0 error:&error];
    if (data == nil) {
        NSString *fallback = [NSString stringWithFormat:@"{\"status\":\"error\",\"error\":\"json_encode_failed\",\"message\":\"%@\"}", error.localizedDescription ?: @"Unable to encode payload."];
        fprintf(stdout, "%s\n", fallback.UTF8String);
        exit(statusCode);
    }
    NSString *json = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    fprintf(stdout, "%s\n", json.UTF8String);
    exit(statusCode);
}

static void PrintJSONLine(NSDictionary *payload) {
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:payload options:0 error:&error];
    if (data == nil) {
        return;
    }
    NSString *json = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    fprintf(stdout, "%s\n", json.UTF8String);
    fflush(stdout);
}

static double DecibelsForAmplitude(double amplitude) {
    if (!(amplitude > 0.0)) {
        return -160.0;
    }
    return 20.0 * log10(amplitude);
}

static void AddAudioLevelMetrics(NSMutableDictionary *payload, double peakAmplitude, double sumSquares, NSUInteger sampleCount) {
    if (payload == nil) {
        return;
    }
    double rmsAmplitude = 0.0;
    if (sampleCount > 0) {
        rmsAmplitude = sqrt(sumSquares / (double)sampleCount);
    }
    payload[@"audio_sample_count"] = @(sampleCount);
    payload[@"audio_peak_dbfs"] = @(DecibelsForAmplitude(peakAmplitude));
    payload[@"audio_rms_dbfs"] = @(DecibelsForAmplitude(rmsAmplitude));
    payload[@"audio_signal_detected"] = @((peakAmplitude >= 0.01) || (rmsAmplitude >= 0.003));
}

static void RunLoopUntil(BOOL (^condition)(void), NSTimeInterval timeoutSeconds) {
    NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeoutSeconds];
    while (!condition() && [deadline timeIntervalSinceNow] > 0) {
        [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.05]];
    }
}

static BOOL RequestMicrophoneAccess(void) {
    if (@available(macOS 10.14, *)) {
        __block BOOL granted = NO;
        __block BOOL completed = NO;
        [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio completionHandler:^(BOOL accessGranted) {
            granted = accessGranted;
            completed = YES;
        }];
        RunLoopUntil(^BOOL{
            return completed;
        }, 10.0);
        return granted;
    }
    return YES;
}

static SFSpeechRecognizerAuthorizationStatus RequestSpeechAccess(void) {
    __block SFSpeechRecognizerAuthorizationStatus status = SFSpeechRecognizerAuthorizationStatusNotDetermined;
    __block BOOL completed = NO;
    [SFSpeechRecognizer requestAuthorization:^(SFSpeechRecognizerAuthorizationStatus authStatus) {
        status = authStatus;
        completed = YES;
    }];
    RunLoopUntil(^BOOL{
        return completed;
    }, 10.0);
    return status;
}

static BOOL RecordAudioToURL(NSURL *url, NSTimeInterval duration, NSError **error) {
    NSDictionary *settings = @{
        AVFormatIDKey : @(kAudioFormatMPEG4AAC),
        AVSampleRateKey : @16000.0,
        AVNumberOfChannelsKey : @1,
        AVEncoderAudioQualityKey : @(AVAudioQualityMedium),
    };
    AVAudioRecorder *recorder = [[AVAudioRecorder alloc] initWithURL:url settings:settings error:error];
    if (recorder == nil) {
        return NO;
    }
    if (![recorder prepareToRecord]) {
        if (error != NULL && *error == nil) {
            *error = [NSError errorWithDomain:@"InteractionSpeech" code:10 userInfo:@{NSLocalizedDescriptionKey : @"Unable to prepare the microphone recorder."}];
        }
        return NO;
    }
    if (![recorder record]) {
        if (error != NULL && *error == nil) {
            *error = [NSError errorWithDomain:@"InteractionSpeech" code:11 userInfo:@{NSLocalizedDescriptionKey : @"Unable to start microphone capture."}];
        }
        return NO;
    }
    [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:duration]];
    [recorder stop];
    return YES;
}

static NSTimeInterval AudioDurationSeconds(NSURL *url) {
    AVURLAsset *asset = [AVURLAsset URLAssetWithURL:url options:nil];
    CMTime duration = asset.duration;
    if (CMTIME_IS_INVALID(duration) || CMTIME_IS_INDEFINITE(duration)) {
        return 0.0;
    }
    Float64 seconds = CMTimeGetSeconds(duration);
    if (!isfinite(seconds) || seconds < 0.0) {
        return 0.0;
    }
    return seconds;
}

static float AverageConfidence(SFTranscription *transcription) {
    if (transcription.segments.count == 0) {
        return -1.0f;
    }
    double total = 0.0;
    for (SFTranscriptionSegment *segment in transcription.segments) {
        total += segment.confidence;
    }
    return (float)(total / transcription.segments.count);
}

static float AverageConfidenceForRange(SFTranscription *transcription, NSInteger startIndex, NSInteger endIndexExclusive) {
    NSInteger boundedStart = MAX(0, startIndex);
    NSInteger boundedEnd = MIN((NSInteger)transcription.segments.count, endIndexExclusive);
    if (boundedStart >= boundedEnd) {
        return -1.0f;
    }
    double total = 0.0;
    NSInteger count = 0;
    for (NSInteger index = boundedStart; index < boundedEnd; index++) {
        SFTranscriptionSegment *segment = transcription.segments[index];
        total += segment.confidence;
        count += 1;
    }
    if (count == 0) {
        return -1.0f;
    }
    return (float)(total / count);
}

static NSString *JoinedSegmentText(SFTranscription *transcription, NSInteger startIndex, NSInteger endIndexExclusive) {
    NSInteger boundedStart = MAX(0, startIndex);
    NSInteger boundedEnd = MIN((NSInteger)transcription.segments.count, endIndexExclusive);
    if (boundedStart >= boundedEnd) {
        return @"";
    }
    NSMutableArray<NSString *> *parts = [NSMutableArray array];
    for (NSInteger index = boundedStart; index < boundedEnd; index++) {
        SFTranscriptionSegment *segment = transcription.segments[index];
        if (segment.substring.length > 0) {
            [parts addObject:segment.substring];
        }
    }
    return [parts componentsJoinedByString:@" "];
}

static NSDictionary *TranscribeRecording(NSURL *url, NSString *localeIdentifier, NSTimeInterval durationSeconds, NSString *recordingPath, NSString *providerName) {
    NSLocale *locale = [[NSLocale alloc] initWithLocaleIdentifier:localeIdentifier];
    SFSpeechRecognizer *recognizer = [[SFSpeechRecognizer alloc] initWithLocale:locale];
    if (recognizer == nil) {
        return @{
            @"status" : @"error",
            @"error" : @"speech_recognizer_unavailable",
            @"message" : @"Speech recognizer is unavailable for the requested locale.",
            @"locale" : localeIdentifier,
        };
    }

    SFSpeechURLRecognitionRequest *request = [[SFSpeechURLRecognitionRequest alloc] initWithURL:url];
    request.taskHint = SFSpeechRecognitionTaskHintDictation;
    BOOL usedOnDevice = NO;

    __block NSString *bestTranscript = nil;
    __block float confidence = -1.0f;
    __block NSError *recognitionError = nil;
    __block BOOL completed = NO;
    __block BOOL hadResult = NO;

    __unused SFSpeechRecognitionTask *task = [recognizer recognitionTaskWithRequest:request
                                                                       resultHandler:^(SFSpeechRecognitionResult * _Nullable result, NSError * _Nullable error) {
        if (result != nil) {
            hadResult = YES;
            if (result.bestTranscription.formattedString.length > 0) {
                bestTranscript = result.bestTranscription.formattedString;
                confidence = AverageConfidence(result.bestTranscription);
            }
            if (result.isFinal) {
                completed = YES;
            }
        }
        if (error != nil) {
            recognitionError = error;
            completed = YES;
        }
    }];

    RunLoopUntil(^BOOL{
        return completed;
    }, MAX(8.0, durationSeconds + 3.0));

    if (!completed) {
        return @{
            @"status" : @"error",
            @"error" : @"recognition_timeout",
            @"message" : @"Speech recognition did not complete before the timeout.",
            @"locale" : localeIdentifier,
        };
    }

    if (recognitionError != nil) {
        return @{
            @"status" : @"error",
            @"error" : @"recognition_failed",
            @"message" : recognitionError.localizedDescription ?: @"Speech recognition failed.",
            @"locale" : localeIdentifier,
        };
    }

    if (!hadResult || bestTranscript.length == 0) {
        return @{
            @"status" : @"error",
            @"error" : @"empty_transcript",
            @"message" : @"Speech recognition completed without a transcript.",
            @"locale" : localeIdentifier,
        };
    }

    NSMutableDictionary *payload = [@{
        @"status" : @"success",
        @"provider" : providerName ?: @"macos_speech",
        @"transcript" : bestTranscript,
        @"locale" : localeIdentifier,
        @"duration_s" : @(durationSeconds),
        @"permission_state" : @"granted",
    } mutableCopy];
    if (confidence >= 0.0f) {
        payload[@"confidence"] = @(confidence);
    }
    payload[@"used_on_device"] = @(usedOnDevice);
    if (recordingPath.length > 0) {
        payload[@"recording_path"] = recordingPath;
    }
    return payload;
}

static NSDictionary *CaptureStreamingSpeech(NSString *localeIdentifier, NSTimeInterval durationSeconds, NSInteger pauseMs, NSString *recordingPath) {
    NSLocale *locale = [[NSLocale alloc] initWithLocaleIdentifier:localeIdentifier];
    SFSpeechRecognizer *recognizer = [[SFSpeechRecognizer alloc] initWithLocale:locale];
    if (recognizer == nil) {
        return @{
            @"event" : @"final",
            @"status" : @"error",
            @"error" : @"speech_recognizer_unavailable",
            @"message" : @"Speech recognizer is unavailable for the requested locale.",
            @"locale" : localeIdentifier,
        };
    }

    SFSpeechAudioBufferRecognitionRequest *request = [[SFSpeechAudioBufferRecognitionRequest alloc] init];
    request.shouldReportPartialResults = YES;
    request.taskHint = SFSpeechRecognitionTaskHintDictation;
    BOOL usedOnDevice = NO;

    AVAudioEngine *audioEngine = [[AVAudioEngine alloc] init];
    AVAudioInputNode *inputNode = audioEngine.inputNode;
    if (inputNode == nil) {
        return @{
            @"event" : @"final",
            @"status" : @"error",
            @"error" : @"audio_input_unavailable",
            @"message" : @"No audio input node is available for live speech capture.",
            @"locale" : localeIdentifier,
        };
    }

    AVAudioFormat *recordFormat = [inputNode outputFormatForBus:0];
    __block AVAudioFile *debugRecordingFile = nil;
    __block NSError *debugRecordingError = nil;
    __block double observedPeakAmplitude = 0.0;
    __block double observedSumSquares = 0.0;
    __block NSUInteger observedSampleCount = 0;
    if (recordingPath.length > 0) {
        NSURL *recordingURL = [NSURL fileURLWithPath:recordingPath];
        [[NSFileManager defaultManager] createDirectoryAtURL:[recordingURL URLByDeletingLastPathComponent]
                                 withIntermediateDirectories:YES
                                                  attributes:nil
                                                       error:nil];
        debugRecordingFile = [[AVAudioFile alloc] initForWriting:recordingURL settings:recordFormat.settings error:&debugRecordingError];
    }
    [inputNode installTapOnBus:0
                    bufferSize:1024
                        format:recordFormat
                         block:^(AVAudioPCMBuffer *buffer, AVAudioTime *when) {
        (void)when;
        [request appendAudioPCMBuffer:buffer];
        AVAudioFrameCount frameLength = buffer.frameLength;
        AVAudioChannelCount channelCount = buffer.format.channelCount;
        float *const *channelData = buffer.floatChannelData;
        if (channelData != nil && frameLength > 0 && channelCount > 0) {
            float *samples = channelData[0];
            if (samples != nil) {
                for (AVAudioFrameCount frameIndex = 0; frameIndex < frameLength; frameIndex++) {
                    double sample = fabs((double)samples[frameIndex]);
                    if (sample > observedPeakAmplitude) {
                        observedPeakAmplitude = sample;
                    }
                    observedSumSquares += sample * sample;
                }
                observedSampleCount += (NSUInteger)frameLength;
            }
        }
        if (debugRecordingFile != nil) {
            NSError *writeError = nil;
            [debugRecordingFile writeFromBuffer:buffer error:&writeError];
            if (writeError != nil) {
                debugRecordingError = writeError;
                debugRecordingFile = nil;
            }
        }
    }];

    __block NSError *recognitionError = nil;
    __block BOOL completed = NO;
    __block BOOL hadResult = NO;
    __block SFTranscription *latestTranscription = nil;
    __block NSString *latestPartialText = nil;
    __block NSTimeInterval lastTranscriptUpdateTime = CACurrentMediaTime();
    __block NSInteger flushedSegmentCount = 0;
    __block NSInteger clauseIndex = 0;

    NSTimeInterval pauseSeconds = MAX(0.15, ((NSTimeInterval)MAX(0, pauseMs)) / 1000.0);
    __block void (^flushPendingClause)(BOOL) = ^(BOOL forceFlush) {
        if (latestTranscription == nil) {
            return;
        }
        NSInteger segmentCount = (NSInteger)latestTranscription.segments.count;
        if (segmentCount <= flushedSegmentCount) {
            return;
        }
        NSTimeInterval now = CACurrentMediaTime();
        if (!forceFlush && (now - lastTranscriptUpdateTime) < pauseSeconds) {
            return;
        }
        NSString *clauseText = JoinedSegmentText(latestTranscription, flushedSegmentCount, segmentCount);
        clauseText = [clauseText stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
        if (clauseText.length == 0) {
            flushedSegmentCount = segmentCount;
            return;
        }
        float clauseConfidence = AverageConfidenceForRange(latestTranscription, flushedSegmentCount, segmentCount);
        NSMutableDictionary *payload = [@{
            @"event" : @"clause",
            @"clause_index" : @(clauseIndex),
            @"transcript" : clauseText,
            @"locale" : localeIdentifier,
        } mutableCopy];
        if (clauseConfidence >= 0.0f) {
            payload[@"confidence"] = @(clauseConfidence);
        }
        PrintJSONLine(payload);
        flushedSegmentCount = segmentCount;
        clauseIndex += 1;
    };

    __unused SFSpeechRecognitionTask *task = [recognizer recognitionTaskWithRequest:request
                                                                       resultHandler:^(SFSpeechRecognitionResult * _Nullable result, NSError * _Nullable error) {
        if (result != nil) {
            hadResult = YES;
            latestTranscription = result.bestTranscription;
            NSString *partialText = result.bestTranscription.formattedString ?: @"";
            partialText = [partialText stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
            if (partialText.length > 0 && (latestPartialText == nil || ![latestPartialText isEqualToString:partialText])) {
                latestPartialText = partialText;
                lastTranscriptUpdateTime = CACurrentMediaTime();
                float partialConfidence = AverageConfidence(result.bestTranscription);
                NSMutableDictionary *payload = [@{
                    @"event" : @"partial",
                    @"transcript" : partialText,
                    @"locale" : localeIdentifier,
                } mutableCopy];
                if (partialConfidence >= 0.0f) {
                    payload[@"confidence"] = @(partialConfidence);
                }
                PrintJSONLine(payload);
            }
            if (result.isFinal) {
                completed = YES;
            }
        }
        if (error != nil) {
            recognitionError = error;
            completed = YES;
        }
    }];

    NSError *audioError = nil;
    if (![audioEngine startAndReturnError:&audioError]) {
        [inputNode removeTapOnBus:0];
        NSMutableDictionary *payload = [@{
            @"event" : @"final",
            @"status" : @"error",
            @"error" : @"recording_failed",
            @"message" : audioError.localizedDescription ?: @"Unable to start live microphone capture.",
            @"locale" : localeIdentifier,
            @"recording_path" : recordingPath ?: @"",
        } mutableCopy];
        AddAudioLevelMetrics(payload, observedPeakAmplitude, observedSumSquares, observedSampleCount);
        return payload;
    }

    NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:durationSeconds];
    while ([deadline timeIntervalSinceNow] > 0 && !completed) {
        flushPendingClause(NO);
        [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.05]];
    }

    [audioEngine stop];
    [inputNode removeTapOnBus:0];
    [request endAudio];
    RunLoopUntil(^BOOL{
        return completed;
    }, MAX(3.0, pauseSeconds + 1.0));

    if (recognitionError != nil) {
        NSMutableDictionary *payload = [@{
            @"event" : @"final",
            @"status" : @"error",
            @"error" : @"recognition_failed",
            @"message" : recognitionError.localizedDescription ?: @"Speech recognition failed.",
            @"locale" : localeIdentifier,
            @"recording_path" : recordingPath ?: @"",
        } mutableCopy];
        AddAudioLevelMetrics(payload, observedPeakAmplitude, observedSumSquares, observedSampleCount);
        return payload;
    }

    if (!completed && !hadResult) {
        NSMutableDictionary *payload = [@{
            @"event" : @"final",
            @"status" : @"error",
            @"error" : @"recognition_timeout",
            @"message" : @"Streaming speech recognition did not complete before the timeout.",
            @"locale" : localeIdentifier,
            @"recording_path" : recordingPath ?: @"",
        } mutableCopy];
        AddAudioLevelMetrics(payload, observedPeakAmplitude, observedSumSquares, observedSampleCount);
        return payload;
    }

    if (!hadResult || latestTranscription.formattedString.length == 0) {
        NSMutableDictionary *payload = [@{
            @"event" : @"final",
            @"status" : @"error",
            @"error" : @"empty_transcript",
            @"message" : @"Streaming speech recognition completed without a transcript.",
            @"locale" : localeIdentifier,
            @"recording_path" : recordingPath ?: @"",
        } mutableCopy];
        AddAudioLevelMetrics(payload, observedPeakAmplitude, observedSumSquares, observedSampleCount);
        return payload;
    }

    flushPendingClause(YES);

    NSMutableDictionary *payload = [@{
        @"event" : @"final",
        @"status" : @"success",
        @"provider" : @"macos_streaming_speech",
        @"transcript" : latestTranscription.formattedString,
        @"locale" : localeIdentifier,
        @"duration_s" : @(durationSeconds),
        @"permission_state" : @"granted",
        @"used_on_device" : @(usedOnDevice),
        @"clause_count" : @(clauseIndex),
    } mutableCopy];
    float confidence = AverageConfidence(latestTranscription);
    if (confidence >= 0.0f) {
        payload[@"confidence"] = @(confidence);
    }
    if (recordingPath.length > 0) {
        payload[@"recording_path"] = recordingPath;
    }
    if (debugRecordingError != nil) {
        payload[@"recording_error"] = debugRecordingError.localizedDescription ?: @"Audio recording failed.";
    }
    AddAudioLevelMetrics(payload, observedPeakAmplitude, observedSumSquares, observedSampleCount);
    return payload;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        double durationSeconds = 4.0;
        NSString *localeIdentifier = @"en-US";
        BOOL streamingMode = NO;
        NSInteger pauseMs = 700;
        NSString *recordingPath = nil;
        NSString *audioFilePath = nil;

        for (int index = 1; index < argc; index++) {
            NSString *argument = [NSString stringWithUTF8String:argv[index]];
            if ([argument isEqualToString:@"--streaming"]) {
                streamingMode = YES;
                continue;
            }
            if ([argument isEqualToString:@"--duration"] && index + 1 < argc) {
                durationSeconds = [[NSString stringWithUTF8String:argv[++index]] doubleValue];
                continue;
            }
            if ([argument isEqualToString:@"--locale"] && index + 1 < argc) {
                localeIdentifier = [NSString stringWithUTF8String:argv[++index]];
                continue;
            }
            if ([argument isEqualToString:@"--pause-ms"] && index + 1 < argc) {
                pauseMs = [[NSString stringWithUTF8String:argv[++index]] integerValue];
                continue;
            }
            if ([argument isEqualToString:@"--record-path"] && index + 1 < argc) {
                recordingPath = [NSString stringWithUTF8String:argv[++index]];
                continue;
            }
            if ([argument isEqualToString:@"--audio-file"] && index + 1 < argc) {
                audioFilePath = [NSString stringWithUTF8String:argv[++index]];
                continue;
            }
        }

        if (audioFilePath.length > 0 && streamingMode) {
            PrintJSONAndExit(@{
                @"status" : @"error",
                @"error" : @"unsupported_mode",
                @"message" : @"Streaming transcription is not supported for prerecorded audio files.",
            }, 1);
        }

        if (audioFilePath.length == 0 && !RequestMicrophoneAccess()) {
            PrintJSONAndExit(@{
                @"status" : @"error",
                @"error" : @"microphone_denied",
                @"message" : @"Microphone permission was denied for the speech helper.",
                @"permission_state" : @"denied",
            }, 1);
        }

        SFSpeechRecognizerAuthorizationStatus speechStatus = RequestSpeechAccess();
        if (speechStatus != SFSpeechRecognizerAuthorizationStatusAuthorized) {
            NSString *message = @"Speech recognition permission was not granted.";
            if (speechStatus == SFSpeechRecognizerAuthorizationStatusDenied) {
                message = @"Speech recognition permission was denied.";
            } else if (speechStatus == SFSpeechRecognizerAuthorizationStatusRestricted) {
                message = @"Speech recognition is restricted on this device.";
            }
            PrintJSONAndExit(@{
                @"status" : @"error",
                @"error" : @"speech_permission_denied",
                @"message" : message,
                @"permission_state" : @"denied",
            }, 1);
        }

        NSDictionary *payload = nil;
        if (streamingMode) {
            payload = CaptureStreamingSpeech(localeIdentifier, durationSeconds, pauseMs, recordingPath);
        } else if (audioFilePath.length > 0) {
            NSURL *audioURL = [NSURL fileURLWithPath:audioFilePath];
            if (![[NSFileManager defaultManager] fileExistsAtPath:audioFilePath]) {
                PrintJSONAndExit(@{
                    @"status" : @"error",
                    @"error" : @"audio_file_missing",
                    @"message" : @"The requested audio file does not exist.",
                    @"recording_path" : audioFilePath,
                }, 1);
            }
            NSTimeInterval resolvedDuration = AudioDurationSeconds(audioURL);
            if (!(resolvedDuration > 0.0)) {
                resolvedDuration = durationSeconds;
            }
            payload = TranscribeRecording(audioURL, localeIdentifier, resolvedDuration, audioFilePath, @"macos_file_speech");
        } else {
            BOOL preserveRecording = recordingPath.length > 0;
            NSString *resolvedRecordingPath = recordingPath;
            if (resolvedRecordingPath.length == 0) {
                NSString *filename = [NSString stringWithFormat:@"interaction-speech-%@.m4a", NSUUID.UUID.UUIDString];
                resolvedRecordingPath = [NSTemporaryDirectory() stringByAppendingPathComponent:filename];
            }
            NSURL *recordingURL = [NSURL fileURLWithPath:resolvedRecordingPath];
            [[NSFileManager defaultManager] createDirectoryAtURL:[recordingURL URLByDeletingLastPathComponent]
                                     withIntermediateDirectories:YES
                                                      attributes:nil
                                                           error:nil];

            NSError *recordError = nil;
            if (!RecordAudioToURL(recordingURL, durationSeconds, &recordError)) {
                PrintJSONAndExit(@{
                    @"status" : @"error",
                    @"error" : @"recording_failed",
                    @"message" : recordError.localizedDescription ?: @"Microphone recording failed.",
                }, 1);
            }

            payload = TranscribeRecording(recordingURL, localeIdentifier, durationSeconds, preserveRecording ? resolvedRecordingPath : nil, @"macos_speech");
            if (!preserveRecording) {
                [[NSFileManager defaultManager] removeItemAtURL:recordingURL error:nil];
            }
        }
        if ([payload[@"status"] isEqualToString:@"success"]) {
            PrintJSONAndExit(payload, 0);
        }
        PrintJSONAndExit(payload, 1);
    }
}
