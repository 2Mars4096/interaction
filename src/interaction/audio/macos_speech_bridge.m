#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import <Speech/Speech.h>
#include <errno.h>
#include <math.h>
#include <stdlib.h>

static void PrintJSONAndExit(NSDictionary *payload, int statusCode) {
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:payload options:0 error:&error];
    if (data == nil) {
        fprintf(stdout, "{\"status\":\"error\",\"error\":\"json_encode_failed\",\"message\":\"Unable to encode speech helper output.\"}\n");
        exit(1);
    }
    NSString *json = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    fprintf(stdout, "%s\n", json.UTF8String);
    exit(statusCode);
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
        [recorder stop];
        return NO;
    }
    if (![recorder record]) {
        if (error != NULL && *error == nil) {
            *error = [NSError errorWithDomain:@"InteractionSpeech" code:11 userInfo:@{NSLocalizedDescriptionKey : @"Unable to start microphone capture."}];
        }
        [recorder stop];
        return NO;
    }
    @try {
        [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:duration]];
    } @finally {
        [recorder stop];
    }
    return YES;
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

static NSDictionary *TranscribeRecording(NSURL *url, NSString *localeIdentifier, NSTimeInterval durationSeconds) {
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
    BOOL usedOnDevice = NO;
    if (@available(macOS 10.15, *)) {
        if (recognizer.supportsOnDeviceRecognition) {
            request.requiresOnDeviceRecognition = YES;
            usedOnDevice = YES;
        }
    }

    __block NSString *bestTranscript = nil;
    __block float confidence = -1.0f;
    __block NSError *recognitionError = nil;
    __block BOOL completed = NO;
    __block BOOL hadResult = NO;

    SFSpeechRecognitionTask *task = [recognizer recognitionTaskWithRequest:request
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
    }, MAX(15.0, durationSeconds + 8.0));

    if (!completed) {
        [task cancel];
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
        @"provider" : @"macos_speech",
        @"transcript" : bestTranscript,
        @"locale" : localeIdentifier,
        @"duration_s" : @(durationSeconds),
        @"permission_state" : @"granted",
    } mutableCopy];
    if (isfinite(confidence) && confidence >= 0.0f && confidence <= 1.0f) {
        payload[@"confidence"] = @(confidence);
    }
    payload[@"used_on_device"] = @(usedOnDevice);
    return payload;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        double durationSeconds = 4.0;
        NSString *localeIdentifier = @"en-US";
        NSString *recordingPath = nil;

        for (int index = 1; index < argc; index++) {
            NSString *argument = [NSString stringWithUTF8String:argv[index]];
            if ([argument isEqualToString:@"--duration"] && index + 1 < argc) {
                const char *value = argv[++index];
                char *end = NULL;
                errno = 0;
                durationSeconds = strtod(value, &end);
                if (errno == ERANGE || end == value || *end != '\0' || !isfinite(durationSeconds) || durationSeconds <= 0.0 || durationSeconds > 30.0) {
                    PrintJSONAndExit(@{
                        @"status" : @"error", @"error" : @"invalid_capture_arguments",
                        @"message" : @"Duration must be finite, positive, and at most 30 seconds.",
                    }, 2);
                }
                continue;
            }
            if ([argument isEqualToString:@"--locale"] && index + 1 < argc) {
                localeIdentifier = [NSString stringWithUTF8String:argv[++index]];
                continue;
            }
            if ([argument isEqualToString:@"--recording-path"] && index + 1 < argc) {
                recordingPath = [NSString stringWithUTF8String:argv[++index]];
                continue;
            }
            PrintJSONAndExit(@{
                @"status" : @"error", @"error" : @"invalid_capture_arguments",
                @"message" : @"Unknown or incomplete speech capture argument.",
            }, 2);
        }
        localeIdentifier = [localeIdentifier stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
        BOOL parentIsDirectory = NO;
        BOOL parentExists = recordingPath != nil && [[NSFileManager defaultManager]
            fileExistsAtPath:recordingPath.stringByDeletingLastPathComponent isDirectory:&parentIsDirectory];
        if (localeIdentifier.length == 0 || !recordingPath.isAbsolutePath || !parentExists || !parentIsDirectory ||
            [[NSFileManager defaultManager] fileExistsAtPath:recordingPath]) {
            PrintJSONAndExit(@{
                @"status" : @"error", @"error" : @"invalid_capture_arguments",
                @"message" : @"A nonempty locale and a new recording path inside an existing directory are required.",
            }, 2);
        }

        if (!RequestMicrophoneAccess()) {
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

        NSURL *recordingURL = [NSURL fileURLWithPath:recordingPath];
        NSDictionary *payload = nil;
        @try {
            NSError *recordError = nil;
            if (!RecordAudioToURL(recordingURL, durationSeconds, &recordError)) {
                payload = @{
                    @"status" : @"error",
                    @"error" : @"recording_failed",
                    @"message" : recordError.localizedDescription ?: @"Microphone recording failed.",
                };
            } else {
                payload = TranscribeRecording(recordingURL, localeIdentifier, durationSeconds);
            }
        } @catch (NSException *exception) {
            payload = @{
                @"status" : @"error", @"error" : @"native_capture_failed",
                @"message" : exception.reason ?: @"Native speech capture failed.",
            };
        } @finally {
            [[NSFileManager defaultManager] removeItemAtURL:recordingURL error:nil];
        }
        if ([payload[@"status"] isEqualToString:@"success"]) {
            PrintJSONAndExit(payload, 0);
        }
        PrintJSONAndExit(payload, 1);
    }
}
