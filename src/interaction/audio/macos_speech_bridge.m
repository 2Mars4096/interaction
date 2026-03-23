#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import <Speech/Speech.h>

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
    }, MAX(15.0, durationSeconds + 8.0));

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
        @"provider" : @"macos_speech",
        @"transcript" : bestTranscript,
        @"locale" : localeIdentifier,
        @"duration_s" : @(durationSeconds),
        @"permission_state" : @"granted",
    } mutableCopy];
    if (confidence >= 0.0f) {
        payload[@"confidence"] = @(confidence);
    }
    payload[@"used_on_device"] = @(usedOnDevice);
    return payload;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        double durationSeconds = 4.0;
        NSString *localeIdentifier = @"en-US";

        for (int index = 1; index < argc; index++) {
            NSString *argument = [NSString stringWithUTF8String:argv[index]];
            if ([argument isEqualToString:@"--duration"] && index + 1 < argc) {
                durationSeconds = [[NSString stringWithUTF8String:argv[++index]] doubleValue];
                continue;
            }
            if ([argument isEqualToString:@"--locale"] && index + 1 < argc) {
                localeIdentifier = [NSString stringWithUTF8String:argv[++index]];
                continue;
            }
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

        NSString *filename = [NSString stringWithFormat:@"interaction-speech-%@.m4a", NSUUID.UUID.UUIDString];
        NSURL *recordingURL = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:filename]];

        NSError *recordError = nil;
        if (!RecordAudioToURL(recordingURL, durationSeconds, &recordError)) {
            PrintJSONAndExit(@{
                @"status" : @"error",
                @"error" : @"recording_failed",
                @"message" : recordError.localizedDescription ?: @"Microphone recording failed.",
            }, 1);
        }

        NSDictionary *payload = TranscribeRecording(recordingURL, localeIdentifier, durationSeconds);
        [[NSFileManager defaultManager] removeItemAtURL:recordingURL error:nil];
        if ([payload[@"status"] isEqualToString:@"success"]) {
            PrintJSONAndExit(payload, 0);
        }
        PrintJSONAndExit(payload, 1);
    }
}
