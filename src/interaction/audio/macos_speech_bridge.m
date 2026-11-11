#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import <Speech/Speech.h>
#import <dispatch/dispatch.h>
#include <errno.h>
#include <math.h>
#include <signal.h>
#include <stdlib.h>

static BOOL StreamMode = NO;
static NSString *StreamLocale = @"en-US";
static NSUInteger StreamSequence = 0;
static NSUInteger StreamBytes = 0;
static volatile sig_atomic_t StreamInterrupted = 0;

static void InterruptStream(int signalNumber) {
    StreamInterrupted = signalNumber;
}

// Only the main thread emits protocol records. Recognition handlers run on
// the main queue; the audio tap updates locked state and never writes stdout.
static BOOL EmitStreamEvent(NSString *event, NSDictionary *fields) {
    BOOL terminal = [event isEqualToString:@"final"] || [event isEqualToString:@"error"];
    if (StreamSequence >= (terminal ? 1024 : 1023)) {
        return NO;
    }
    NSMutableDictionary *payload = [fields mutableCopy];
    [payload removeObjectForKey:@"status"];
    payload[@"schema_version"] = @1;
    payload[@"sequence"] = @(StreamSequence + 1);
    payload[@"provider"] = @"macos_speech";
    payload[@"locale"] = StreamLocale;
    payload[@"event"] = event;
    NSData *data = [NSJSONSerialization dataWithJSONObject:payload options:0 error:nil];
    // Reserve room for a terminal error if previews reach the total limit.
    NSUInteger byteLimit = terminal ? 1024 * 1024 : 1024 * 1024 - 2048;
    if (data == nil || data.length + 1 > 64 * 1024 || StreamBytes + data.length + 1 > byteLimit) {
        return NO;
    }
    if (fwrite(data.bytes, 1, data.length, stdout) != data.length || fputc('\n', stdout) == EOF || fflush(stdout) != 0) {
        return NO;
    }
    StreamSequence += 1;
    StreamBytes += data.length + 1;
    return YES;
}

static void PrintJSONAndExit(NSDictionary *payload, int statusCode) {
    if (StreamMode) {
        NSString *event = statusCode == 0 ? @"final" : @"error";
        if (!EmitStreamEvent(event, payload)) {
            fprintf(stderr, "Unable to write a bounded speech protocol record.\n");
            exit(1);
        }
        exit(statusCode);
    }
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
    while (!condition() && [deadline timeIntervalSinceNow] > 0 && !(StreamMode && StreamInterrupted)) {
        [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.05]];
    }
}

static BOOL RequestMicrophoneAccess(void) {
    if (@available(macOS 10.14, *)) {
        __block BOOL granted = NO;
        __block BOOL completed = NO;
        [AVCaptureDevice requestAccessForMediaType:AVMediaTypeAudio completionHandler:^(BOOL accessGranted) {
            dispatch_async(dispatch_get_main_queue(), ^{
                granted = accessGranted;
                completed = YES;
            });
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
        dispatch_async(dispatch_get_main_queue(), ^{
            status = authStatus;
            completed = YES;
        });
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

static NSDictionary *StreamError(NSString *code, NSString *message) {
    return @{@"status" : @"error", @"error" : code, @"message" : message};
}

static void PumpStreamRunLoop(void) {
    [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.02]];
}

static NSDictionary *TranscribeStream(NSString *localeIdentifier, NSTimeInterval maxDuration, NSTimeInterval silenceDuration) {
    __attribute__((objc_precise_lifetime)) SFSpeechRecognizer *recognizer = nil;
    AVAudioEngine *engine = nil;
    AVAudioInputNode *input = nil;
    SFSpeechAudioBufferRecognitionRequest *request = nil;
    SFSpeechRecognitionTask *task = nil;
    NSLock *audioLock = [[NSLock alloc] init];
    __block BOOL acceptingAudio = NO;
    __block NSTimeInterval voicedAudio = 0.0;
    __block NSTimeInterval quietAudio = 0.0;
    __block NSString *audioFailure = nil;
    __block NSString *partialTranscript = nil;
    __block NSString *finalTranscript = nil;
    __block float finalConfidence = -1.0f;
    __block NSError *recognitionError = nil;
    __block BOOL receivedFinal = NO;
    __block BOOL recognitionActive = YES;
    BOOL tapInstalled = NO;
    BOOL audioEnded = NO;
    BOOL usedOnDevice = NO;
    NSTimeInterval startedAt = 0.0;
    NSTimeInterval capturedDuration = 0.0;
    NSString *endpoint = nil;
    NSString *lastPreview = nil;
    NSDictionary *payload = nil;

    @try {
        do {
            recognizer = [[SFSpeechRecognizer alloc] initWithLocale:[[NSLocale alloc] initWithLocaleIdentifier:localeIdentifier]];
            if (recognizer == nil || !recognizer.isAvailable) {
                payload = StreamError(@"speech_recognizer_unavailable", @"Speech recognizer is unavailable for the requested locale.");
                break;
            }
            recognizer.queue = [NSOperationQueue mainQueue];
            request = [[SFSpeechAudioBufferRecognitionRequest alloc] init];
            request.shouldReportPartialResults = YES;
            if (@available(macOS 10.15, *)) {
                if (recognizer.supportsOnDeviceRecognition) {
                    request.requiresOnDeviceRecognition = YES;
                    usedOnDevice = YES;
                }
            }
            engine = [[AVAudioEngine alloc] init];
            input = engine.inputNode;
            AVAudioFormat *format = [input outputFormatForBus:0];
            if (format.channelCount == 0 || !isfinite(format.sampleRate) || format.sampleRate <= 0.0) {
                payload = StreamError(@"audio_input_unavailable", @"The microphone did not provide a usable audio format.");
                break;
            }
            // Tap callbacks may run on an audio thread. Serialize appendAudio
            // with stopping input, and protect the energy counters from the
            // main thread. Never stop the engine while holding this lock.
            [input installTapOnBus:0 bufferSize:1024 format:format block:^(AVAudioPCMBuffer *buffer, AVAudioTime *when) {
                @autoreleasepool {
                    [audioLock lock];
                    @try {
                        if (!acceptingAudio) {
                            return;
                        }
                        float * const *channels = buffer.floatChannelData;
                        AVAudioFrameCount frames = buffer.frameLength;
                        AVAudioChannelCount channelCount = buffer.format.channelCount;
                        double rate = buffer.format.sampleRate;
                        if (channels == NULL || channelCount == 0 || !isfinite(rate) || rate <= 0.0) {
                            audioFailure = @"The microphone provided an unsupported audio sample format.";
                            acceptingAudio = NO;
                            return;
                        }
                        if (frames == 0) {
                            return;
                        }
                        double sumSquares = 0.0;
                        for (AVAudioChannelCount channel = 0; channel < channelCount; channel++) {
                            for (AVAudioFrameCount frame = 0; frame < frames; frame++) {
                                double sample = channels[channel][frame * buffer.stride];
                                if (!isfinite(sample)) {
                                    audioFailure = @"The microphone provided invalid audio samples.";
                                    acceptingAudio = NO;
                                    return;
                                }
                                sumSquares += sample * sample;
                            }
                        }
                        double meanSquare = sumSquares / ((double)frames * channelCount);
                        NSTimeInterval bufferDuration = frames / rate;
                        // -45 dBFS, an engineering starting point rather than a
                        // measured speech/noise classifier. Quiet time advances
                        // only when actual quiet audio buffers arrive.
                        if (meanSquare >= pow(10.0, -45.0 / 10.0)) {
                            voicedAudio += bufferDuration;
                            quietAudio = 0.0;
                        } else {
                            quietAudio += bufferDuration;
                        }
                        [request appendAudioPCMBuffer:buffer];
                    } @catch (NSException *exception) {
                        audioFailure = exception.reason ?: @"Unable to stream microphone audio.";
                        acceptingAudio = NO;
                    } @finally {
                        [audioLock unlock];
                    }
                }
            }];
            tapInstalled = YES;
            task = [recognizer recognitionTaskWithRequest:request resultHandler:^(SFSpeechRecognitionResult *result, NSError *error) {
                // SFSpeechRecognizer.queue is explicitly the main queue. These
                // values are never accessed by the audio tap.
                if (!recognitionActive) {
                    return;
                }
                if (error != nil) {
                    recognitionError = error;
                }
                if (result != nil) {
                    if (result.isFinal) {
                        receivedFinal = YES;
                        finalTranscript = [result.bestTranscription.formattedString copy];
                        finalConfidence = AverageConfidence(result.bestTranscription);
                    } else if (result.bestTranscription.formattedString.length > 0) {
                        partialTranscript = [result.bestTranscription.formattedString copy];
                    }
                }
            }];
            if (task == nil) {
                payload = StreamError(@"speech_recognizer_unavailable", @"Unable to start a speech recognition task.");
                break;
            }
            [engine prepare];
            [audioLock lock];
            acceptingAudio = YES;
            [audioLock unlock];
            NSError *startError = nil;
            if (![engine startAndReturnError:&startError]) {
                payload = StreamError(@"recording_failed", startError.localizedDescription ?: @"Unable to start microphone capture.");
                break;
            }
            startedAt = NSProcessInfo.processInfo.systemUptime;
            if (!EmitStreamEvent(@"ready", @{
                @"max_duration_s" : @(maxDuration), @"silence_duration_s" : @(silenceDuration),
                @"permission_state" : @"granted", @"used_on_device" : @(usedOnDevice),
            })) {
                payload = StreamError(@"stream_output_limit", @"Unable to emit the speech stream readiness event.");
                break;
            }

            while (endpoint == nil && payload == nil) {
                NSTimeInterval elapsed = MAX(0.0, NSProcessInfo.processInfo.systemUptime - startedAt);
                [audioLock lock];
                NSString *currentAudioFailure = audioFailure;
                NSTimeInterval currentVoiced = voicedAudio;
                NSTimeInterval currentQuiet = quietAudio;
                [audioLock unlock];
                if (StreamInterrupted) {
                    payload = StreamError(@"capture_cancelled", @"Speech capture was cancelled.");
                } else if (currentAudioFailure != nil) {
                    payload = StreamError(@"audio_stream_failed", currentAudioFailure);
                } else if (recognitionError != nil) {
                    payload = StreamError(@"recognition_failed", recognitionError.localizedDescription ?: @"Speech recognition failed.");
                } else if (elapsed > maxDuration + 0.5) {
                    payload = StreamError(@"capture_timeout", @"Microphone capture exceeded its maximum duration.");
                } else if (receivedFinal) {
                    endpoint = @"recognizer_final";
                } else if (currentVoiced >= 0.15 && currentQuiet >= silenceDuration) {
                    endpoint = @"silence";
                } else if (elapsed >= maxDuration) {
                    // An endAudio-generated final could faithfully transcribe
                    // only a clipped command. Do not turn that into an action.
                    NSString *code = currentVoiced == 0.0 && partialTranscript.length == 0 ? @"empty_transcript" : @"capture_limit_reached";
                    NSString *message = [code isEqualToString:@"empty_transcript"]
                        ? @"No speech was heard before the capture limit."
                        : @"The capture limit arrived before the utterance ended. Repeat with a longer maximum duration.";
                    payload = @{@"status" : @"error", @"error" : code, @"message" : message, @"endpoint_reason" : @"max_duration"};
                }
                if (endpoint != nil || payload != nil) {
                    capturedDuration = elapsed;
                    break;
                }
                if (partialTranscript.length > 0 && ![partialTranscript isEqualToString:lastPreview]) {
                    if (!EmitStreamEvent(@"partial", @{@"transcript" : partialTranscript, @"elapsed_s" : @(elapsed)})) {
                        payload = StreamError(@"stream_output_limit", @"Speech preview output exceeded its limits or could not be written.");
                        break;
                    }
                    lastPreview = partialTranscript;
                }
                PumpStreamRunLoop();
            }
            // Stop accepting buffers before engine shutdown and endAudio, so a
            // late tap callback cannot append after recognition finalization.
            [audioLock lock];
            acceptingAudio = NO;
            [audioLock unlock];
            [engine stop];
            if (tapInstalled) {
                [input removeTapOnBus:0];
                tapInstalled = NO;
            }
            if (payload != nil) {
                break;
            }
            [request endAudio];
            audioEnded = YES;
            NSTimeInterval finalDeadline = startedAt + capturedDuration + 8.0;
            while (!receivedFinal && recognitionError == nil && !StreamInterrupted && NSProcessInfo.processInfo.systemUptime < finalDeadline) {
                NSTimeInterval elapsed = MAX(0.0, NSProcessInfo.processInfo.systemUptime - startedAt);
                if (partialTranscript.length > 0 && ![partialTranscript isEqualToString:lastPreview]) {
                    if (!EmitStreamEvent(@"partial", @{@"transcript" : partialTranscript, @"elapsed_s" : @(elapsed)})) {
                        payload = StreamError(@"stream_output_limit", @"Speech preview output exceeded its limits or could not be written.");
                        break;
                    }
                    lastPreview = partialTranscript;
                }
                PumpStreamRunLoop();
            }
            if (payload != nil) {
                break;
            }
            if (StreamInterrupted) {
                payload = StreamError(@"capture_cancelled", @"Speech capture was cancelled.");
            } else if (recognitionError != nil) {
                payload = StreamError(@"recognition_failed", recognitionError.localizedDescription ?: @"Speech recognition failed.");
            } else if (!receivedFinal) {
                payload = StreamError(@"recognition_timeout", @"Speech recognition did not finalize before its timeout.");
            } else if ([finalTranscript stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet].length == 0) {
                payload = StreamError(@"empty_transcript", @"Speech recognition completed without a transcript.");
            } else {
                id confidence = isfinite(finalConfidence) && finalConfidence >= 0.0f && finalConfidence <= 1.0f ? @(finalConfidence) : NSNull.null;
                payload = @{
                    @"status" : @"success", @"transcript" : finalTranscript, @"confidence" : confidence,
                    @"duration_s" : @(capturedDuration), @"max_duration_s" : @(maxDuration),
                    @"endpoint_reason" : endpoint, @"permission_state" : @"granted", @"used_on_device" : @(usedOnDevice),
                };
            }
        } while (NO);
    } @catch (NSException *exception) {
        payload = StreamError(@"native_capture_failed", exception.reason ?: @"Native speech capture failed.");
    } @finally {
        recognitionActive = NO;
        [audioLock lock];
        acceptingAudio = NO;
        [audioLock unlock];
        // Attempt every cleanup step even if a prior one throws. Only emit the
        // terminal payload after all cleanup has completed.
        @try {
            [engine stop];
        } @catch (NSException *exception) {
            payload = StreamError(@"audio_cleanup_failed", exception.reason ?: @"Unable to stop microphone capture.");
        }
        @try {
            if (tapInstalled) {
                [input removeTapOnBus:0];
            }
        } @catch (NSException *exception) {
            payload = StreamError(@"audio_cleanup_failed", exception.reason ?: @"Unable to remove the microphone tap.");
        }
        @try {
            if (request != nil && !audioEnded) {
                [request endAudio];
            }
        } @catch (NSException *exception) {
            payload = StreamError(@"audio_cleanup_failed", exception.reason ?: @"Unable to finish the microphone audio stream.");
        }
        @try {
            [task cancel];
        } @catch (NSException *exception) {
            payload = StreamError(@"audio_cleanup_failed", exception.reason ?: @"Unable to release speech recognition.");
        }
    }
    return payload ?: StreamError(@"native_capture_failed", @"Native speech capture did not produce a result.");
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        double durationSeconds = 4.0;
        double silenceSeconds = 0.8;
        BOOL silenceProvided = NO;
        NSString *localeIdentifier = @"en-US";
        NSString *recordingPath = nil;

        for (int index = 1; index < argc; index++) {
            NSString *argument = [NSString stringWithUTF8String:argv[index]];
            if ([argument isEqualToString:@"--stream"] && !StreamMode) {
                StreamMode = YES;
                continue;
            }
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
                if (localeIdentifier.length > 0) {
                    StreamLocale = localeIdentifier;
                }
                continue;
            }
            if ([argument isEqualToString:@"--silence-duration"] && index + 1 < argc) {
                const char *value = argv[++index];
                char *end = NULL;
                errno = 0;
                silenceSeconds = strtod(value, &end);
                if (errno == ERANGE || end == value || *end != '\0' || !isfinite(silenceSeconds) || silenceSeconds < 0.2 || silenceSeconds > 5.0) {
                    PrintJSONAndExit(@{
                        @"status" : @"error", @"error" : @"invalid_capture_arguments",
                        @"message" : @"Silence duration must be finite and between 0.2 and 5 seconds.",
                    }, 2);
                }
                silenceProvided = YES;
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
        if (localeIdentifier.length > 0) {
            StreamLocale = localeIdentifier;
        }
        BOOL parentIsDirectory = NO;
        BOOL parentExists = recordingPath != nil && [[NSFileManager defaultManager]
            fileExistsAtPath:recordingPath.stringByDeletingLastPathComponent isDirectory:&parentIsDirectory];
        BOOL invalidRecordedPath = !StreamMode && (!recordingPath.isAbsolutePath || !parentExists || !parentIsDirectory ||
            [[NSFileManager defaultManager] fileExistsAtPath:recordingPath]);
        if (localeIdentifier.length == 0 || invalidRecordedPath || (StreamMode && recordingPath != nil) || (!StreamMode && silenceProvided)) {
            PrintJSONAndExit(@{
                @"status" : @"error", @"error" : @"invalid_capture_arguments",
                @"message" : StreamMode
                    ? @"Stream capture requires a nonempty locale and must not use a recording path."
                    : @"Recorded capture requires a nonempty locale, a new recording path, and no stream-only arguments.",
            }, 2);
        }

        if (StreamMode) {
            signal(SIGINT, InterruptStream);
            signal(SIGTERM, InterruptStream);
            signal(SIGPIPE, SIG_IGN);
        }

        if (!RequestMicrophoneAccess()) {
            PrintJSONAndExit(@{
                @"status" : @"error",
                @"error" : StreamInterrupted ? @"capture_cancelled" : @"microphone_denied",
                @"message" : StreamInterrupted ? @"Speech capture was cancelled." : @"Microphone permission was denied for the speech helper.",
                @"permission_state" : @"denied",
            }, 1);
        }

        SFSpeechRecognizerAuthorizationStatus speechStatus = RequestSpeechAccess();
        if (StreamInterrupted) {
            PrintJSONAndExit(StreamError(@"capture_cancelled", @"Speech capture was cancelled."), 1);
        }
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

        if (StreamMode) {
            NSDictionary *payload = TranscribeStream(localeIdentifier, durationSeconds, silenceSeconds);
            PrintJSONAndExit(payload, [payload[@"status"] isEqualToString:@"success"] ? 0 : 1);
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
