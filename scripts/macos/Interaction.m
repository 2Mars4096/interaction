#import <Cocoa/Cocoa.h>
#import <signal.h>
#import <Carbon/Carbon.h>

static OSStatus stopHotKey(EventHandlerCallRef next, EventRef event, void *context);

@interface InteractionSurface : NSView
@end
@implementation InteractionSurface
- (BOOL)isOpaque { return YES; }
- (void)drawRect:(NSRect)dirtyRect {
    [[NSColor colorWithCalibratedRed:0.95 green:0.96 blue:0.97 alpha:1] setFill];
    NSRectFill(dirtyRect);
}
@end

@interface InteractionController : NSObject <NSApplicationDelegate, NSWindowDelegate>
@property NSWindow *window;
@property NSPopUpButton *mode;
@property NSButton *execute;
@property NSButton *start;
@property NSButton *stop;
@property NSButton *calibrate;
@property NSTextField *status;
@property NSTextField *hint;
@property NSTextView *pad;
@property NSTextView *activity;
@property NSTask *task;
@property NSMutableData *pending;
@property NSDictionary *config;
@property NSString *runtime;
@property BOOL stopping;
@property BOOL failed;
@property BOOL quitting;
@property EventHotKeyRef hotKey;
@end

@implementation InteractionController
- (NSTextField *)label:(NSString *)text frame:(NSRect)frame size:(CGFloat)size bold:(BOOL)bold {
    NSTextField *label = [NSTextField labelWithString:text];
    label.frame = frame;
    label.font = bold ? [NSFont boldSystemFontOfSize:size] : [NSFont systemFontOfSize:size];
    label.lineBreakMode = NSLineBreakByWordWrapping;
    [self.window.contentView addSubview:label];
    return label;
}
- (NSButton *)button:(NSString *)text frame:(NSRect)frame action:(SEL)action {
    NSButton *button = [NSButton buttonWithTitle:text target:self action:action];
    button.frame = frame;
    button.bezelStyle = NSBezelStyleRounded;
    [self.window.contentView addSubview:button];
    return button;
}
- (NSTextView *)textArea:(NSRect)frame editable:(BOOL)editable {
    NSScrollView *scroll = [[NSScrollView alloc] initWithFrame:frame];
    scroll.hasVerticalScroller = YES;
    scroll.borderType = NSBezelBorder;
    NSTextView *view = [[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, frame.size.width, frame.size.height)];
    view.editable = editable;
    view.richText = NO;
    view.font = editable ? [NSFont systemFontOfSize:16] : [NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightRegular];
    view.textContainerInset = NSMakeSize(12, 12);
    view.autoresizingMask = NSViewWidthSizable;
    view.textContainer.widthTracksTextView = YES;
    view.automaticQuoteSubstitutionEnabled = NO;
    scroll.documentView = view;
    [self.window.contentView addSubview:scroll];
    return view;
}
- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    NSApp.appearance = [NSAppearance appearanceNamed:NSAppearanceNameAqua];
    NSString *resources = NSBundle.mainBundle.resourcePath;
    NSData *data = [NSData dataWithContentsOfFile:[resources stringByAppendingPathComponent:@"config.json"]];
    self.config = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
    self.runtime = self.config[@"runtime"] ?: [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Application Support/Interaction"];
    self.window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 740, 730)
         styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable
         backing:NSBackingStoreBuffered defer:NO];
    self.window.contentView = [[InteractionSurface alloc] initWithFrame:NSMakeRect(0, 0, 740, 730)];
    self.window.title = @"Interaction";
    self.window.delegate = self;
    self.window.releasedWhenClosed = NO;
    [self.window center];
    [self label:@"Interaction" frame:NSMakeRect(28, 666, 500, 38) size:30 bold:YES];
    [self label:@"Mode" frame:NSMakeRect(30, 593, 60, 24) size:13 bold:YES];
    self.mode = [[NSPopUpButton alloc] initWithFrame:NSMakeRect(90, 588, 250, 32) pullsDown:NO];
    [self.mode addItemsWithTitles:@[@"Voice commands", @"Dictation", @"Gaze cursor", @"Gaze + voice"]];
    self.mode.target = self;
    self.mode.action = @selector(modeChanged:);
    [self.window.contentView addSubview:self.mode];
    self.execute = [NSButton checkboxWithTitle:@"Control my Mac" target:self action:@selector(modeChanged:)];
    self.execute.frame = NSMakeRect(400, 590, 250, 28);
    self.execute.state = NSControlStateValueOff;
    [self.window.contentView addSubview:self.execute];
    self.hint = [self label:@"" frame:NSMakeRect(30, 535, 680, 44) size:13 bold:NO];
    self.hint.textColor = NSColor.secondaryLabelColor;
    self.start = [self button:@"Start" frame:NSMakeRect(26, 484, 118, 38) action:@selector(startPressed:)];
    self.start.keyEquivalent = @"\r";
    self.start.bezelColor = NSColor.controlAccentColor;
    self.stop = [self button:@"Stop" frame:NSMakeRect(150, 484, 110, 38) action:@selector(stopPressed:)];
    self.stop.enabled = NO;
    self.stop.keyEquivalent = @".";
    self.stop.keyEquivalentModifierMask = NSEventModifierFlagCommand;
    self.calibrate = [self button:@"Calibrate gaze…" frame:NSMakeRect(530, 484, 180, 38) action:@selector(calibratePressed:)];
    self.status = [self label:@"Checking setup…" frame:NSMakeRect(30, 431, 680, 42) size:14 bold:YES];
    [self label:@"Test pad" frame:NSMakeRect(30, 397, 200, 24) size:14 bold:YES];
    self.pad = [self textArea:NSMakeRect(30, 244, 680, 145) editable:YES];
    self.pad.accessibilityLabel = @"Dictation test pad";
    [self label:@"Activity" frame:NSMakeRect(30, 208, 140, 24) size:13 bold:YES];
    self.activity = [self textArea:NSMakeRect(30, 70, 680, 131) editable:NO];
    self.activity.accessibilityLabel = @"Session activity";
    [self button:@"Privacy & access…" frame:NSMakeRect(26, 20, 155, 32) action:@selector(permissionsPressed:)];
    [self button:@"Show logs" frame:NSMakeRect(184, 20, 120, 32) action:@selector(logsPressed:)];
    EventTypeSpec hotKeyEvent = {kEventClassKeyboard, kEventHotKeyPressed};
    InstallApplicationEventHandler(&stopHotKey, 1, &hotKeyEvent, (__bridge void *)self, NULL);
    EventHotKeyID hotKeyID = {'INTR', 1};
    EventHotKeyRef hotKey = NULL;
    OSStatus shortcutStatus = RegisterEventHotKey(53, cmdKey | shiftKey, hotKeyID, GetApplicationEventTarget(), 0, &hotKey);
    self.hotKey = hotKey;
    NSString *shortcutText = shortcutStatus == noErr ? @"⌘⇧Esc stops from any app." : @"Use Stop, or ⌘. while this window is active.";
    [self label:shortcutText frame:NSMakeRect(322, 24, 390, 22) size:11 bold:NO];
    [self modeChanged:nil];
    [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
    BOOL stopTest = [NSProcessInfo.processInfo.arguments containsObject:@"--smoke-test-stop"];
    [self runMode:@"check" delay:stopTest ? 30 : 0];
    if (stopTest) dispatch_after(dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC), dispatch_get_main_queue(), ^{ [self stopPressed:nil]; });
    // Used only by the local build smoke test; never starts a device.
    if ([NSProcessInfo.processInfo.arguments containsObject:@"--smoke-test"] || stopTest) {
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 3*NSEC_PER_SEC), dispatch_get_main_queue(), ^{
            NSString *marker = [self.runtime stringByAppendingPathComponent:@"desktop-smoke.json"];
            NSDictionary *result = @{@"windowVisible": @(self.window.visible), @"setupReady": @(!self.failed && !self.task), @"status": self.status.stringValue, @"startEnabled": @(self.start.enabled), @"stopEnabled": @(self.stop.enabled), @"previewDefault": @(self.execute.state == NSControlStateValueOff), @"windowNumber": @(self.window.windowNumber), @"globalStopAvailable": @(self.hotKey != NULL)};
            NSData *json = [NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:nil];
            [json writeToFile:marker atomically:YES];
            NSView *view = self.window.contentView;
            NSBitmapImageRep *bitmap = [view bitmapImageRepForCachingDisplayInRect:view.bounds];
            [view cacheDisplayInRect:view.bounds toBitmapImageRep:bitmap];
            // AppKit view snapshots have transparent backing; flatten against
            // the actual window surface so offscreen review matches the app.
            NSImage *snapshot = [[NSImage alloc] initWithSize:view.bounds.size];
            [snapshot lockFocus];
            [[NSColor colorWithCalibratedWhite:0.93 alpha:1] setFill];
            NSRectFill(view.bounds);
            [bitmap drawInRect:view.bounds];
            [snapshot unlockFocus];
            NSBitmapImageRep *flat = [[NSBitmapImageRep alloc] initWithData:snapshot.TIFFRepresentation];
            [[flat representationUsingType:NSBitmapImageFileTypePNG properties:@{}]
                writeToFile:[self.runtime stringByAppendingPathComponent:@"desktop-preview.png"] atomically:YES];
            [NSApp terminate:nil];
        });
    }
}
- (void)modeChanged:(id)sender {
    NSArray *hints = @[@"Say “open Safari”, “scroll down”, or “type hello”. Confirm with “yes” or “cancel”.",
                      @"Click the test pad or another text field after Start. Edit corrections manually.",
                      @"Calibrate first. Look at large targets; Stop ends cursor tracking.",
                      @"Calibrate first. Look at a target and say “click this”; keep looking while confirming."];
    self.hint.stringValue = hints[self.mode.indexOfSelectedItem];
    self.calibrate.hidden = self.mode.indexOfSelectedItem < 2;
    if (!self.task) self.status.stringValue = self.execute.state == NSControlStateValueOn ? @"Ready · live control enabled" : @"Ready · preview only";
}
- (void)setRunning:(BOOL)running {
    self.start.enabled = !running;
    self.mode.enabled = !running;
    self.execute.enabled = !running;
    self.calibrate.enabled = !running;
    self.stop.enabled = running;
}
- (void)startPressed:(id)sender {
    NSArray *modes = @[@"voice", @"dictation", @"gaze", @"fusion"];
    NSString *mode = modes[self.mode.indexOfSelectedItem];
    if ([mode isEqual:@"gaze"] || [mode isEqual:@"fusion"]) {
        NSString *profile = [self.runtime stringByAppendingPathComponent:@"profiles/webcam-live.json"];
        if (![NSFileManager.defaultManager fileExistsAtPath:profile]) {
            self.status.stringValue = @"Calibrate gaze first, then start this mode.";
            return;
        }
    }
    [self runMode:mode delay:4];
    if ([mode isEqual:@"dictation"]) [self.window makeFirstResponder:self.pad];
}
- (void)calibratePressed:(id)sender {
    self.status.stringValue = @"Look at each dot; press Space or click the dot to capture.";
    [self runMode:@"calibrate" delay:1];
}
- (void)appendLine:(NSString *)line {
    if (!line.length || [line hasPrefix:@"[voice] mic:"]) return;
    NSString *message = line;
    for (NSString *prefix in @[@"[interaction] ", @"[voice] ", @"[fusion] ", @"[gaze] "]) {
        if ([message hasPrefix:prefix]) { message = [message substringFromIndex:prefix.length]; break; }
    }
    if ([message hasPrefix:@"partial: "]) message = [@"Heard: " stringByAppendingString:[message substringFromIndex:9]];
    else if ([message hasPrefix:@"final: "]) message = [@"Heard: " stringByAppendingString:[message substringFromIndex:7]];
    if ([line hasPrefix:@"READY\t"]) return;
    if ([line hasPrefix:@"STATUS\t"]) message = [line substringFromIndex:7];
    if ([line hasPrefix:@"ERROR\t"]) { message = [line substringFromIndex:6]; self.failed = YES; }
    if (self.stopping) return;
    if ([message isEqual:@"Ready. Choose a mode and press Start."]) {
        self.status.stringValue = @"Ready · preview only";
        return;
    }
    self.status.stringValue = message.length > 180 ? [[message substringToIndex:180] stringByAppendingString:@"…"] : message;
    NSString *entry = [NSString stringWithFormat:@"%@\n", message];
    [self.activity.textStorage appendAttributedString:[[NSAttributedString alloc] initWithString:entry attributes:@{NSFontAttributeName:[NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightRegular], NSForegroundColorAttributeName:NSColor.textColor}]];
    if (self.activity.string.length > 24000) [self.activity.textStorage deleteCharactersInRange:NSMakeRange(0, self.activity.string.length - 18000)];
    [self.activity scrollRangeToVisible:NSMakeRange(self.activity.string.length, 0)];
}
- (void)receiveData:(NSData *)data {
    [self.pending appendData:data];
    const unsigned char *bytes = self.pending.bytes;
    NSUInteger consumed = 0;
    for (NSUInteger i=0; i<self.pending.length; i++) {
        if (bytes[i] == '\n') {
            NSData *part = [self.pending subdataWithRange:NSMakeRange(consumed, i-consumed)];
            NSString *line = [[NSString alloc] initWithData:part encoding:NSUTF8StringEncoding];
            if (line) [self appendLine:line];
            consumed = i+1;
        }
    }
    if (consumed) [self.pending replaceBytesInRange:NSMakeRange(0, consumed) withBytes:NULL length:0];
}
- (void)runMode:(NSString *)mode delay:(int)delay {
    if (self.task) return;
    NSString *python = self.config[@"python"];
    if (!python || ![NSFileManager.defaultManager isExecutableFileAtPath:python]) {
        self.failed = YES;
        self.status.stringValue = @"Python environment is missing. Rebuild this local test app.";
        return;
    }
    [NSFileManager.defaultManager createDirectoryAtPath:self.runtime withIntermediateDirectories:YES attributes:nil error:nil];
    [NSFileManager.defaultManager createDirectoryAtPath:[self.runtime stringByAppendingPathComponent:@"logs"] withIntermediateDirectories:YES attributes:nil error:nil];
    self.failed = NO;
    self.stopping = NO;
    self.pending = [NSMutableData data];
    NSTask *task = [NSTask new];
    task.executableURL = [NSURL fileURLWithPath:python];
    NSMutableArray *args = [@[@"-B", @"-u", @"-m", @"interaction.desktop_worker", @"--mode", mode,
                             @"--runtime-dir", self.runtime, @"--delay", [NSString stringWithFormat:@"%d",delay]] mutableCopy];
    if (self.execute.state == NSControlStateValueOn) [args addObject:@"--execute"];
    task.arguments = args;
    NSMutableDictionary *env = [NSProcessInfo.processInfo.environment mutableCopy];
    env[@"PYTHONPATH"] = [NSBundle.mainBundle.resourcePath stringByAppendingPathComponent:@"engine"];
    env[@"PYTHONUNBUFFERED"] = @"1";
    env[@"PYTHONDONTWRITEBYTECODE"] = @"1";
    task.environment = env;
    task.currentDirectoryURL = [NSURL fileURLWithPath:self.runtime];
    task.standardInput = [NSFileHandle fileHandleWithNullDevice];
    NSPipe *pipe = [NSPipe pipe];
    task.standardOutput = pipe;
    task.standardError = pipe;
    NSError *error;
    if (![task launchAndReturnError:&error]) { self.failed=YES; self.status.stringValue=error.localizedDescription; return; }
    self.task = task;
    [self setRunning:YES];
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0), ^{
        @try {
            NSData *data;
            while ((data = [pipe.fileHandleForReading availableData]).length) {
                NSData *copy = [data copy];
                dispatch_async(dispatch_get_main_queue(), ^{ if (self.task == task) [self receiveData:copy]; });
            }
        } @catch (NSException *exception) { }
        [task waitUntilExit];
        dispatch_async(dispatch_get_main_queue(), ^{
            if (self.task != task) return;
            // A worker must not leave a capture helper behind after exiting.
            kill(-task.processIdentifier, SIGTERM);
            if (self.pending.length) {
                NSString *tail = [[NSString alloc] initWithData:self.pending encoding:NSUTF8StringEncoding];
                if (tail) [self appendLine:tail];
            }
            if (self.stopping) self.status.stringValue = @"Stopped";
            else if (task.terminationStatus != 0 && !self.failed) self.status.stringValue = @"Session ended unexpectedly. See Activity or Show logs.";
            self.task = nil;
            [self setRunning:NO];
            if (self.quitting) [NSApp replyToApplicationShouldTerminate:YES];
        });
    });
}
- (void)stopPressed:(id)sender {
    NSTask *task = self.task;
    if (!task) return;
    self.stopping = YES;
    self.status.stringValue = @"Stopping…";
    self.stop.enabled = NO;
    pid_t pid = task.processIdentifier;
    if (kill(-pid, SIGINT) != 0 && task.running) [task interrupt];
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC), dispatch_get_main_queue(), ^{
        if (self.task == task) { kill(-pid, SIGTERM); if (task.running) [task terminate]; }
    });
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 2*NSEC_PER_SEC), dispatch_get_main_queue(), ^{
        if (self.task == task) { kill(-pid, SIGKILL); if (task.running) kill(pid, SIGKILL); }
    });
}
- (void)permissionsPressed:(id)sender {
    NSAlert *alert = [NSAlert new];
    alert.messageText = @"Privacy and access";
    alert.informativeText = @"Voice needs Microphone and Speech Recognition. Gaze needs Camera. Live control needs Accessibility, and macOS may request Automation access. Enable the Interaction app or Python/helper entry macOS shows, then restart the session.\n\nSpeech uses Apple's recognition service; on-device-only processing is not enforced. Transcripts are saved in the configured logs folder, which is inside this Dropbox project in the default build.";
    [alert addButtonWithTitle:@"Open Privacy Settings"];
    [alert addButtonWithTitle:@"Cancel"];
    if ([alert runModal] == NSAlertFirstButtonReturn) [NSWorkspace.sharedWorkspace openURL:[NSURL URLWithString:@"x-apple.systempreferences:com.apple.preference.security"]];
}
- (void)logsPressed:(id)sender {
    [NSWorkspace.sharedWorkspace openURL:[NSURL fileURLWithPath:[self.runtime stringByAppendingPathComponent:@"logs"]]];
}
- (void)applicationWillTerminate:(NSNotification *)notification {
    if (self.hotKey) UnregisterEventHotKey(self.hotKey);
}
- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender { return YES; }
- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender {
    if (!self.task) return NSTerminateNow;
    self.quitting = YES;
    [self stopPressed:nil];
    return NSTerminateLater;
}
@end

static OSStatus stopHotKey(EventHandlerCallRef next, EventRef event, void *context) {
    InteractionController *controller = (__bridge InteractionController *)context;
    [controller stopPressed:nil];
    return noErr;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSApplication *app = NSApplication.sharedApplication;
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        NSMenu *menu = [NSMenu new];
        NSMenuItem *appItem = [NSMenuItem new];
        [menu addItem:appItem];
        NSMenu *appMenu = [NSMenu new];
        [appMenu addItemWithTitle:@"Quit Interaction" action:@selector(terminate:) keyEquivalent:@"q"];
        appItem.submenu = appMenu;
        NSMenuItem *editItem = [NSMenuItem new];
        [menu addItem:editItem];
        NSMenu *edit = [[NSMenu alloc] initWithTitle:@"Edit"];
        [edit addItemWithTitle:@"Undo" action:@selector(undo:) keyEquivalent:@"z"];
        [edit addItemWithTitle:@"Cut" action:@selector(cut:) keyEquivalent:@"x"];
        [edit addItemWithTitle:@"Copy" action:@selector(copy:) keyEquivalent:@"c"];
        [edit addItemWithTitle:@"Paste" action:@selector(paste:) keyEquivalent:@"v"];
        [edit addItemWithTitle:@"Select All" action:@selector(selectAll:) keyEquivalent:@"a"];
        editItem.submenu = edit;
        app.mainMenu = menu;
        InteractionController *controller = [InteractionController new];
        app.delegate = controller;
        [app run];
    }
    return 0;
}
