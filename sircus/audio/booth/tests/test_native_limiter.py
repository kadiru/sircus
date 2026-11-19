"""Render the actual Apple peak-limiter graph offline; no MIDI or audio-device I/O.

macOS-only integration coverage. The tested limiter/gain/mute helpers are extracted
from Native.swift, not reimplemented here. Native.swift itself is never executed.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


FIXTURE = r'''
struct Rendered {let peak:Float;let rms:Double;let mutedPeak:Float;let resumedPeak:Float
    var json:[String:Any] {["peak":peak,"rms":rms,"muted_peak":mutedPeak,"resumed_peak":resumedPeak]}}
let assetRoot=URL(fileURLWithPath:CommandLine.arguments[1])
func render(_ boost:Double,_ voices:Int,_ amplitude:Float,_ impulse:Bool=false,
            _ testMute:Bool=false,_ assetNames:[String]=[],_ velocity:Int=127,
            _ route:String="native",_ ready:Bool=true,_ optedIn:Bool=true) throws -> Rendered {
    let engine=AVAudioEngine(),master=AVAudioMixerNode(),limiter=try makePeakLimiter()
    let format=AVAudioFormat(standardFormatWithSampleRate:48000,channels:1)!
    engine.attach(master);engine.attach(limiter)
    engine.connect(master,to:limiter,format:format);engine.connect(limiter,to:engine.mainMixerNode,format:format)
    var players=[AVAudioPlayerNode](),buffers=[AVAudioPCMBuffer]()
    for index in 0..<voices {
        let player=AVAudioPlayerNode();engine.attach(player);engine.connect(player,to:master,format:format)
        let input:AVAudioPCMBuffer
        if !assetNames.isEmpty {
            let file=try AVAudioFile(forReading:assetRoot.appendingPathComponent(assetNames[index%assetNames.count]+".wav"))
            input=AVAudioPCMBuffer(pcmFormat:file.processingFormat,frameCapacity:AVAudioFrameCount(file.length))!
            try file.read(into:input)
        } else {
            input=AVAudioPCMBuffer(pcmFormat:format,frameCapacity:48000)!;input.frameLength=48000
            for i in 0..<48000 {input.floatChannelData![0][i]=impulse ? (i%500==0 ? amplitude:0):amplitude*Float(sin(2*Double.pi*1000*Double(i)/48000))}
        }
        player.volume=Float(velocity)/127
        players.append(player);buffers.append(input)
    }
    applyNativeOutputGain(master,engine.mainMixerNode,optedIn,route,ready,1)
    try setPeakLimiterBoost(limiter,boost)
    // Manual offline mode is configured BEFORE start; no hardware output is opened.
    try engine.enableManualRenderingMode(.offline,format:format,maximumFrameCount:1024)
    try engine.start();defer{engine.stop()}
    for (p,b) in zip(players,buffers){p.scheduleBuffer(b);p.play()}
    let output=AVAudioPCMBuffer(pcmFormat:engine.manualRenderingFormat,frameCapacity:1024)!
    var peak:Float=0,mutedPeak:Float=0,resumedPeak:Float=0,square=0.0,count=0
    for block in 0..<60 {
        if testMute && block==24 {
            var before:Float=0,after:Float=0
            precondition(AudioUnitGetParameter(limiter.audioUnit,AudioUnitParameterID(kLimiterParam_PreGain),kAudioUnitScope_Global,0,&before)==noErr)
            muteNativeOutput(master,engine.mainMixerNode,limiter)
            precondition(AudioUnitGetParameter(limiter.audioUnit,AudioUnitParameterID(kLimiterParam_PreGain),kAudioUnitScope_Global,0,&after)==noErr)
            precondition(before==after && abs(Double(after)-boost)<0.00001,"Mute/reset must preserve acknowledged PreGain")
            for p in players{p.stop();p.reset()}
        }
        if testMute && block==28 {applyNativeOutputGain(master,engine.mainMixerNode,true,"native",true,1)}
        let status=try engine.renderOffline(1024,to:output)
        precondition(status == .success,"Offline rendering failed: \(status)")
        for i in 0..<Int(output.frameLength){let value=abs(output.floatChannelData![0][i]);precondition(value.isFinite)
            if testMute && block>=24 {if block<28{mutedPeak=max(mutedPeak,value)}else{resumedPeak=max(resumedPeak,value)}} else{peak=max(peak,value)}
            if block>=10 && block<20 {square += Double(value*value);count+=1}
        }
    }
    return Rendered(peak:peak,rms:sqrt(square/Double(count)),mutedPeak:mutedPeak,resumedPeak:resumedPeak)
}
var results=[String:Any]()
results["tone_0"]=try render(0,1,0.1).json
results["tone_3x"]=try render(sircusMaximumBoostDB,1,0.1).json
results["overlap_8"]=try render(sircusMaximumBoostDB,8,0.9).json
results["transients_8"]=try render(sircusMaximumBoostDB,8,1,true).json
results["mute"]=try render(sircusMaximumBoostDB,8,0.9,false,true).json
results["mute_6db"]=try render(6,8,0.9,false,true).json
for velocity in [25,127] {
    results["boing_\(velocity)_0"]=try render(0,1,0,false,false,["cartoon_boing"],velocity).json
    results["boing_\(velocity)_3x"]=try render(sircusMaximumBoostDB,1,0,false,false,["cartoon_boing"],velocity).json
}
results["assets_overlap_8"]=try render(sircusMaximumBoostDB,8,0,false,false,
    ["cartoon_boing","cartoon_squeak","cartoon_spring","cartoon_plop"]).json
results["logic"]=try render(sircusMaximumBoostDB,1,0.1,false,false,[],127,"logic").json
results["not_ready"]=try render(sircusMaximumBoostDB,1,0.1,false,false,[],127,"native",false).json
results["disabled"]=try render(sircusMaximumBoostDB,1,0.1,false,false,[],127,"native",true,false).json
let validator=try makePeakLimiter();try setPeakLimiterBoost(validator,6)
var rejected=0
for invalid in [Double.nan,Double.infinity,-Double.infinity,-0.1,sircusMaximumBoostDB+0.001] {
    do {try setPeakLimiterBoost(validator,invalid)} catch {rejected += 1}
}
var preserved:Float=0
let getStatus=AudioUnitGetParameter(validator.audioUnit,AudioUnitParameterID(kLimiterParam_PreGain),kAudioUnitScope_Global,0,&preserved)
results["validation"]=["rejected":rejected,"preserved_db":preserved,"status":Int(getStatus)]
let bytes=try JSONSerialization.data(withJSONObject:results,options:.sortedKeys)
print(String(data:bytes,encoding:.utf8)!)
'''


@unittest.skipUnless(sys.platform == 'darwin' and shutil.which('swiftc'),
                     'Apple AudioUnit offline rendering requires macOS and Swift')
class NativePeakLimiter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        source=(root/'src/Native.swift').read_text()
        code=source.split('// BEGIN NATIVE PEAK LIMITER (offline renderable without MIDI or audio devices)\n',1)[1].split('// END NATIVE PEAK LIMITER',1)[0]
        with tempfile.TemporaryDirectory(prefix='sircus-offline-limiter-') as folder:
            directory=Path(folder);swift=directory/'test.swift';binary=directory/'test'
            swift.write_text('import Foundation\nimport AVFoundation\nimport AudioToolbox\n'+code+FIXTURE)
            cache=os.environ.get('SIRCUS_SWIFT_MODULE_CACHE',str(directory/'module-cache'))
            compiled=subprocess.run(['swiftc','-module-cache-path',cache,str(swift),'-o',str(binary)],
                                    capture_output=True,text=True,timeout=120)
            if compiled.returncode:raise AssertionError('Offline test compilation failed: '+compiled.stderr)
            result=subprocess.run([str(binary),str(root/'assets')],capture_output=True,text=True,timeout=30)
            if result.returncode:raise AssertionError('Offline DSP render failed: '+result.stderr)
            cls.results=json.loads(result.stdout)

    def test_threefold_gain_preserves_original_volume_curve_below_limit(self):
        baseline=self.results['tone_0'];boosted=self.results['tone_3x']
        self.assertAlmostEqual(baseline['peak'],.05,places=5) # 0.1 signal * original 0.5 master.
        self.assertAlmostEqual(boosted['rms']/baseline['rms'],3.,places=5)

    def test_original_boing_at_representative_and_max_velocity(self):
        for velocity in (25,127):
            with self.subTest(velocity=velocity):
                old=self.results[f'boing_{velocity}_0']['peak'];new=self.results[f'boing_{velocity}_3x']['peak']
                self.assertGreater(old,0);self.assertAlmostEqual(new/old,3.,places=4)
                self.assertLessEqual(new,1.000001)
        ratio=self.results['boing_25_0']['peak']/self.results['boing_127_0']['peak']
        self.assertAlmostEqual(ratio,25/127,places=5) # Boost does not rewrite measured velocity.

    def test_limiter_caps_summed_voices_and_transients(self):
        for key in ('overlap_8','transients_8','assets_overlap_8'):
            with self.subTest(case=key):
                peak=self.results[key]['peak']
                self.assertGreater(peak,.1);self.assertLessEqual(peak,1.000001)

    def test_mute_clears_limiter_tail_and_never_replays_on_enable(self):
        for key in ('mute','mute_6db'):
            with self.subTest(case=key):
                self.assertEqual(self.results[key]['muted_peak'],0)
                self.assertEqual(self.results[key]['resumed_peak'],0)

    def test_boost_does_not_enable_or_leak_into_logic_or_unready_native(self):
        for key in ('logic','not_ready','disabled'):
            with self.subTest(case=key):self.assertEqual(self.results[key]['peak'],0)

    def test_invalid_gain_rejected_without_changing_existing_parameter(self):
        self.assertEqual(self.results['validation'],{'rejected':5,'preserved_db':6,'status':0})


if __name__=='__main__':unittest.main()
