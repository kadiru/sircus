"""Compile the actual pure decoder without initializing CoreMIDI or AVAudioEngine."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

@unittest.skipUnless(shutil.which('swiftc'),'Swift compiler unavailable; native decoder compilation is a Mac commissioning check')
class NativeDecoder(unittest.TestCase):
    def test_interleaved_ball_streams_have_independent_running_status(self):
        source=(Path(__file__).resolve().parents[1]/'src/Native.swift').read_text()
        pure=source.split('// BEGIN PURE MIDI DECODER (tested without opening CoreMIDI or audio)\n',1)[1].split('// END PURE MIDI DECODER',1)[0]
        fixture='''
var pink=SircusMIDIDecoder(), purple=SircusMIDIDecoder(), green=SircusMIDIDecoder(), gray=SircusMIDIDecoder()
precondition(pink.feed([0x90,0]).isEmpty)
precondition(purple.feed([0x91,2,64]).first!.0==0x91)
let p=pink.feed([90]);precondition(p.count==1 && p[0].0==0x90 && p[0].1==[0,90])
precondition(purple.feed([3,65]).first!.1==[3,65])
precondition(green.feed([0x92,0,0xf8,80]).first!.1==[0,80])
precondition(gray.feed([70]).isEmpty)
precondition(pink.feed([0xf0,1,2,0xf8,3]).isEmpty)
precondition(purple.feed([4,66]).first!.1==[4,66])
precondition(pink.feed([0xf7,0,90]).isEmpty)
precondition(pink.feed([0x90,0,100]).first!.1==[0,100])
precondition(gray.feed([0xc3,10,11]).count==2)
print("PASS independent MIDI running status, partial packets, realtime, SysEx and single-byte messages")
'''
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'decoder.swift';path.write_text(pure+fixture)
            binary=Path(folder)/'decoder'
            subprocess.run(['swiftc',str(path),'-o',str(binary)],check=True,capture_output=True,timeout=60)
            result=subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=10)
            self.assertIn('PASS independent MIDI',result.stdout)

    def test_scoped_voice_cancellation_keeps_newer_ball_and_shared_logic_notes(self):
        source=(Path(__file__).resolve().parents[1]/'src/Native.swift').read_text()
        pure=source.split('// BEGIN PURE VOICE OWNERSHIP (tested without audio devices)\n',1)[1].split('// END PURE VOICE OWNERSHIP',1)[0]
        fixture='''
var v=SircusVoiceOwners()
v.claimNative(1,"pink-old");v.claimNative(2,"purple-new");v.claimNative(0,"purple-new")
let a=v.addLogic(60,"pink-old"),b=v.addLogic(60,"purple-new")
let c=v.addLogic(61,"pink-old")
let stopped=v.removeTrial("pink-old")
precondition(stopped.0==[1]);precondition(stopped.1==[61])
precondition(v.native[0]=="purple-new" && v.native[2]=="purple-new")
precondition(v.removeLogic(a)==nil);precondition(v.removeLogic(b)==60)
v.claimNative(2,"green-newer");precondition(v.removeTrial("purple-new").0==[0])
precondition(v.native[2]=="green-newer")
print("PASS scoped voice cancellation")
'''
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'voices.swift';path.write_text('import Foundation\n'+pure+fixture)
            binary=Path(folder)/'voices'
            subprocess.run(['swiftc',str(path),'-o',str(binary)],check=True,capture_output=True,timeout=60)
            result=subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=10)
            self.assertIn('PASS scoped voice',result.stdout)
