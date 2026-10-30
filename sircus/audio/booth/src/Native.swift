import Foundation
import AVFoundation
import CoreMIDI

// JSON-lines bridge. All mutation serialized on the main queue. No robot APIs.
let root = URL(fileURLWithPath: CommandLine.arguments[1])
func emit(_ object: [String:Any]) {
    if let d = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]) {
        FileHandle.standardOutput.write(d); FileHandle.standardOutput.write(Data([10]))
    }
}
let engine = AVAudioEngine()
let master = AVAudioMixerNode()
var players = [AVAudioPlayerNode]()
var speeds = [AVAudioUnitVarispeed]()
var notes = [String:Int]()
var loopNote: Int? = nil
var loopGeneration = 0
var buffers = [String:AVAudioPCMBuffer]()
var audioReady = false
var enabled = false
var volume: Float = 0.35
var route = "native"
var nextVoice = 1
var generation = 0
var peak: Float = 0
var currentPeak: Float = 0
var blockTime: UInt64 = 0
var heartbeat = DispatchTime.now().uptimeNanoseconds
var midiClient = MIDIClientRef()
var inputPort = MIDIPortRef()
var virtualSource = MIDIEndpointRef()
var selectedUID: Int32? = nil
var connectedEndpoint = MIDIEndpointRef()
var lastSources = ""
var runningStatus: UInt8 = 0
var pending = [UInt8]()
var sysEx = false
let channel: UInt8 = 15 // channel 16, deliberate isolation from raw ODD channel 1

func midiProperty(_ obj: MIDIObjectRef, _ key: CFString) -> String {
    var value: Unmanaged<CFString>?
    if MIDIObjectGetStringProperty(obj, key, &value) == noErr, let v=value {return v.takeRetainedValue() as String}
    return ""
}
func uid(_ obj: MIDIObjectRef) -> Int32 {var n:Int32=0;MIDIObjectGetIntegerProperty(obj,kMIDIPropertyUniqueID,&n);return n}
func online(_ obj: MIDIObjectRef) -> Bool {var n:Int32=0;MIDIObjectGetIntegerProperty(obj,kMIDIPropertyOffline,&n);return n==0}
func sources() -> [[String:Any]] {
    (0..<MIDIGetNumberOfSources()).compactMap { i in
        let ep=MIDIGetSource(i);let name=midiProperty(ep,kMIDIPropertyDisplayName)
        var entity=MIDIEntityRef();MIDIEndpointGetEntity(ep,&entity)
        if ep==virtualSource || name.lowercased().contains("sircus") || entity==0 {return nil}
        return ["id":uid(ep),"name":name,"online":online(ep)]
    }
}
func send(_ bytes:[UInt8]) {
    guard virtualSource != 0 else { return }
    let ptr=UnsafeMutableRawPointer.allocate(byteCount:1024,alignment:16);defer{ptr.deallocate()}
    let list=ptr.assumingMemoryBound(to:MIDIPacketList.self)
    let p=MIDIPacketListInit(list)
    bytes.withUnsafeBufferPointer { b in
        _ = MIDIPacketListAdd(list,1024,p,0,b.count,b.baseAddress!)
    }
    MIDIReceived(virtualSource,list)
}
func silence() {
    generation+=1
    master.outputVolume=0
    for p in players {p.stop();p.reset()}
    // Only this isolated output channel/port. Never sends back to the ODD Ball.
    send([0xbf,64,0]);send([0xbf,11,0]);send([0xbf,120,0]);send([0xbf,123,0])
    loopGeneration+=1;loopNote=nil
    for n in notes.values {send([0x8f,UInt8(n),0])}
}
func gain() {master.outputVolume = enabled && route=="native" ? volume*0.5 : 0}
func stopTransport() {
    loopGeneration+=1
    if players.count>0 {players[0].stop();players[0].reset()}
    if let note=loopNote {send([0x8f,UInt8(note),0])};loopNote=nil
}
func play(_ name:String,_ velocity:Int,_ looping:Bool=false,_ pitch:Double=0,_ maxSeconds:Double=8) {
    guard enabled,let note=notes[name],let b=buffers[name] else {return}
    let v=Float(max(1,min(127,velocity)))/127
    let rate=Float(pow(2,max(-12,min(12,pitch))/12))
    if looping {stopTransport();loopNote=note}
    if route=="logic" {
        send([0x9f,UInt8(note),UInt8(max(1,min(127,velocity)))])
        if !looping {
            let g=generation
            DispatchQueue.main.asyncAfter(deadline:.now()+Double(b.frameLength)/b.format.sampleRate) {
                if g==generation {send([0x8f,UInt8(note),0])}
            }
        }
    } else if audioReady {
        let i=looping ? 0 : nextVoice
        if !looping {nextVoice = nextVoice % 7 + 1}
        let p=players[i];p.stop();p.reset();p.volume=v
        speeds[i].rate=rate
        p.scheduleBuffer(b,at:nil,options:looping ? [.loops] : [],completionHandler:nil)
        p.play()
    }
    if looping {
        let g=loopGeneration
        DispatchQueue.main.asyncAfter(deadline:.now()+max(0.2,min(20,maxSeconds))) {
            if g==loopGeneration {stopTransport();emit(["type":"loop_stopped"])}
        }
    }
    emit(["type":"played","sound":name,"velocity":velocity,"route":route,"loop":looping])
}
func decode(_ data:[UInt8],_ source:Int32) {
    for b in data {
        if b>=0xf8 {continue}
        if b==0xf0 {sysEx=true;runningStatus=0;pending=[];continue}
        if b==0xf7 {sysEx=false;continue}
        if sysEx {continue}
        if b&0x80 != 0 {runningStatus=b<0xf0 ? b : 0;pending=[];continue}
        guard runningStatus != 0 else {continue}
        pending.append(b)
        let kind=runningStatus&0xf0
        let needed=(kind==0xc0 || kind==0xd0) ? 1 : 2
        if pending.count==needed {
            emit(["type":"midi","input":source,"status":Int(runningStatus),"channel":Int(runningStatus&15)+1,"data":pending.map{Int($0)},"time_ns":DispatchTime.now().uptimeNanoseconds])
            pending=[]
        }
    }
}
let cstatus=MIDIClientCreateWithBlock("Sircus Audio" as CFString,&midiClient) { _ in }
let pstatus=MIDIInputPortCreateWithBlock(midiClient,"Sircus selected input" as CFString,&inputPort) { list, context in
    let source=Int32(truncatingIfNeeded:Int(bitPattern:context))
    var p=UnsafeRawPointer(list).advanced(by:MemoryLayout<MIDIPacketList>.offset(of:\.packet)!).assumingMemoryBound(to:MIDIPacket.self)
    var bytes=[UInt8]()
    for _ in 0..<list.pointee.numPackets {
        let data=UnsafeRawPointer(p).advanced(by:MemoryLayout<MIDIPacket>.offset(of:\.data)!).assumingMemoryBound(to:UInt8.self)
        bytes.append(contentsOf:UnsafeBufferPointer(start:data,count:Int(p.pointee.length)))
        p=UnsafePointer(MIDIPacketNext(p))
    }
    DispatchQueue.main.async {if selectedUID==source {decode(bytes,source)}}
}
let sstatus=MIDISourceCreate(midiClient,"Sircus FX" as CFString,&virtualSource)

func reconcile() {
    var found=MIDIEndpointRef()
    if let wanted=selectedUID {
        for i in 0..<MIDIGetNumberOfSources() {let ep=MIDIGetSource(i);if uid(ep)==wanted && online(ep) && sources().contains(where:{($0["id"] as? Int32)==wanted}) {found=ep;break}}
    }
    if found != connectedEndpoint {
        let wasConnected=connectedEndpoint != 0
        if wasConnected {MIDIPortDisconnectSource(inputPort,connectedEndpoint)}
        connectedEndpoint=found;runningStatus=0;pending=[]
        if found != 0 {
            let status=MIDIPortConnectSource(inputPort,found,UnsafeMutableRawPointer(bitPattern:Int(selectedUID!)))
            if status != 0 {connectedEndpoint=0;emit(["type":"error","message":"MIDI connect failed \(status)"])}
        }
        enabled=false;silence()
        emit(["type":"connection","connected":connectedEndpoint != 0,"input":selectedUID as Any? ?? NSNull(),"fx_forced_off":true])
    }
    let rows=sources();let key=String(data:(try? JSONSerialization.data(withJSONObject:rows,options:.sortedKeys)) ?? Data(),encoding:.utf8) ?? ""
    if key != lastSources {lastSources=key;emit(["type":"inputs","inputs":rows])}
}

do {
    let rows=try JSONSerialization.jsonObject(with:Data(contentsOf:root.appendingPathComponent("assets/map.json"))) as! [[String:Any]]
    engine.attach(master)
    let format=AVAudioFormat(standardFormatWithSampleRate:48000,channels:1)!
    engine.connect(master,to:engine.mainMixerNode,format:format)
    for _ in 0..<8 {
        let p=AVAudioPlayerNode();let speed=AVAudioUnitVarispeed()
        engine.attach(p);engine.attach(speed);engine.connect(p,to:speed,format:format)
        engine.connect(speed,to:master,format:format);players.append(p);speeds.append(speed)
    }
    for row in rows {
        notes[row["name"] as! String]=row["note"] as? Int
        let f=try AVAudioFile(forReading:root.appendingPathComponent("assets/\(row["file"] as! String)"))
        let b=AVAudioPCMBuffer(pcmFormat:f.processingFormat,frameCapacity:AVAudioFrameCount(f.length))!
        try f.read(into:b);buffers[row["name"] as! String]=b
    }
    master.outputVolume=0
    engine.mainMixerNode.installTap(onBus:0,bufferSize:1024,format:nil) { b,_ in
        guard let d=b.floatChannelData else{return}
        var v:Float=0
        for i in 0..<Int(b.frameLength) {v=max(v,abs(d[0][i]))}
        let ts=DispatchTime.now().uptimeNanoseconds
        DispatchQueue.main.async {peak=max(peak,v);currentPeak=v;blockTime=ts}
    }
    try engine.start();audioReady=true
} catch {emit(["type":"error","message":"Native audio unavailable: \(error)"])}
silence()
emit(["type":"ready","audio_ready":audioReady,"midi_status":[cstatus,pstatus,sstatus],"output":"Sircus FX","output_channel":16])
let timer=DispatchSource.makeTimerSource(queue:.main)
timer.schedule(deadline:.now(),repeating:1)
timer.setEventHandler {
    reconcile()
    if enabled && DispatchTime.now().uptimeNanoseconds-heartbeat > 3_000_000_000 {enabled=false;silence();emit(["type":"watchdog","message":"Audio parent heartbeat lost; muted" ])}
    emit(["type":"meter","peak":peak,"current_peak":currentPeak,"block_time_ns":blockTime,"audio_ready":audioReady && engine.isRunning,"enabled":enabled,"route":route]);peak=0
}
timer.resume()
DispatchQueue.global().async {
    while let line=readLine() {
        guard let data=line.data(using:.utf8),let obj=(try? JSONSerialization.jsonObject(with:data)) as? [String:Any] else{continue}
        DispatchQueue.main.async {
            switch obj["cmd"] as? String {
            case "heartbeat":heartbeat=DispatchTime.now().uptimeNanoseconds
            case "enable":
                enabled=obj["value"] as? Bool ?? false
                if !enabled {silence()} else {gain();if route=="logic" {send([0xbf,11,UInt8(volume*127)])}}
            case "volume":volume=Float((obj["value"] as? Double) ?? 0.35);gain();if enabled && route=="logic" {send([0xbf,11,UInt8(max(0,min(127,volume*127)))])}
            case "route":enabled=false;silence();route=obj["value"] as? String ?? "native";gain()
            case "play":play(obj["sound"] as? String ?? "",obj["velocity"] as? Int ?? 85,
                             obj["loop"] as? Bool ?? false,obj["pitch"] as? Double ?? 0,
                             obj["max_seconds"] as? Double ?? 8)
            case "stop_transport":stopTransport()
            case "panic":enabled=false;silence()
            case "stop_all":silence();gain()
            case "input":
                if connectedEndpoint != 0 {MIDIPortDisconnectSource(inputPort,connectedEndpoint);connectedEndpoint=0}
                selectedUID=(obj["value"] as? NSNumber)?.int32Value;enabled=false;silence();reconcile()
            case "list":emit(["type":"inputs","inputs":sources()])
            default:break
            }
        }
    }
    DispatchQueue.main.async {enabled=false;silence();engine.stop();exit(0)}
}
dispatchMain()
