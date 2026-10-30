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
var voiceOwners = SircusVoiceOwners()
var loopLogicID: UUID? = nil
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
var selectedUIDs = Set<Int32>()
var connectedEndpoints = [Int32:MIDIEndpointRef]()
var lastSources = ""
var decoders = [Int32:SircusMIDIDecoder]()
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
    loopGeneration+=1;loopNote=nil;loopLogicID=nil;voiceOwners=SircusVoiceOwners()
    for n in notes.values {send([0x8f,UInt8(n),0])}
}
func gain() {master.outputVolume = enabled && route=="native" ? volume*0.5 : 0}
// BEGIN PURE VOICE OWNERSHIP (tested without audio devices)
struct SircusVoiceOwners {
    var native = [Int:String]()
    var logic = [UUID:(note:Int,trial:String?)]()
    mutating func claimNative(_ index:Int,_ trial:String?) {native[index]=trial}
    mutating func addLogic(_ note:Int,_ trial:String?) -> UUID {
        let id=UUID();logic[id]=(note,trial);return id
    }
    mutating func removeLogic(_ id:UUID) -> Int? {
        guard let voice=logic.removeValue(forKey:id) else {return nil}
        return logic.values.contains(where:{$0.note==voice.note}) ? nil : voice.note
    }
    mutating func removeTrial(_ trial:String) -> ([Int],[Int]) {
        let indices=native.filter{$0.value==trial}.map{$0.key}
        for index in indices {native.removeValue(forKey:index)}
        let ids=logic.filter{$0.value.trial==trial}.map{$0.key}
        var notes=[Int]()
        for id in ids {if let note=removeLogic(id) {notes.append(note)}}
        return (indices,notes)
    }
}
// END PURE VOICE OWNERSHIP
func stopTransport() {
    loopGeneration+=1
    if players.count>0 {players[0].stop();players[0].reset()}
    voiceOwners.native.removeValue(forKey:0)
    if let id=loopLogicID,let note=voiceOwners.removeLogic(id) {send([0x8f,UInt8(note),0])}
    loopNote=nil;loopLogicID=nil
}
func stopTrial(_ trial:String) {
    guard !trial.isEmpty else {return}
    if voiceOwners.native[0]==trial || (loopLogicID.flatMap{voiceOwners.logic[$0]?.trial} == trial) {stopTransport()}
    let (indices,notesToStop)=voiceOwners.removeTrial(trial)
    for index in indices where index<players.count {players[index].stop();players[index].reset()}
    for note in notesToStop {send([0x8f,UInt8(note),0])}
}
func play(_ name:String,_ velocity:Int,_ looping:Bool=false,_ pitch:Double=0,_ maxSeconds:Double=8,_ trial:String?=nil) {
    guard enabled,let note=notes[name],let b=buffers[name] else {return}
    let v=Float(max(1,min(127,velocity)))/127
    let rate=Float(pow(2,max(-12,min(12,pitch))/12))
    if looping {stopTransport();loopNote=note}
    if route=="logic" {
        let id=voiceOwners.addLogic(note,trial)
        if looping {loopLogicID=id}
        send([0x9f,UInt8(note),UInt8(max(1,min(127,velocity)))])
        if !looping {
            let g=generation
            DispatchQueue.main.asyncAfter(deadline:.now()+Double(b.frameLength)/b.format.sampleRate) {
                if g==generation,let off=voiceOwners.removeLogic(id) {send([0x8f,UInt8(off),0])}
            }
        }
    } else if audioReady {
        let i=looping ? 0 : nextVoice
        if !looping {nextVoice = nextVoice % 7 + 1}
        let p=players[i];p.stop();p.reset();p.volume=v
        voiceOwners.claimNative(i,trial)
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
    var played:[String:Any] = ["type":"played","sound":name,"velocity":velocity,"route":route,"loop":looping]
    if let id=trial {played["trial_id"]=id};emit(played)
}
// BEGIN PURE MIDI DECODER (tested without opening CoreMIDI or audio)
struct SircusMIDIDecoder {
    var runningStatus: UInt8 = 0
    var pending = [UInt8]()
    var sysEx = false
    mutating func feed(_ bytes:[UInt8]) -> [(UInt8,[UInt8])] {
        var messages = [(UInt8,[UInt8])]()
        for b in bytes {
            if b>=0xf8 {continue}
            if b==0xf0 {sysEx=true;runningStatus=0;pending=[];continue}
            if b==0xf7 {sysEx=false;runningStatus=0;pending=[];continue}
            if sysEx {continue}
            if b&0x80 != 0 {runningStatus=b<0xf0 ? b : 0;pending=[];continue}
            guard runningStatus != 0 else {continue}
            pending.append(b)
            let kind=runningStatus&0xf0
            let needed=(kind==0xc0 || kind==0xd0) ? 1 : 2
            if pending.count==needed {messages.append((runningStatus,pending));pending=[]}
        }
        return messages
    }
}
// END PURE MIDI DECODER
func decode(_ data:[UInt8],_ source:Int32) {
    var parser=decoders[source] ?? SircusMIDIDecoder()
    for (status,bytes) in parser.feed(data) {
        emit(["type":"midi","input":source,"status":Int(status),"channel":Int(status&15)+1,"data":bytes.map{Int($0)},"time_ns":DispatchTime.now().uptimeNanoseconds])
    }
    decoders[source]=parser
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
    DispatchQueue.main.async {if selectedUIDs.contains(source) && connectedEndpoints[source] != nil {decode(bytes,source)}}
}
let sstatus=MIDISourceCreate(midiClient,"Sircus FX" as CFString,&virtualSource)

func reconcile(force:Bool=false) {
    let rows=sources()
    let allowed=Set(rows.compactMap { ($0["id"] as? NSNumber)?.int32Value })
    var found=[Int32:MIDIEndpointRef]()
    for i in 0..<MIDIGetNumberOfSources() {
        let ep=MIDIGetSource(i), identity=uid(ep)
        if selectedUIDs.contains(identity) && allowed.contains(identity) && online(ep) {found[identity]=ep}
    }
    if force || found != connectedEndpoints {
        for endpoint in connectedEndpoints.values {MIDIPortDisconnectSource(inputPort,endpoint)}
        connectedEndpoints=[:];decoders=[:]
        for (identity,endpoint) in found {
            let status=MIDIPortConnectSource(inputPort,endpoint,UnsafeMutableRawPointer(bitPattern:Int(identity)))
            if status==noErr {connectedEndpoints[identity]=endpoint}
            else {emit(["type":"error","message":"MIDI connect failed \(status)"])}
        }
        enabled=false;silence()
        emit(["type":"connection","connected":!connectedEndpoints.isEmpty,
              "input":selectedUIDs.count==1 ? selectedUIDs.first! as Any : NSNull(),
              "selected_inputs":selectedUIDs.sorted(),"connected_inputs":connectedEndpoints.keys.sorted(),
              "fx_forced_off":true])
    }
    let key=String(data:(try? JSONSerialization.data(withJSONObject:rows,options:.sortedKeys)) ?? Data(),encoding:.utf8) ?? ""
    if key != lastSources {lastSources=key;emit(["type":"inputs","inputs":rows])}
}
func setInputs(_ values:[Int32]) {
    guard values.count<=4 && Set(values).count==values.count && !values.contains(0) else {
        enabled=false;silence();emit(["type":"error","message":"Choose at most four unique nonzero MIDI inputs"]);return
    }
    selectedUIDs=Set(values);enabled=false;silence();reconcile(force:true)
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
                             obj["max_seconds"] as? Double ?? 8,obj["trial_id"] as? String)
            case "stop_trial":if let trial=obj["trial_id"] as? String {stopTrial(trial)}
            case "stop_transport":stopTransport()
            case "panic":enabled=false;silence()
            case "stop_all":silence();gain()
            case "input":
                if let value=obj["value"] as? NSNumber {setInputs([value.int32Value])} else {setInputs([])}
            case "inputs":
                if let values=obj["values"] as? [Int32] {setInputs(values)}
                else {enabled=false;silence();emit(["type":"error","message":"MIDI inputs must be signed32 identities"])}
            case "list":emit(["type":"inputs","inputs":sources()])
            default:break
            }
        }
    }
    DispatchQueue.main.async {enabled=false;silence();engine.stop();exit(0)}
}
// CoreMIDI topology updates must be serviced by the main run loop. On this Mac,
// dispatchMain() kept timers/audio alive but cached the pre-pairing source list;
// a read-only A/B probe received hot-plug notifications only with RunLoop.run.
// DispatchQueue.main callbacks still execute on the same serialized thread.
RunLoop.main.run()
