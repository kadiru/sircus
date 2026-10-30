// Native macOS virtual MIDI-1 source. No network, robot or audio-device access.
// Build: xcrun swiftc sircus/audio/coremidi.swift -o .runtime/sircus-midi
import Foundation
import CoreMIDI

var client = MIDIClientRef()
var source = MIDIEndpointRef()
func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}
guard MIDIClientCreateWithBlock("Sircus SFX" as CFString, &client, nil) == noErr else {
    fail("Cannot create CoreMIDI client")
}
defer { MIDIClientDispose(client) }
// MIDI-1 API is retained for interoperability with Logic Scripter.
guard MIDISourceCreate(client, "Sircus SFX" as CFString, &source) == noErr else {
    fail("Cannot create virtual MIDI source")
}
defer { MIDIEndpointDispose(source) }
// Stable app-local ID permits Logic to remember the port. Refuse collisions.
let uniqueID: MIDIUniqueID = 1397119025
var existing = MIDIObjectRef()
var existingType = MIDIObjectType.other
if MIDIObjectFindByUniqueID(uniqueID, &existing, &existingType) == noErr && existing != source {
    fail("Sircus SFX endpoint already exists; run only one helper")
}
guard MIDIObjectSetIntegerProperty(source, kMIDIPropertyUniqueID, uniqueID) == noErr else {
    fail("Cannot set stable source ID")
}
func send(_ bytes: [UInt8]) {
    var packets = MIDIPacketList()
    withUnsafeMutablePointer(to: &packets) { list in
        let packet = MIDIPacketListInit(list)
        bytes.withUnsafeBufferPointer { buffer in
            // A single three-byte message fits the stack MIDIPacketList.
            _ = MIDIPacketListAdd(list, MemoryLayout<MIDIPacketList>.size, packet,
                                  0, buffer.count, buffer.baseAddress!)
            if MIDIReceived(source, list) != noErr { fail("CoreMIDI delivery failed") }
        }
    }
}
if CommandLine.arguments.contains("--check") {
    print("CoreMIDI source created; no MIDI sent")
} else {
    FileHandle.standardOutput.write(Data("{\"ready\":true,\"port\":\"Sircus SFX\"}\n".utf8))
    while let line = readLine() {
        guard line.utf8.count <= 128,
              let data = line.data(using: .utf8),
              let numbers = try? JSONDecoder().decode([Int].self, from: data),
              numbers.count == 3,
              (0x80...0xEF).contains(numbers[0]),
              [0x80, 0x90, 0xB0, 0xE0].contains(numbers[0] & 0xF0),
              (0...127).contains(numbers[1]), (0...127).contains(numbers[2]) else { continue }
        send(numbers.map { UInt8($0) })
    }
    for ch: UInt8 in 0...15 {
        send([0xB0 | ch, 120, 0]); send([0xB0 | ch, 123, 0]); send([0xE0 | ch, 0, 64])
    }
}
