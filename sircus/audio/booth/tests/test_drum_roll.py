import io
import json
from pathlib import Path
import struct
import sys
import unittest
import wave
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import generate_drum_roll as drums


class DrumRollTests(unittest.TestCase):
    def test_asset_is_reproducible_bounded_and_sample_continuous(self):
        raw=(drums.ROOT/'assets/drum_roll.wav').read_bytes()
        self.assertEqual(raw,drums.wav_bytes())
        with wave.open(io.BytesIO(raw),'rb') as stream:
            self.assertEqual((stream.getnchannels(),stream.getsampwidth(),stream.getframerate(),stream.getnframes()),(1,2,48000,48000))
            values=struct.unpack('<48000h',stream.readframes(48000))
        self.assertLessEqual(max(map(abs,values)),5243)
        self.assertGreater(max(map(abs,values)),5000)
        self.assertEqual(values[0],0);self.assertEqual(values[-1],0)
        for i in range(8):
            hit=values[i*6000:(i+1)*6000]
            self.assertGreater(sum(abs(v) for v in hit[:3000]),100000)
            self.assertTrue(all(v==0 for v in hit[5280:]))
        mapping=json.loads((drums.ROOT/'assets/map.json').read_text())
        row=next(r for r in mapping if r['name']=='drum_roll')
        self.assertEqual(row['note'],54);self.assertEqual(row['seconds'],1)
        self.assertEqual(len({r['note'] for r in mapping}),len(mapping))

if __name__=='__main__':unittest.main()
