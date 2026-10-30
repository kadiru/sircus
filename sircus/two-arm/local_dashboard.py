"""Run the existing two-arm dashboard against local fake-only8083, no cloud keys."""
import argparse,asyncio,sys
from pathlib import Path
from local_client import connect
ROOT=Path(__file__).resolve().parent
# Match the baseline server's imports: its own legacy physics is only imported,
# never executed, because OFFLINE remains false and all API requests go to Viam.
sys.path.insert(0,str(ROOT.parent/'simulation'))
import server

class LocalBridge(server.ViamBridge):
    async def invoke(self,command):
        if self.robot is None:self.robot,self.controller=await connect()
        return await asyncio.wait_for(self.controller.do_command(command,timeout=25),timeout=28)

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);a=p.parse_args()
    server.OFFLINE=False;server.BRIDGE=LocalBridge()
    server.connection=lambda connected=True: {'backend':'viam','connected':connected,'machine_id':None,
        'machine_url':None,'part_id':None,'label':'LOCAL FAKE-ONLY · localhost:8083','physical_hardware':False}
    print(f'Two-arm local dashboard http://127.0.0.1:{a.port} (fake server8083; no cloud identity)',flush=True)
    server.ThreadingHTTPServer(('127.0.0.1',a.port),server.Handler).serve_forever()
if __name__=='__main__':main()
