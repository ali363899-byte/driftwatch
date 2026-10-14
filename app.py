"""Local-only development API. Python 3.10+; no package installation."""
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
ROOT=Path(__file__).resolve().parent
MODE=ROOT.name
if MODE=='atlas-evidence':
    from engine import EvidenceIndex, evaluate
    documents=json.loads((ROOT/'sample_documents.json').read_text(encoding='utf-8'))
    index=EvidenceIndex(documents)
else:
    from monitor import report
    from demo import make_data

class Handler(BaseHTTPRequestHandler):
    def respond(self,code,data,kind='application/json'):
        body=(json.dumps(data,ensure_ascii=False,allow_nan=False) if kind=='application/json' else data).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type',kind+'; charset=utf-8')
        self.send_header('Content-Length',str(len(body)))
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Cache-Control','no-store')
        self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        path=urlparse(self.path).path
        if path=='/': self.respond(200,(ROOT/'index.html').read_text(encoding='utf-8'),'text/html')
        elif path=='/api/health': self.respond(200,{'status':'ok','project':MODE})
        else: self.respond(404,{'error':'Route not found'})
    def do_POST(self):
        try:
            # Reject browser cross-origin writes; never enable CORS.
            origin=self.headers.get('Origin')
            if origin and origin!='http://'+self.headers.get('Host',''):
                self.respond(403,{'error':'Cross-origin requests rejected'}); return
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':
                self.respond(415,{'error':'Use application/json'}); return
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=1000000:
                self.respond(413,{'error':'Body must be 1 byte to 1 MB'}); return
            data=json.loads(self.rfile.read(length))
            if not isinstance(data,dict): raise ValueError('Expected JSON object')
            path=urlparse(self.path).path
            global index
            if MODE=='atlas-evidence' and path=='/api/search':
                result=index.answer(data.get('query',''),int(data.get('k',3)))
            elif MODE=='atlas-evidence' and path=='/api/evaluate':
                result=evaluate(index,json.loads((ROOT/'sample_queries.json').read_text(encoding='utf-8')),int(data.get('k',3)))
            elif MODE=='atlas-evidence' and path=='/api/documents':
                docs=data['documents']
                if not isinstance(docs,list) or len(docs)>100: raise ValueError('At most 100 documents')
                candidate=EvidenceIndex(docs)
                index=candidate
                result={'documents':len(docs),'chunks':len(index.chunks)}
            elif MODE=='driftwatch' and path=='/api/report':
                if 'reference' in data:
                    reference,current=data['reference'],data['current']
                else:
                    reference,current=make_data(400,float(data.get('shift',1)))
                if len(reference)+len(current)>10000: raise ValueError('At most 10000 rows')
                result=report(reference,current,float(data.get('threshold',.5)),float(data.get('alert',.2)))
            else: self.respond(404,{'error':'Route not found'}); return
            self.respond(200,result)
        except (ValueError,TypeError,KeyError,OverflowError,AttributeError) as error:
            self.respond(400,{'error':str(error)})
    def log_message(self,*args): pass

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--port',type=int,default=8765 if MODE=='atlas-evidence' else 8766)
    port=parser.parse_args().port
    print(f'{MODE} ready at http://127.0.0.1:{port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()
