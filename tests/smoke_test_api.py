import urllib.request, json

def post(text):
    data = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8000/moderate/text",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    res = json.loads(urllib.request.urlopen(req).read().decode())
    cats = [v["category"] for v in res["detected_violations"]]
    print(f"INPUT   : {text}")
    print(f"DECISION: {res['decision']}  (confidence: {res['confidence_score']})")
    print(f"VIOLATIONS: {cats if cats else 'none'}")
    print(f"REASONING : {res['reasoning']['text_explanation'][:120]}")
    print()

post("p0is0n and k1ll y0urs3lf")
post("h a t e  a l l [inferior race] people")
post("I h@t3 $p33ch targeting [inferior race]")
post("Beautiful sunset over the mountains today! Love nature!")
