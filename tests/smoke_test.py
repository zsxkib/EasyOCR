from pathlib import Path
from predict import Predictor
from cog import Path as CogPath

# Simple smoke test: instantiate and run on bowers.jpg
if __name__ == "__main__":
    p = Predictor()
    p.setup()
    here = Path(__file__).resolve().parent.parent
    img = CogPath(str(here / "bowers.jpg"))
    out = p.predict(image=img, languages="en", min_confidence=0.25, preprocessing=True)
    print({k: out[k] for k in ("success", "total_detections", "average_confidence")})
