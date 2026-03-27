from mcp.server import Server
import pickle
from pathlib import Path

# Load trained model
current_dir = Path(__file__).parent
model_path = current_dir / "models" / "best_model.pkl"

with open(model_path, 'rb') as f:
 model = pickle.load(f)
server = Server("my-ml-model")
@server.tool()

def predict(features: list[float]) -> dict:
 """Make a prediction using the trained model."""
 prediction = model.predict([features])[0]
 return {"prediction": prediction}
if __name__ == "__main__":

 server.run()
