import tensorflow as tf
import numpy as np


def main():
    try:
        # Load the model
        model = tf.saved_model.load(
            "/Users/lau/Documents/09_work_past/uzh/wibex_local/webibex/core/embedding_model"
        )
        print("Model loaded successfully.")

        # Create a dummy input tensor (adjust shape to match your model's input)
        array = np.random.rand(288, 144, 3).astype(np.float32)
        input_tensor = tf.expand_dims(array, axis=0)

        # Run inference
        embedder = model.signatures["serving_default"]
        output = embedder(input_tensor)["output_tensor"].numpy().tolist()[0]
        print(f"Inference successful. Output size: {len(output)}")
    except Exception as e:
        print(f"Error during model loading or inference: {e}")


if __name__ == "__main__":
    main()
