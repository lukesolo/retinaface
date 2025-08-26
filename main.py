import time
import tensorflow as tf
from retinaface.model.retinaface_model import build_model


def main():
    # Build model
    model = build_model()

    # Random input batch (float32, range [0,1))
    x = tf.random.uniform((1, 256, 256, 3), dtype=tf.float32)

    # Warm-up run (graph/build caches, GPU warmup)
    _ = model(x, training=False)
    # Ensure completion
    _ = [t.numpy() for t in _] if isinstance(_, (list, tuple)) else _.numpy()

    for batch_size in [1, 2, 4, 8]:
        # Inference settings
        height, width = 320, 320

        x = tf.random.uniform((batch_size, height, width, 3), dtype=tf.float32)
        # Timed run
        start = time.perf_counter()
        y = model(x, training=False)
        # Force sync to measure true execution time
        _ = [t.numpy() for t in y] if isinstance(y, (list, tuple)) else y.numpy()
        elapsed = time.perf_counter() - start

        per_image_ms = (elapsed / batch_size) * 1000.0
        print(f"Batch size: {batch_size}, Input: {height}x{width}")
        print(f"Inference time: {elapsed:.3f} s ({per_image_ms:.2f} ms/image)")

    export_file = "retinaface_full_model.h5"
    model.save(export_file)
    print(f"Keras H5 model saved to: {export_file}")

    # # Save the whole model
    # try:
    #     tf_version_major = int(tf.__version__.split(".", maxsplit=1)[0])
    #     if tf_version_major >= 2:
    #         export_dir = "retinaface_saved_model"
    #         tf.saved_model.save(model, export_dir)
    #         print(f"SavedModel exported to: {export_dir}")
    #     else:
    #         export_file = "retinaface_full_model.h5"
    #         model.save(export_file)
    #         print(f"Keras H5 model saved to: {export_file}")
    # except Exception as e:
    #     print(f"Failed to save model: {e}")


if __name__ == "__main__":
    main()
