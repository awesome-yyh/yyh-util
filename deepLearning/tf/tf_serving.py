import time
import numpy as np
import tensorflow as tf
import json
import requests


# docker run -t --rm -p 8500:8500 -p 8501:8501 --mount type=bind,
# source=/Users/yaheyang/mypython/yyh-util/models/multiModel,
# target=/models/multiModel --platform linux/amd64 
# cedricbl/tf-serving-universal-amd64 
# --model_config_file=/models/multiModel/models.config 
# --model_config_file_poll_wait_seconds=60 
# --allow_version_labels_for_unavailable_models=true

# root_url = "http://127.0.0.1:8501"
# url = "%s/v1/models/cnn/metadata" % root_url
# resp = requests.get(url)
# print(resp.text)

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

t0 = time.time()
# pre_url = "http://localhost:8501/v1/models/cnn/versions/100000:predict"
# pre_url = "http://localhost:8501/v1/models/mlp/versions/100000:predict"
pre_url = "http://localhost:8501/v1/models/mlp/labels/stable:predict"
headers = {"content-type": "application/json"}
test_input = {
   "instances": [np.array(test_images[0]).tolist()]
}
test_input = json.dumps(test_input)

response_json = requests.post(pre_url, data=test_input, headers=headers)
print(np.argmax(json.loads(response_json.text)["predictions"]))
print((time.time()-t0)*1000, " ms")
