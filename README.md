# FASHION COMPATIBILITY

## Installation
```
pip install -r requirements.txt
```
```
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
```
```
sudo apt-get update && sudo apt-get install tensorflow-model-server
# or sudo apt-get upgrade tensorflow-model-server
```
## Download data
* `data_demo`: https://drive.google.com/file/d/1qFVAAo_Ws5c65GYEXFtEPYb-PAWP8tsa/view?usp=sharing
## Open port:
```
tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=fashion-compatibility --model_base_path="/home/queen/Documents/IFashion/visual-compatibility/fc_serving/fashion_batch8/serving/" --enable_batching=true --batching_parameters_file="/home/queen/Documents/IFashion/visual-compatibility/fc_serving/fashion_batch8/batching_parameters.txt"
```
## Running:
```
python main.py --questions 193458045 --questions 195937486 --questions 123858646 --types 'all-body' --types 'tops' --k 10 --is_save True
```
#### Parameters:
* --questions: the image name of questions (image)
* --types: the types of answers
* --k: the number of answers
* --is_save: if true, the result is saved to image
