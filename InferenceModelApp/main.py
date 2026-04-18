from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import gc
from flask import Flask, request, jsonify, render_template_string, render_template
from flask_cors import CORS

from engineModel.engAutoreg import EngineAutoreg

from engineModel.baseBert import BaseBert
from engineModel.distilBert import DistilBert
from engineModel.engineRoBert import EngineRoBert
from engineModel.engineClassificationSmall import EngineTransformerClassifier

from engineModel.engineRESNET18 import EngineRESNET18
from engineModel.engineMobileNetV2 import EngineMobileNetV2
from engineModel.engineSE_ResNet18 import EngineSEResNetClassifier

from engineModel.engineU_net import EngineU_net
from engineModel.engineAttentionUNet import EngineAttentionUNet
from engineModel.engineUNet3D import EngineUNet3D
from engineModel.engineAttentionUNet3D import EngineAttentionUNet3D

import kagglehub
#this collector
from pathlib import Path
DATASET_NAME = "andrewmvd/pulmonary-embolism-in-ct-images"

def get_dataset_path():
    path = kagglehub.dataset_download(DATASET_NAME)
    return Path(path) / "FUMPE"

dataset_path = get_dataset_path()

#y.evaluate_invariance2(
#    dcm_dir=dataset_path / "PAT034",
#    mat_path=dataset_path / "PAT034.mat",
#    threshold=0.5
#)

#main
class Manager:
    def MyCollector(self, model):
        if hasattr(model, 'model'):
            model.model.cpu()
            del model.model
        if hasattr(model, 'tokenizer'):
            del model.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        del model
        gc.collect()

        gc.collect()

    #this Load-Use-Unload
    
    def ThisControllerLanguageModel(self, promt, MyMagicObject):
        # my magic logic
        # archAutoRegr
        if MyMagicObject == 1:
            model = EngineAutoreg()
            result = model.generate(promt)
            self.MyCollector(model)
            print(result)
            return result
        #bertBaseVanila 
        elif MyMagicObject == 2:
            model = BaseBert()
            result = model.predict(promt)
            self.MyCollector(model)
            print(result)
            return result
        #distilBert 
        elif MyMagicObject == 3:
            model = DistilBert()
            result = model.predict(promt)
            self.MyCollector(model)
            print(result)
            return result
        #Robert
        elif MyMagicObject == 4:
            model = EngineRoBert()
            result = model.predict(promt)
            self.MyCollector(model)
            print(result)
            return result
        #ClassificationSmall
        elif MyMagicObject == 5:
            model = EngineTransformerClassifier()
            result = model.predict()
            self.MyCollector(model)
            print(result)
            return result

    def ThisControllerImageLetterModel(self, image, MyMagicObject):
        #ResNet18(Vanilla)
        if MyMagicObject == 1:
            model = EngineRESNET18()
            result = model.predict()
            self.MyCollector(model)
            print(result)
            return result
        #MobileNetV2
        elif MyMagicObject == 2:
            model = EngineMobileNetV2()
            result = model.predict()
            self.MyCollector(model)
            print(result)
            return result
        #SE-ResNet18
        elif MyMagicObject == 3:
            model = EngineSEResNetClassifier()
            result = model.predict()
            self.MyCollector(model)
            print(result)
            return result
#y.evaluate_invariance2(
#    dcm_dir=dataset_path / "PAT034",
#    mat_path=dataset_path / "PAT034.mat",
#    threshold=0.5
#)
        #end embol.
    def ThisControllerImageEmbol(self, MyMagicObject):
        dataset_path = get_dataset_path()
    
        dcm_dir = dataset_path / "CT_scans" / "PAT034"
        mat_path = dataset_path / "GroundTruth" / "PAT034.mat"
        
        #U_net2d
        if MyMagicObject == 1:
            model = EngineU_net()
            model.demo()
            model.evaluate_invariance2(
                dcm_dir=dcm_dir,
                mat_path=mat_path,
                threshold=0.5
            )
            self.MyCollector(model)
        #U-Net attention2d
        elif MyMagicObject == 2:
            model = EngineAttentionUNet()
            import os
            print(f"dataset_path = {dataset_path}")
            print(f"Содержимое: {os.listdir(dataset_path)}")
            model.evaluate_invariance2(
                dcm_dir=dcm_dir,
                mat_path=mat_path,
                threshold=0.5
            )
            model.demo()
            self.MyCollector(model)
        #3D U-Net witch VGG encoder
        elif MyMagicObject == 3:
            import os
            model = EngineUNet3D()
            print(f"dataset_path = {dataset_path}")
            print(f"Содержимое: {os.listdir(dataset_path)}")
            model.evaluate_invariance(
                dcm_dir=dcm_dir,
                mat_path=mat_path,
                threshold=0.5
            )
            model.demo()
            self.MyCollector(model)
        #3D Attention U-Net
        elif MyMagicObject == 4:
            model = EngineAttentionUNet3D()
            model.evaluate_invariance2(
                dcm_dir=dcm_dir,
                mat_path=mat_path,
                threshold=0.5
            )
            model.demo()
            self.MyCollector(model)
managerForModel = Manager()

managerForModel.ThisControllerImageEmbol(4)
#this test 
app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_data', methods=['POST'])
def process_data():
    data = request.get_json()

    print("RAW DATA:", data)

    prompt = data.get("prompt", "")

    print("PROMPT:", prompt)


    return jsonify({
        "response": result
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)