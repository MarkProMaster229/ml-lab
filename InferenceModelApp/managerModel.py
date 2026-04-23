import torch
import gc

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

from pathlib import Path
import os

def get_images_from_folder(folder_path):
    folder = Path(folder_path)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for image_path in folder.iterdir():
        if image_path.suffix.lower() in extensions:
            yield str(image_path)


#image_folder = "myLetterTest"

#for image_path in get_images_from_folder(image_folder):
#    result = obj.ThisControllerImageLetterModel(image_path, MyMagicObject=1)

#y.evaluate_invariance2(
#    dcm_dir=dataset_path / "PAT034",
#    mat_path=dataset_path / "PAT034.mat",
#    threshold=0.5
#)

#main
    def MyCollector(self, model):
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
            result = model.predict(promt)
            self.MyCollector(model)
            print(result)
            return result

    def ThisControllerImageLetterModel(self, image, MyMagicObject):
        #ResNet18(Vanilla)
        if MyMagicObject == 1:
            model = EngineRESNET18()
            result = model.predict(image)
            self.MyCollector(model)
            print(result)
            return result
        #MobileNetV2
        elif MyMagicObject == 2:
            model = EngineMobileNetV2()
            result = model.predict(image)
            self.MyCollector(model)
            print(result)
            return result
        #SE-ResNet18
        elif MyMagicObject == 3:
            model = EngineSEResNetClassifier()
            result = model.predict(image)
            self.MyCollector(model)
            print(result)
            return result
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
