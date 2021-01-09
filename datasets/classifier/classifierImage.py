class ClassfierImage:
    def __init__(self, index, image_path, classifier_id):
        self.index = index
        self.image_path = image_path
        self.classifier_id = classifier_id
    def getImage(self):
        return self.index, self.image_path ,self.classifier_id
    def print(self):
        print("index[%s], image_path[%s] ,classifier_id[%s]\n"%(self.index, self.image_path ,self.classifier_id))