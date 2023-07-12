from presidio_analyzer import EntityRecognizer, RecognizerResult, AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from typing import List

class TransformersRecognizer(EntityRecognizer):
    def __init__(self,
                 model_id_or_path=None,
                 aggregation_strategy="average",
                 supported_language="fr",
                 ignore_labels=["O","MISC"]):
      
        self.tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
        self.model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
        self.pipeline = pipeline('ner', model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
 
        self.label2presidio={
        "PER": "PERSON",
        "LOC": "LOCATION",
        "ORG": "ORGANIZATION"
        }

        super().__init__(supported_entities=list(self.label2presidio.values()),supported_language=supported_language)

    def load(self) -> None:
        """No loading is required."""
        pass

    def analyze(
        self, text: str, entities: List[str] = None, nlp_artifacts=None
    ) -> List[RecognizerResult]:
        """
        Extracts entities using Transformers pipeline
        """
        results = []

        # keep max sequence length in mind
        predicted_entities = self.pipeline(text)
        if len(predicted_entities) >0:
          for e in predicted_entities:
            converted_entity = self.label2presidio[e["entity_group"]]
            if converted_entity in entities or entities is None:
              results.append(
                  RecognizerResult(
                      entity_type=converted_entity,
                      start=e["start"],
                      end=e["end"],
                      score=e["score"]
                      )
                  )
        return results




if __name__ == "__main__":
    

    transformer = TransformersRecognizer()

    analyzer = AnalyzerEngine(supported_languages=["fr"])
    #analyzer.registry.add_recognizer(transformer)

    
    text =" Je suis le Docteur Chepard et j'ai vu en consultation madame Jeanne Dupond pour un AVC ischémique"

    #results = analyzer.analyze(text=sentenses, entities=["PERSON"], language="fr")

    #print(results)




