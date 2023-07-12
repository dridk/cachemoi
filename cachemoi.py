from presidio_analyzer import EntityRecognizer, RecognizerResult, AnalyzerEngine
from presidio_analyzer.nlp_engine import TransformersNlpEngine, nlp_engine
from presidio_anonymizer import AnonymizerEngine

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from typing import List



if __name__ == "__main__":
    
    
    models = {
            "fr": {"spacy": "fr_core_news_sm", "transformers": "Jean-Baptiste/camembert-ner"}
            }
    engine = TransformersNlpEngine(models)

    analyzer = AnalyzerEngine(nlp_engine=engine, supported_languages=["fr"]) 
    
    text = "Docteur Cherpard a vu en consultation ce matin Madame Jeanne Dupond. Mr Endocardite rentre chez lui après une endocardite "

    results = analyzer.analyze(text=text, entities=["PERSON"], language="fr")

    
    anonymizer = AnonymizerEngine()
    an_text = anonymizer.anonymize(text=text, analyzer_results=results).text

    print(text)
    print(an_text)

