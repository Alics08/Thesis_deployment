import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
from keras.preprocessing.sequence import pad_sequences

# Initialization    
class LegalChatbotTrainer:
    def __init__(self, intents_file='data.json'):
        self.lemmatizer = WordNetLemmatizer()
        self.intents = self.load_intents(intents_file)
        self.words, self.classes, self.documents = self.preprocess_data()
        self.max_sequence_length = max(len(bag) for bag, _ in self.documents)
        self.model = self.build_model()
    
    def load_intents(self, filename):
        with open(filename, 'r') as file:
            intents = json.load(file)
        return intents
    
    # Preprocessing Stage
    def preprocess_data(self):
        words = []   
        classes = []
        documents = []
        ignore_words = set(nltk.corpus.stopwords.words('english'))
        crime_related_tags = [
            'theft', 'robbery', 'assault', 'fraud', 'homicide', 'drug', 'violence', 'kidnapping', 'arson',
            'harassment', 'blackmail', 'perjury', 'trespassing', 'property_damage', 'corruption', 'oral_defamation',
            'plain_view_doctrine', 'illegal_detention', 'narcotics', 'slight_physical_injury', 'child_abuse', 'cybercrime',
            'rape', 'arbitrary_detention', 'physical_injury', 'bigamy', 'treason', 'espionage', 'estafa', 'rebellion_insurrection', 
            'sedition', 'illegal_assemblies', 'direct_assaults', 'indirect_assaults', 'evasion_of_sentence', 'using_false_certificates', 
            'grave_scandal', 'vagrants_and_prostitutes', 'infanticide', 'disturbance_of_proceedings', 'chattel_mortgage', 'usurpation', 
            'qualified_theft', 'grave_coercion', 'grave_threats', 'abandoning_minor', 'abortion', 'alarm_and_scandal', 'cyberlibel', 
            'marital_rape', 'statutory_rape', 'parricide', 'false_testimony_civil_cases', 'machinations_in_public_auctions', 
            'false_testimony_favorable_to_defendants', 'false_testimony_against_defendant', 'illegal_use_of_uniforms_or_insignia', 
            'fictitious_name_concealing_true_name', 'false_certificates', 'falsification_by_private_individual', 'falsification_by_public_officer', 
            'forged_signature_counterfeit_seal', 'delivery_of_prisoners', 'tumults', 'murder', 'frustrated_homicide', 'reckless_driving', 'plagiarism',
            'impossiblen crime','burglary','acts of lasciviousness'
        ]

        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word_list = nltk.word_tokenize(pattern.lower())
                words.extend(word_list)
                documents.append((word_list, intent['tag']))
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

        words = [self.lemmatizer.lemmatize(word) for word in words if word not in ignore_words]
        words = sorted(set(words))
        classes = sorted(set(classes))

        training = []
        output_empty = [0] * len(classes)

        for document in documents:
            bag = []
            word_patterns = document[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            for word in words:
                if word in word_patterns:
                    if any(tag in word for tag in crime_related_tags):
                        bag.append(2) 
                    else:
                        bag.append(1)
                else:
                    bag.append(0)

            output_row = list(output_empty)
            output_row[classes.index(document[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        return words, classes, training
    
    # Neural Network Model Building
    def build_model(self):
        model = Sequential()
        model.add(Dense(292, input_shape=(self.max_sequence_length,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(289))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Dense(len(self.classes)))
        model.add(Activation('softmax'))

        adam = Adam(learning_rate=0.006)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        return model
    
    # Model Training
    def train_model(self, model_path='chatbot_model.h5', words_file='words.pkl', classes_file='classes.pkl'):
        train_x = np.array([bag for bag, _ in self.documents])
        train_y = np.array([output_row for _, output_row in self.documents])
        self.model.fit(train_x, train_y, epochs=800, batch_size=32, verbose=1)
        self.model.save(model_path)
        pickle.dump(self.words, open(words_file, 'wb'))
        pickle.dump(self.classes, open(classes_file, 'wb'))
        print("Training completed and model saved.")

if __name__ == "__main__":
    trainer = LegalChatbotTrainer()
    trainer.train_model()
