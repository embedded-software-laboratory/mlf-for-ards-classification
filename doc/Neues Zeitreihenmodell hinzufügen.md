
# Anleitung zum hinzufügen eines KI-Modells, welches mit Zeitreihendaten trainiert wird

Um einen neuen Klassifikations-Algorithmus in das Framework zu integrieren, muss eine neue Klasse, die alle Methoden des Interfaces "Model" (in der Datei "model_interface.py" implementiert, angelegt werden.
Folgende Schritte müssen also abgearbeitet werden:

1. Datei "neues_modell.py" anlegen im Ordner "ml_models" ("neues_modell" sollte hierbei durch den Namen des KI-Algorithmus ersetzt werden).
2. In dieser Datei die Klasse "\<Name\>Model" (der Name sollte hier natürlich auch angepasst werden) anlegen, welche die Klasse timeseries_model.TimeSeriesModel oder timeseries_model.TimeSeriesProbaModel erbt (Zweiteres sollte nur verwendet werden, falls das zu implementierende Modell die möglichkeit bietet Klassifikationswahrscheinlichkeiten auszugeben, z.B. zu erkennen an der Existenz der Methode predict_proba). 
3. Alle Methoden der geerbten Klasse entsprechend der untenstehenden Vorgaben implementieren. (Weitere Hilfsfunktionen können selbstverständlich auch implementiert werden.)
4. Die neue Klasse in der Datei "\_\_init__.py" im Ordner "ml_models" importieren.
5. Die neue Klasse an folgenden Stellen importieren:
    - In der Datei "MLModelManager.py" im Ordner "ml_models".
6. Einträge für den neuen Algorithmus in der "config.yml" angelegen.
   - Unter "models.timeseries_models" die neue Klasse eintragen ohne "Model" am Ende für jede der Phasen einfügen. Dabei die jeweiligen Unterpunkte hinzufügen.
   - Unter "algorithm_base_path" die neue Klasse ohne Model am Ende eintragen und den Pfad zum Speicherort der Modelle eintragen. Gibt es keinen speziellen Pfad so muss "default" eingetragen werden.
   - Unter "supported_algorithms.timeseries_models" die neue Klasse eintragen.

Im Anschluss daran sollte das neu implementierte Modell zusammen mit allen anderen mit diesem Framework benutzt werden können.


## Die Methoden der Interface-Klasse

Im Folgenden wird erläutert, was jede Funktion der neuen Model-Klasse tun muss, welche Inputs sie bekommt und was sie zurückgeben muss. Es ist wichtig, sich an die Vorgaben genau zu halten, damit das Framework fehlerfrei funktioniert.

### def \__init__(self): (Konstruktor)

Hier muss folgende Zeile enthalten sein:
- self.name = "Neues Modell" (Der Name kann beliebig gewählt werden, sollte aber natürlich aussagekräftig sein).
- self.algorithm = "Neues modell" (Der Algorithmusname sollte hierbei dem Namen der Klasse entsprechen ohne "Model" am Ende).
- self.hyperparameters = {} (Hier werden die Hyperparameter des Modells in einem Dictionary gespeichert. Die einzelnen Parameter werden auf die Standardwerte gesetzt, die das Modell hat, wenn es initialisiert wird. Diese Werte können später angepasst werden. Die Namen der Parameter sollten dabei den Namen entsprechen, den das Modell intern verwendet. Gucke dafür in der Dokumentation des Modells nach. Die Werte sollten dabei in der Form "Parametername": Wert angegeben werden. Beispiel: {"n_estimators": 100, "max_depth": 5}).
- self.model = self._init_model() (Hier wird das Modell initialisiert. Diese Methode muss in der Klasse implementiert werden und das Modell zurückgeben)

Weitere Variablen, die ggf. gebraucht werden, können nach Belieben definiert werden.

### def _init_model(self):
Hier wird das Modell initialisiert und zurückgegeben.

### def train_model(self, training_data):

Mit dieser Methode wird die Instanz des KI-Modells mit den übergebenen Testdaten trainiert. 
training_data ist ein Pandas-Dataframe, welches die Trainingsdaten enthält. Jede Spalte des Dataframes gibt einen Parameter an und jede Zeile einen Messzeitpunkt. Die Spalte "ards" ist auf jeden Fall vorhanden.

### def predict(self, patient_data): 

Diese Methode lässt das trainierte Modell die übergebenen Daten klassifizieren.
patient_data ist ein Pandas-Dataframe, welches die Daten enthält, die klassifiziert werden sollen. Jede Spalte gibt wieder einen Parameter an und jede Zeile einen Messzeitpunkt. Die Spalten sind im Wesentlichen die gleichen wie beim training_data-Dataframe in der train_model-Methode, es fehlt jedoch die Spalte "ards", da diese hier ja natürlich nicht mehr gebraucht wird. 
Die Methode muss eine Liste zurückgeben, welches für jede Zeile des übergebenen Dataframes entweder den Wert "0" (kein ARDS) oder den Wert "1" ("ARDS") enthält. 
Beispiel: Wenn das übergebene Dataframe wie folgt aussieht:

index | patient_id | horovitz | peep
---|---|---|---
0 | 1 | 300 | 5
1 | 2 | 200 | 3
2 | 3 | 100 | 7

dann könnte die Rückgabe z.B. wie folgt aussehen: [0,1,1]
oder auch [0,0,1] (das hängt dann natürlich davon ab, welche Vorhersage das Modell trifft). 

### def predict_proba(self, data):
Diese Methode muss nur Implementiert werden wenn die Klasse von TimeSeriesProbaModel erbt, d.h. der Algorithmus hat die Möglichkeit in Wahrscheinlichkeiten statt binären Labels vorherzusagen.

Mit dieser Methode muss sich bestimmen lassen, wie "sicher" sich das Modell mit jeder Klassifizierung ist.
data ist ein Pandas-Dataframe, genauso wie bei der Methode "predict".
Die Methode muss ein Array zurückgeben, welches für jede Zeile des Dataframes ein Array mit zwei Werten enthält. Der erste Wert gibt die Wahrscheinlichkeit an, dass es sich bei diesem Datensatz nicht um ARDS handelt, und der zweite gibt die Wahrscheinlichkeit an, dass es sich um ARDS handelt.
Beispiel: Wenn das o.g. Dataframe übergeben wird, könnte die Rückgabe wie folgt aussehen:
[[0.8, 0,2], [0.6, 0.5], [0,1]]
oder auch 
[[0.95, 0.05], [0.3, 0.7], [0.01, 0.99]]

### def get_params(self):
Diese Methode gibt die aktuellen Hyperparameter des Modells zurück.

### def set_params(self, params):
Diese Methode setzt die Hyperparameter des Modells auf die übergebenen Werte. Dabei werden nur Parameter gesetzt die in self.hyperparameters als Schlüssel vorhanden sind.

### def save_model(self, filepath):

Diese Methode speichert das aktuelle Modell an den übergebenen Speicherort. 
filepath gibt den Dateipfad an, **der jedoch noch um die richtige Dateiendung(z.B. .txt) ergänzt werden muss**. Da hier von verschiedenen Bibliotheken, die KI-Modelle bereitstellen, verschiedene Verfahren zum Speichern der Modelle verwenden, kann das nicht einheitlich vom Framework vorgegeben werden.
Die Methode gibt nichts zurück.

### def load_model(self, filepath):

Die Methode öffnet die Datei vom angegebenen Dateipfad und speichert das darin gespeicherte Modell in der Variable self.model. 
**Auch hier muss der Dateipfad noch um den die richtige Dateiendung, die auch in der Methode save gebraucht wird, ergänzt werden.**

### def has_predict_proba(self):
Diese Methode gibt zurück ob der Algorithmus die Möglichkeit hat Wahrscheinlichkeiten hervorzusagen. Wenn ja, wird True zurückgegeben, ansonsten False.