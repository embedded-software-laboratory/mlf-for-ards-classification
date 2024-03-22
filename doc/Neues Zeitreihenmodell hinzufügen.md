
# Anleitung zum hinzufügen eines KI-Modells, welches mit Zeitreihendaten trainiert wird

Um ein neues Modell in das Framework zu integrieren, muss eine neue Klasse, die alle Methoden des Interfaces "Model" (in der Datei "model_interface.py" implementiert, angelegt werden.
Folgende Schritte müssen also abgearbeitet werden:

1. Datei "neues_modell.py" anlegen ("neues_modell" sollte hierbei durch den Namen des KI-Modells ersetzt werden).
2. In dieser Datei die Klasse "NeuesModell" anlegen, welche die Klasse model_interface.Model erbt (der Name sollte hier natürlich auch angepasst werden). 
3. Alle Methoden der Klasse model_interface.Model entsprechend der untenstehenden Vorgaben implementieren. (Weitere Hilfsfunktionen können selbstverständlich auch implementiert werden.)
4. Die Datei "framework.py" öffnen und dort den Import "from neues_modell.py import NeuesModell" zu Beginn des Skripts einfügen (ans Anpassen der Namen denken).
5. Damit das Framework beim nächsten Start das Modell auch berücksichtigt, muss in der Datei config.yml der Punkt "timeseries_models_to_execute" um den Klassennamen des neuen Modells ergänzt werden sowie ggf. der Modellname mit einem entsprechenden Pfad in den Punkt "loading_paths" eingefügt werden. Für weitere Infos hierzu siehe die Anleitung zur Config-Datei.

Im Anschluss daran sollte das neu implementierte Modell zusammen mit allen anderen mit diesem Framework benutzt werden können.


## Die Methoden der Interface-Klasse

Im Folgenden wird erläutert, was jede Funktion der neuen Model-Klasse tun muss, welche Inputs sie bekommt und was sie zurückgeben muss. Es ist wichtig, sich an die Vorgaben genau zu halten, damit das Framework fehlerfrei funktioniert.

### def __init__(self): (Konstruktor)

Hier muss folgende Zeile enthalten sein:
self.name = "Neues Modell" (Der Name kann beliebig gewählt werden, sollte aber natürlich aussagekräftig sein). 
Des Weiteren muss hier eine untrainierte Instanz des KI-Modells initialisiert werden.
Weitere Variablen, die ggf. gebraucht werden, können nach Belieben definiert werden.

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

Mit dieser Methode muss sich bestimmen lassen, wie "sicher" sich das Modell mit jeder Klassifizierung ist.
data ist ein Pandas-Dataframe, genauso wie bei der Methode "predict".
Die Methode muss ein Array zurückgeben, welches für jede Zeile des Dataframes ein Array mit zwei Werten enthält. Der erste Wert gibt die Wahrscheinlichkeit an, dass es sich bei diesem Datensatz nicht um ARDS handelt, und der zweite gibt die Wahrscheinlichkeit an, dass es sich um ARDS handelt.
Beispiel: Wenn das o.g. Dataframe übergeben wird, könnte die Rückgabe wie folgt aussehen:
[[0.8, 0,2], [0.6, 0.5], [0,1]]
oder auch 
[[0.95, 0.05], [0.3, 0.7], [0.01, 0.99]]

### def save(self, filepath):

Diese Methode speichert das aktuelle Modell an den übergebenen Speicherort. 
filepath gibt den Dateipfad an, **der jedoch noch um die richtige Dateiendung(z.B. .txt) ergänzt werden muss**. Da hier von verschiedenen Bibliotheken, die KI-Modelle bereitstellen, verschiedene Verfahren zum Speichern der Modelle verwenden, kann das nicht einheitlich vom Framework vorgegeben werden.
Die Methode gibt nichts zurück.

### def load(self, filepath):

Die Methode öffnet die Datei vom angegebenen Dateipfad und speichert das darin gespeicherte Modell in der Variable self.model. 
**Auch hier muss der Dateipfad noch um den die richtige Dateiendung, die auch in der Methode save gebraucht wird, ergänzt werden.**