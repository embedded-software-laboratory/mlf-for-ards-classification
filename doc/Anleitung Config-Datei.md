# Anleitung zur Config-Datei

Mit der Konfigurationsdatei "config.yml" kann gesteuert werden, welche Schritte das Framework durchführen soll und es können diverse Parameter eingestellt werden, die das Endergebnis beeinflussen.

## process

In diesem Abschnitt wird festgelegt, welche Schritte ausgeführt werden sollen. Jeder Unterpunkt kann auf "True" (wird ausgeführt) oder auf "False" (wird nicht ausgeführt) gesetzt werden.

1. **load_models**: Hierüber kann gesteuert werden, ob gespeicherte Modelle geladen werden sollen oder nicht. Falls dies auf True gesetzt wird, werden die unter dem Punkt "loading_paths" angegebenen Pfade für das Laden der Modelle verwendet.
2. **load_timeseries_data**: Hierüber kann gesteuert werden, ob Zeitreihendaten geladen werden sollen. Falls dies auf True steht, wird die Datei, die unter dem Punkt "data" angegeben ist, geladen.
3. **perform_anomaly_detection**: Falls True wird auf den geladenen Daten eine Anomalieerkennung durchgeführt. Die Einstellungen für die Anomalieerkennung werden unter [anomaly_detection](#anomaly_detection) erklärt.
4. **perform_imputation**: Falls True, werden fehlende Werte in den geladenen Daten imputiert. Im Bereich "data_processing/imputation" wird genauer eingestellt, wie die Imputation ablaufen soll. 
	Komplett leere Spalten können nicht imputiert werden und werden vom Framework automatisch entfernt.
	Wenn in einigen Spalten keine Werte vorhanden sind, kann dies im weiteren Verlauf zu Abstürzen führen. Daher ist es vermutlich i.d.R. sinnvoll, diesen Punkt zu aktivieren und alle Parameter zu imputieren. 
5. **perform_unit_conversion**: Falls True, wird das Framework versuchen, Einheiten von Parametern in die vom Catalog of Items vorgegebenen Einheiten umzurechnen. Dafür muss beim Punkt "data/database" die Datenbank angegeben werden, aus denen die geladenen Daten stammen. Das Framework nimmt dann an, dass jeder Parameter in der Einheit kommt, die in der Relevanztabelle für die entsprechende Datenbank angegeben ist.
	Aktuell implementiert sind die folgenden Umrechnungen:
	1. Datenbank eICU:
		- Hämoglobin: g/dL nach mmol/L
		- Kreatinin: mg/dL nach µmol/L 
		- Albumin: g/dL nach µmol/L 
		- CRP: mg/L nach nmol/L
		- etCO2: mmHg nach %
		- Bilirubin: mg/dL nach µmol/L
	2. Datenbank MIMIC-3:
		- Hämoglobin: g/dL nach mmol/L
		- Harnstoff: mg/dL nach mmol/L
		- Kreatinin: mg/dL nach µmol/L 
		- Albumin: g/dL nach µmol/L 
		- Bilirubin: mg/dL nach µmol/L
		- CRP: mg/L nach nmol/L
	3. Datenbank MIMIC-4:
		- Hämoglobin: g/dL nach mmol/L
6. **calculate_missing_params**: Falls True, wird das Framework versuchen, fehlende Parameter aus den vorhandenen Daten zu berechnen. Unter "data_processing/params_to_calculate" wird angegeben, welche Parameter noch berechnet werden sollen.
7. **perform_ards_onset_detection**: Falls True, wird das Framework zu jedem einzelnen Patienten in den Daten (identifiziert über die Spalte "patient_id" bestimmen, wann vermutlich ARDS erstmalig aufgetreten ist und diesen Zeitpunkt oder eine gewisse Zeitspanne darum herum zurückgeben. Nach welcher Regel genau der ARDS-Beginn bestimmt werden soll und welche Daten genau zurückgegeben werden sollen, kann im Punkt "data_processing/ards_onset_detection" festgelegt werden.
	Für die bessere Vergleichbarkeit wird der am ehesten passende Zeitpunkt auch bei den Nicht-ARDS-Patienten rausgesucht. 
8. **perform_filtering**: Falls aktiviert, wird das Framework die im Preprocessing-Bereich unter "filtering" eingestellten Filter ausführen, um so Patienten zu filtern, die möglicherweise ein falsches ARDS-Label haben. Genauere Erklärungen, welche Filter es gibt, werden im Preprocessing-Bereich beschrieben. 
9. **perform_feature_selection**: Um die Geschwindigkeit des Trainingsprozesses zu erhöhen, kann mithilfe der Feature Selection berechnet werden, welche Parameter einen besonderes hohen bzw. geringen Einfluss auf den Referenz-Parameter haben. Anschließend werden Parameter mit geringem Einfluss entfernt. Einstellungen zum Feature-Selection-Prozess werden unter "feature_selection" vorgenommen.
10. **perform_data_segregation**: Mit diesem Schritt werden die geladenen Daten in einen Datensatz fürs Modelltraining und einen Datensatz für die Evaluation/Prediktion aufgeteilt. Unter "data_segregation" können weitere Einstellungen vorgenommen werden. 
	Diese Schritt muss nur aktiviert werden, wenn in einem Durchlauf die Modelle trainiert und anschließend evaluiert werden sollen. Falls dieser Schritt deaktiviert wird, wird der komplette geladene Datensatz sowohl für die Klassifizierung und Evaluation als auch fürs Training verwendet; daher ergibt eine Deaktivierung der Aufteilung nur Sinn wenn entweder Klassifizierung/Evaluation oder Modell-Training durchgeführt werden soll. 
11. **perform_timeseries_training**: Wenn dies aktiviert wird, werden alle unter dem Punkt "timeseries_models_to_execute" angegebenen Zeitreihenmodelle mit den geladenen Daten trainiert. Wenn im gleichen Durchlauf eine Evaluation durchgeführt werden soll, sollte der Punkt "perform_data_segregation" ebenfalls aktiviert werden. 
12. **perform_timeseries_classification**: Falls aktiviert, werden werden alle verfügbaren Zeitreihenmodelle die geladenen Daten klassifizieren und für jede Zeile entsprechend 1 für "ARDS" oder 0 für "Nicht ARDS" ausgeben.
13. **perform_threshold_optimization**: Falls aktiviert und für das ausgewählte Modell möglich, wird nach unterschiedlichen Algorithmen die optimale Entscheidungsgrenze bestimmt, welche Algorithmen dafür verwendet werden wird unter evaluation threshold_optimization_algorithms festgelegt.
14. **calculate_evaluation_metrics**: Falls aktiviert, wird das Framework die geladenen Testdaten dazu verwenden, die Modelle diese klassifizieren zu lassen und zu berechnen, wie gut die Ergebnisse der Modelle sind. Dafür werden verschiedene Metriken berechnet. 
15. **perform_cross_validation**: Falls aktiviert, werden die Modelle mit den geladenen Daten kreuzvalidiert. Hierfür können unter "evaluation" noch einige Parameter eingestellt werden. 
16. **save_models**: Falls True, werden alle trainierten Modelle im Ordner "Save" gespeichert. 
17. **load_image_data**: Hierüber kann gesteuert werden, ob Bilddaten für das Training der Röntgenbild-Modelle geladen werden sollen. Falls True, werden die Bilder von dem unter data/image_file_path angegebenem Dateipfad geladen.
18. **train_image_models**: Falls aktiviert, werden die unter "image_models_to_execute" angegebenen Modelle für Röntenbilder trainiert. Dies umfasst das Training für die Erkennung von Lungenentzündungen sowie das Transfer-Learning zu ARDS. 
19. **test_image_models**: Falls aktiviert,w erden die trainierten Bilddaten-Modelle mit dem geladenen ARDS-Bilddatensatz evaluiert. 

## models
In diesem Abschnitt werden die Modelle, die in den unterschiedlichen Schritten des Frameworks verwendet werden sollen festgelgt.

### timeseries_models
Dieser Abschnitt enthält die Modelle, die für die Zeitreihenklassifizierung verwendet werden sollen.

Für jede Phase des Frameworks (Training, Klassifizierung, Evaluation, Cross Validation) gibt es eine eigene Liste, in der die unterschiedlichen Algorithmen aufgeführt werden. Die einzelnen Algorithmen müssen durch das Setzen von "Active: True" aktiviert werden.

Die für die einzelnen Phasen zu verwendenden Modelle werden mit ihrem Namen in einer Liste unter "Names:" aufgeführt. Für die Phasen to_train und to_cross_validate ist es darüber hinaus möglich, den Namen von Konfigurationsdateien für die Hyperparameter anzugeben. Diese Dateien müssen in dem unter "base_path_config" angegebenen Verzeichnis in einem Ordner mit dem Algorithmennamen liegen (siehe Beispielstruktur in Data). Sollte keine Konfigurationsdatei angegeben werden, so muss für dieses Modell der Wert "default" angegeben werden. Einzelnen Modelle werden sowohl in "Names" als auch in "Configs" durch ein Komma getrennt.

Es wird sichergestellt, dass Modelle, die bisher nicht durch Training verfügbar waren, in der Phase, in der sie benötigt werden, von der Festplatte geladen werden. Das Laden von der Festplatte macht nur für die Phasen to_execute und to_evaluate Sinn. Der Speicherort setzt sich dabei aus dem unter "algorithm_base_path" angegebenen Pfad und dem Namen des Modells zusammen.


## image_models_to_execute

Hier wird festgelegt, welche Bilddatenmodelle vom Framework berücksichtigt werden sollen. Jedes zu berücksichtigende Modell wird als eigener Unterpunkt aufgeführt. Der Unterpunkt muss hierbei der Name der entsprechenden Klasse sein. 

## algorithm_base_path

Hier werden die Pfade angegeben, von wo die Modelle geladen werden sollen, falls diese benötigt werden.
Jeder Unterpunkt zum Ordner an wo das zu ladende Modell liegt. Der Name der Datei wird unter "models" in der "Names" Liste der jeweiligen Phase gespeichert. Der Unterpunkt muss den Klassennamen des Algorithmus tragen zu dem das Modell gehört. 
Falls der Pfad auf "default" gesetzt wird, so wird dieser auf das Standard-Out-Verzeichnis gesetzt, welches unter "storage_path" angeben ist.

## storage_path
Dieser Pfad gibt an, wo sich das Standard-Output-Verzeichnis befindet, in dem Modelle, Daten, Metadaten und Ergebnisse gespeichert werden.
Sollte dieser Eintrag nicht gesetzt sein, so wird das Standard-Output-Verzeichnis auf den Ordner "Save/%Y-%m-%d_%H-%M-%S" im aktuellen Arbeitsverzeichnis gesetzt.
 

## data

Hier wird definiert, wo die Patientendaten liegen, die geladen werden und zum Training oder zur Klassifizierung / Evaluation verwendet werden sollen. 
Mit dem Unterpunkt "file_path" wird der Dateipfad für die Zeitreihenmodelle angegeben. Aktuell werden npy-Dateien und csv-Dateien unterstützt in dem Format, dass vom [Data-Extractor](https://git-ce.rwth-aachen.de/smith-project/ARDS-MLP/data-basis/data-extraction) erzeugt wird. Falls npy verwendet wird, muss ebenfalls die dazugehörige vars-Datei, die vom Data-Extractor beim Download erzeugt wird, im gleichen Ordner liegen.
Im Unterpunkt "database" wird angegeben, aus welcher Datenbank die Daten kommen. Dies ist für die Umrechnung der Einheiten relevant, s.o. Aktuell unterstützt werden "eICU", "MIMIC3", "MIMIC4" und "UKA".
Unter "image_file_path" wird angegeben, wo die Datensätze für die Bilddaten-Modelle liegen. Dies muss ein Ordner sein, der die folgenden Elemente enthält:
	- Einen Ordner "chexpert" und einen Ordner "mimic", der jeweils die Trainings-, und im Falle von mimic den ARDS-Testdatensatz enthält. Jeder Datensatz besteht aus einem Ordner, der die Bilder enthält ("image") und einem Ordner, der die Label enthält ("label"). 
	- Einen Ordner "models", der die trainierten Modelle speichert. Für jedes Modell muss ein Ordner vorhanden sein, und jeder dieser Modell-Ordner muss zwei Unterordner enthalten: Einen für das Modell für die Lungenentzündung ("pneumonia") und einen für die Erkennung von ARDS ("ards"). In jedem dieser Ordner muss noch ein Ordner "main" eingefügt werden.
	- Einen Ordner "results", in welchem die Evaluationsergebnisse gespeichert werden. Hier muss dieselbe Unterordnerstruktur angelegt werden wie für den Ordner "models". 
	- Ein Datei "aug_tech.txt", welche den folgenden Inhalt enthält: 
		colorinvert
		jitter
		emboss
		fog
		gamma
	Die Funktion dieser Datei bzw. der einzelnen Einträge ist mir zum Zeitpunkt der Erstellung dieser Dokumentation leider selbst nicht bekannt. 

	
Mit den Unterpunkten "pneumonia_dataset" bzw. "ards_dataset" wird angegeben, welche Datensets genau für das Pneumonie-Training bzw. das ARDS-Training verwendet werden solle. (Hier gibt es verschiedene, da in den Vorarbeiten mehrere Datensets mit jeweils unterschiedlichen Gewichtungen und Balancierungen erzeugt wurden). Es muss der Name des Ordners, in dem das Datenset liegt, angegeben werden. 


## preprocessing

In diesem Bereich werden einige Einstellungen vorgenommen, die genau festlegen, wie genau das Data-Preprocessing abläuft.

* **patients_per_process**: Gibt an wie viele Patienten in einem Prozess verarbeitet werden sollen
* **max_processes**: Gibt an wie viele Prozesse maximal parallel laufen sollen

### anomaly_detection
Hier wird der zu verwendende Anomalieerkennungsalgorithmus konfiguriert. Jeder Algorithmus hat seinen eigenen Abschnitt. Aktuell werden folgende Algorithmen unterstützt:
* Physiological Limits
* SW-ABSAD-MOD
* DeepAnt
* ALAD

Jeder dieser Algorithmen hat die folgenden Einstellungsmöglichkeiten:
* **active**: Gibt an ob der Algorithmus zur Anomalieerkennung verwendet werden soll. Es wird immer der erste Algorithmus verwendet bei dem active auf True steht
* **name**: Name des Algorithmus wie er in den Metadaten auftauchen soll
* **columns_to_check**: Liste welche Spalten auf Anomalien überprüft werden soll. Je nach verwendetem AD-Algorithmus werden hier andere Eingaben erwartet. Für Details bitte in die Implementierung gucken. Ist [] gesetzt werden alle Spalten bis auf patient_id und timestamp überprüft.
* **handling_strategy**: Wie sollen erkannte Anomalien behandelt werden. Mögliche Werte sind:
    * delete_value: Löscht den Wert der als Anomalie erkannt wurde
	* delete_than_impute: Löscht den Wert und imputiert ihn dann so wie es unter **fix_algorithm** beschrieben ist
	* delete_row_if_any_anomaly: Löscht die gesamte Zeile, wenn ein Wert als Anomalie markiert wird
	* delete_row_if_many_anomalies: Löscht die gesamte Zeile, wenn die Anzahl der Werte die als Anomalie markiert wurden **anomaly_threshold** übersteigt. Sollte die Zeile nicht gelöscht werden, werden Werte die als Anomalie erkannt wurden mit dem unter **fix_algorithm** angegeben Algorithmus imputiert
* **fix_algorithm**: Gibt an wie Werte, die wegen AD gelöscht wurden, imputiert werden. Mögliche Werte sind:
	* **forward**: Letzter nicht gelöschter Wert wird aufgefüllt
	* **backward**: Nächster nicht gelöschter Wert wird aufgefüllt
	* **interpolate**: Lineare Interpolation zwischen letztem und nächsten Wert. Achtung: Wert wird durch die Anzahl der fehldenden Datenpunkte beeinflusst
* **anomaly_threshold**: Relative Anzahl an Anomalien ab der eine Zeile gelöscht wird wenn **delete_row_if_many_anomalies** ausgewählt ist
* **supported_stages** Gibt an welche Arbeitsschritte vom AD-Algorithmus unterstützt werden. Dieser Eintrag sollte nur verändert werden, wenn für ein Modell zusätzliche Arbeitsschritte implementiert oder Arbeitsschritte entfernt werden. Mögliche Werte sind:
	* **prepare**: Arbeitsschritt, der die Daten für die Verarbeitung durch den Algorithmus vorbereitet
	* **train**: Arbeitsschritt, der die DeepLearning Modelle trainiert
	* **predict**: Arbeitsschritt, der die Anomalien auf Grundlage des gewählten Algorithmus erkennt
	* **fix**: Arbeitsschritt, der die erkannten Anomalien mit dem unter **handling_strategy** angegebenen Algorithmus behandelt
* **active_stages**: Arbeitsschritte, die ausgeführt werden sollen wenn der gewählte Algorithmus ausgeführt wird. Es handelt sich hier um eine Liste. Mögliche Einträge sind unter **supported_stages** beschrieben.
* **anomaly_data_dir**: Ordner in dem die Ergebnisse des **predict** Schritts gespeichert werden
* **prepared_data_dir**: Ordner, in dem die Ergebnisse des **prepare** Schritts gespeichert werden

Danach folgen Algorithmen spezifische Einstellungen. Für eine Erklärung dieser entweder in die Implementierung oder original Paper gucken.
DeepLearning Ansätze haben darüber hinaus folgenden Einstellungen:

* **run_dir**: Hier werden logs von Training und AD-Schritt gespeichert
* **checkpoint_dir**: Hier werden Models für die im Code angegebenen Checkpoints gespeichert
* **load_data**: Gibt an, ob im Prepare Schritt existierende Daten genutzt werden sollen oder nicht
* **save_data**: Gibt an, ob die im Prepare Schritt erzeugten Trainings/Test/Eval Datensätze bereits existierende Datensätze überschreiben sollen
* **retrain_models**: Dictionary, das angibt ob bereits existierende Modelle genutzt werden sollen oder neue Modelle trainiert werden sollen. Für mögliche Werte bitte Implementierung checken.

### filtering
Hier wird festgelegt, welche Filter zum Filtern von Patienten, die möglicherweise ein falsches ARDS-Label haben, aktiviert werden sollen. Möglich sind die Filter Strict, Lite und BD (jeweils als eigener Stichpunkt unter dem Punkt "filter").

* **Strict**: Filter Strict entfernt alle Patienten aus den Daten, die angeblich kein ARDS haben, bei denen aber ein Horovitz-Quotient unter 200 mmHg aufgezeichnet wurde.
* **Lite**: Filter Lite entfernt alle Patienten aus den Daten, die angeblich kein ARDS haben und zusätzlich weder Hypervolämie, noch ein Lungenödem noch Herzversagen, die aber trotzdem einen Horovitz-Quotienten unter 200 mmHg aufweisen. Filter Strict und Lite sollten sinnvollerweise nicht gleichzeitig verwendet werden, da Filter Lite ähnlich zu Strict ist, nur etwas weniger restriktiv.
* **BD**: Filter BD (Berlin Definition) entfernt alle Patienten aus den Daten, die angeblich ARDS haben, aber bei denen nie ein Horovitz-Quotient unter 300 mmHg gemessen wurde (widerspricht der ARDS-Definition). 

2. **imputation**: Hier wird festgelegt, bei welchen Parametern fehlende Daten imputiert werden sollen und welcher Imputationsalgorithmus für jeden Paramter verwendet werden soll. Zur Verfügung stehen die folgenden Algorithmen:
	- **forward**: Fehlende Werte werden mit dem letzten bekannten Wert aufgefüllt. Falls ganz zu Beginn fehlende Werte auftauchen, es also keinen vorherigen bekannten Wert gibt, wird der erste vorhandene Wert für diese Lücken verwendet.
	- **backfill**: Fehlende Werte werden mit dem nächsten bekannten Wert aufgefüllt. Falls ganz am Ende fehlende Werte auftauchen, es also keinen nächsten bekannten Wert gibt, wird der letzte vorhandene Wert für diese Lücken verwendet.
	- **mean**: Von allen vorhandenen Werten wird der Durchschnitt berechnet. Alle Lücken werden mit diesem Durchschnitt aufgefüllt. 
	- **linear_interpolation**: Fehlende Werte werden durch lineare Interpolation zwischen den zwei nächsten Nachbarn berechnet. Randwerte werden durch forward bzw. backfill aufgefüllt. 
	
	Einige Parameter stellen einen binären Zustand dar, z.B. weil sie angeben, ob eine bestimmte Diagnose gestellt wurde oder nicht. Aktuell sind dies die Parameter "ards", "heart-failure", "hypervolemia", "mech-vent", "pneumonia", "xray", "sepsis", "chest-injury". Bei diesen dürfen ausschließlich die Methoden "forward" oder "backfill" verwendet werden.
	
	Mit "impute_empty_cells" wird festgelegt, was mit Einträgen passieren soll, in denen auch nach der Imputation noch leere Werte vorhanden sind. Dies kann passieren, wenn bestimmte Werte für einen Patienten komplett fehlen. Falls True, werden alle fehlenden Werte in solchen Spalten auf -100000 gesetzt. Falls False, werden alle Spalten, in denen nur NaN-Werte vorkommen, sowie alle Zeilen, in denen mind. ein NaN-Wert vorkommt, gelöscht.
	Unter "default_imputation_method" wird die Standard-Imputationsmethode angegeben, die für alle zu imputierenden Parameter verwendet wird, für die keine separate Methode angegeben wurde.
	Unter "params_to_impute" werden alle Parameter angegeben, die imputiert werden sollen. Wenn hier "all" angegeben wird, werden alle Parameter inmputiert. Für jeden Parameter kann ebenfalls der Imputationsalgorithmus angegeben werden, der für speziell diesen Parameter verwendet werden soll. Die Syntax dafür sieht so aus: "- ards, forward".
	Fehlende Werte in den Daten können zu Abstürzen des Programms führen. Daher ist es sinnvoll, immer alle Parameter zu imputieren. 
	
3. **params_to_calculate**: Hier wird angegeben, welche Parameter aus den vorhandenen Daten berechnet werden sollen. Aktuell sind die folgenden Parameter möglich:
	- delta-p (hierfür müssen die Parameter "p-ei" und "peep" in den Daten gegeben sein).
	- tidal-vol-per-kg (hierfür müssen die Parameter "height", "weight" und "tidal-volume" angegeben werden. Wünschenswert wäre zudem noch der Paramter "gender". Falls das Geschlecht nicht angegeben ist, wird die Formel für das männliche Geschlecht verwendet.)
	- liquid-balance (hierfür müssen die Parameter "liquid-input" und "liquid-output" gegeben sein)
	- lymphocytes_abs (hierfür müssen die Parameter "lymphocytes (percentage)" und "leucocytes" gegeben sein)
	- horovitz (hierfür werden die Parameter "fio2" und "pao2" benötigt)
	- i-e (hierfür werden die Parameter inspiry-time und expiry-time benötigt
	- lymphocytes (relative) (hierfür werden die Parameter lymphocytes_abs und leucocytes benötigt)
	Parameter, die berechnet werden sollen, aber bereits in den Daten vorhanden sind, werden übersprungen.
	Wenn einer der Parameter, die zur Berechnung eines neuen Parameters gebraucht wird, nicht in den Daten gegeben ist, wird das Framework eine Fehlermeldung ausgeben. 
	
4.	**ards_onset_detection**: Hier wird eingestellt, nach welcher Regel der Beginn eines ARDS-Verlaufs erkannt werden soll und welche Daten genau zurückgegeben werden sollen.
	- **detection_rule**: Dies ist die Regel, nach der der ARDS-Beginn erkannt wird. Zur Auswahl stehen die folgenden Optionen:
		- *lowest_horovitz*: Der Zeitpunkt mit dem niedrigsten Horovitz-Quotienten eines Patienten
		- *first_horovitz*: Der erste Horovitz-Quotient eines Patienten, der unter 300 mmHg  liegt. Falls es keinen gibt, wird der niedrigste Horovitz-Quotient genommen.
		- *4h*, *12h*, *24h*: Der erste Zeitpunkt, an dem der Horovitz-Quotient des Patienten für die nächsten 4/12/24 Stunden unter 300 mmHg liegt. Wenn diese Bedingung nicht erfüllt werden kann, wird der Beginn des Zeitpunktest, ab dem der Horovitz-Quotient am längsten unter 300 liegt, genommen.
		- *4h_50*: Der erste Zeitpunkt, an dem mindestens 50% der Horovitz-Werte des Patienten für die nächsten 4 Stunden unter 300 mmHg liegt. Wenn es keine solche Serie gibt, wird der Beginn der Serie genommen, an dem der Prozentsatz der Horovitz-Quotienten unter 300 für die nächsten 4 Stunden am höchsten ist. 
	- **return_rule**: Hiermit wird angegeben, welche Daten genau zurückgegeben werden sollen. Zur Auswahl stehen die folgenden Regeln:
		- *datapoint*: Es wird für jeden Patienten ausschließlich der genaue Zeitpunkt des bestimmten ARDS-Patienten zurückgegeben, alle weiteren Zeilen werden nicht weiter verarbeitet.
		- *data_series_as_series*: Es werden alle Zeilen zu dem Patienten zurückgegeben, die zwischen zwei definierten Punkten liegen. Die beiden Randpunkte werden über "series_start_point" und "series_end_point" definiert, s.u.
		- *data_series_as_point*: Es werden wie bei *data_series_as_series* zunächst alle Zeilen rausgesucht, die zwischen den zwei definierten Punkten liegen. Anschließend werden alle diese Zeilen zu einem einzelnen Datensatz zusammengefügt. Das heißt, aus der folgenden Serie
			patient_id | time | horovitz | peep
			---|---|---|---
			1 | 0 | 300 | 5
			1 | 5 | 250 | 6
			
			würde die folgende Zeile werden: 
			
			patient_id | horovitz1 | peep1 | horovitz2 | peep2
			---|---|---|---|---
			1 | 300 | 5 | 250 | 6
			
			Die Spalten "time", "patient_id" und "ards" werden dabei nicht vervielfacht. Unter "time" wir anschließend lediglich der genaue Zeitpunkt des erkannten ARDS-Beginns gespeichert. 
	- **series_start_point** und **series_end_point**: Hierüber wird definiert, von wo bis wo die zurückgegebene Serie sein soll, falls als Rückgaberegel *data_series_as_series* oder *data_series_as_point* gewählt wurde. Beide geben den Beginn/das Ende der Reihe relativ zum erkannten ARDS-Beginn in Sekunden an. D.h. wenn series_start_point auf -200000 gesetzt wird, werden alle Messwerte bis 20000 Sekunden vor dem erkannten ARDS-Beginn berücksichtigt. 
		Wenn als Rückgaberegel "datapoint" gewählt wurde, ist es unwichtig, was hier eingetragen wird.
	- **remove_ards_patients_without_onset**: Es kann passieren, dass Patienten in den Daten als ARDS-Patient markiert wurde, aber kein ARDS-Beginn gemäß der o.g. Regel gefunden wird. Über diesen Parameter kann gesteuert werden, was mit diesen Patienten passieren soll - falls True, werden diese Patienten aus den Daten entfernt. Nicht-ARDS-Patienten sind davon nicht betroffen.
	- **impute_missing_rows**: Falls als Rückgaberegel "data_series_as_series" oder "data_series_as_point" gewählt wird, kann es passieren, dass der gewählte Rückgabezeitraum zu groß ist für die vorhandenen Daten. Wenn dieser Punkt hier auf True gesetzt wird, werden etwaige fehlende Zeilen ergänzt und die entsprechenden Daten mit den Werten -100000 imputiert.
	- **update_ards_values**: Falls True und falls als Rückgaberegel *data_series_as_series* gewählt wurde, wird bei jedem Patienten jeder ARDS-Wert vor dem erkannten ARDS-Beginn auf 0 und ab dem erkannten ARDS-Beginn auf 1 gesetzt. 
	
	Der Algorithmus wird ebenfalls bei Nicht-ARDS-Patienten versuchen, gemäß der oben gewählten Regel einen ARDS-Beginn zu finden. Häufig sollte aber natürlich da nichts gefunden werden. In dem Fall wird nach den o.g. Regeln verfahren, um einen Zeitpunkt zu bestimmen. Es werden auch für Nicht-ARDS-Patienten entsprechend der unter "return_rule" festgelegten Regel Daten zurückgegeben.
	
## feature_selection:
Hier kann das Verfahren gewählt werden, welches für die Feature-Selection verwendet werden soll. Je nach Verfahren werden einige weitere Parameter benötigt.
- **method**: Hier wird das genaue Verfahren angegeben. Zur Auswahl stehen:
	- *low_variance*: Parameter, deren Werte eine geringe Varianz aufweisen, werden entfernt. Die Varianzschwelle wird unter "variance" angegeben.
	- *univariate*: Wählt die k besten Parameter aus, gemäß eines univariaten, statistischen Tests. Der Wert von k muss hier festgelegt werden, s.u.
	- *recursive*: Rekursive Feature-Eliminierung. Die gewünschte Anzahl an Features, die beibehalten werden soll, kann über "k" definiert werden. Falls k nicht definiert wird, wird eine vom Feature-Selection-Verfahren empfohlene Anzahl zurückgegeben.
	- *recursive_with_cv*: Auch Rekursive Feature-Eliminierung, verwendet aber zusätzlich Kreuzvalidierung, um die optimale Anzahl an Features zu finden. Über "k" kann aber auch hier die gewünschte Anzahl an Features vorgegeben werden (muss aber nicht). 
	- *L1*: L1-basierte Feature-Selection
	- *tree*: Baum-basierte Feature-Selection
	- *sequential*: Sequentielle Feature-Selection
- **variance**: Hier wird die Varianz angegeben, unterhalb derer Parameter entfernt werden sollen, wenn die Methode "low_variance" verwendet wird.
- **k**: Hier wird die angestrebte Anzahl an Parametern, die man verwenden möchte, angegeben. Wird von den Methoden "univariate", "recursive" und "recursive_with_cv" berücksichtigt. In ersterem Fall ist die Angabe zwingend. In den letzten beiden Fällen kann die Angabe weggelassen werden, dann wählt das Verfahren selbst die optimale Anzahl an Parametern aus. 

## data_segregation:
Hier wird festgelegt, wie genau die geladenen Daten in Trainings- und Testdaten aufgeteilt werden sollen. 
- **training_test_ratio**: Hier wird angegeben, welcher Anteil der Daten für das Training verwendet werden soll. Wenn also 0.8 angegeben wird, werden 80% der Patienten fürs Training und 20% fürs Testen verwendet.
- **percentage_of_ards_patients**: Hierüber wird angegeben, wie hoch der Anteil der ARDS-Patienten sein soll. Dies gilt sowohl für den Trainingsdatensatz als auch für den Testdatensatz. Diese Einstellung führt voraussichtlich dazu, dass Daten entfernt werden müssen, um das geforderte Verhältnis zu erreichen.

## evaluation: 

- **cross_validation**: Hier können einige Einstellungen für die Kreuzvalidierung vorgenommen werden.
  - **n_splits**: Hier wird die Anzahl an Teilmengen, in die die Testdaten zur Kreuzvalidierung aufgeteilt werden sollen, angegeben.
  - **shuffle**: (True oder False) - hierüber wird angegeben, ob die Daten vor dem Aufteilen gemischt werden sollen.
- **threshold_optimization_algorithms**: Aufzählung der Algorithmen, die jeweils eine optimale Entscheidungsgrenze ermitteln
- **evaluation_metrics**: Liste der Namen der Metriken, die während der Evaluation berechnet werden
- **evaluation_name**: Name unter dem das Evaluationsergebnis angezeigt und abgespeichert werden soll

## image_model_parameters

Hier können einige Parameter definieren, die den Trainingsprozess für die Bilddatenmodelle steuern. 
- **method**: Hier wird eingestellt, welche Schichten des ursprünglichen Modells (vor dem Transfer auf das neue Problem; beim Transfer Learning werden diese Schichten eingefroren und neue Schichten dem Modell hinzugefügt) für ein finales Fine-Tuning "entfroren" werden sollen. Zur Auswahl stehen:
	- *model*: Das ganze Modell wird entfroren
	- *last_block*: Der letzte Block wird entfroren
	- *classifier*: Die Schicht des Klassifizierers wird entfroren.
- **mode**: Hiermit wird ausgewählt, ob die ursprünglichen oder die augmentierten Datensätze für das Training und das Testen verwendet werden sollen.
	- *mode1": Nicht-augmentierte Datensätze
	- *mode2": Nicht-augmentiertes Training, augmentiertes Testen
	- *mode3": Augmentiertes Training, nicht-augmentiertes Testen
	- *mode4": Augmentiertes Training und Testen
- **num_epochs_pneumonia**: Anzahl der Trainingsdurchläufe für das Pneumonie-Modell
- **num_epochs_ards**: Anzahl der Trainingsdurchläufe für das ARDS-Modell
- **batch_size_pneumonia**: Anzahl der Batches, in die der Trainingsdatensatz für das Pneumonie-Modell für einen einzigen Trainingsdurchlauf aufgeteilt wird
- **batch_size_ards**: Anzahl der Batches, in die der Trainingsdatensatz für das ARDS-Modell für einen einzigen Trainingsdurchlauf aufgeteilt wird
- **SEED_pneumonia**: Für verschiedene beim Training des Pneumonie-Modells verwendete Zufallsfunktionen manuell festgelegter Seed, um Reproduzierbarkeit sicherzustellen
- **SEED_ards**: Für verschiedene beim Training des ARDS-Modells verwendete Zufallsfunktionen manuell festgelegter Seed, um Reproduzierbarkeit sicherzustellen
- **learning_rate**: Lernrate (Hyperparameter, der vorgibt, wie stark die Gewichte des Netzwerks beim Trainingsdurchlauf angepasst werden)
- **k_folds*: Anzahl der Teilmengen, in die die Trainingsdatensätze für die Kreuzvalidierung aufgeteilt werden. 
-- **path*: Pfad zu den Datensätzen für die Bilddatenmodelle, siehe *image_file_path*