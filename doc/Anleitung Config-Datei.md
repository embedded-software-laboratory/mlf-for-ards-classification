# Anleitung zur Config-Datei

Mit der Konfigurationsdatei "config.yml" kann gesteuert werden, welche Schritte das Framework durchführen soll und es können diverse Parameter eingestellt werden, die das Endergebnis beeinflussen.

## process

In diesem Abschnitt wird festgelegt, welche Schritte ausgeführt werden sollen. Jeder Unterpunkt kann auf "True" (wird ausgeführt) oder auf "False" (wird nicht ausgeführt) gesetzt werden.

1. **load_models**: Hierüber kann gesteuert werden, ob gespeicherte Modelle geladen werden sollen oder nicht. Falls dies auf True gesetzt wird, werden die unter dem Punkt "loading_paths" angegebenen Pfade für das Laden der Modelle verwendet.
2. **load_timeseries_data**: Hierüber kann gesteuert werden, ob Zeitreihendaten geladen werden sollen. Falls dies auf True steht, wird die Datei, die unter dem Punkt "data" angegeben ist, geladen.
3. **perform_imputation**: Falls True, werden fehlende Werte in den geladenen Daten imputiert. Im Bereich "data_processing/imputation" wird genauer eingestellt, wie die Imputation ablaufen soll. 
	Komplett leere Spalten können nicht imputiert werden und werden vom Framework automatisch entfernt.
	Wenn in einigen Spalten keine Werte vorhanden sind, kann dies im weiteren Verlauf zu Abstürzen führen. Daher ist es vermutlich i.d.R. sinnvoll, diesen Punkt zu aktivieren und alle Parameter zu imputieren. 
4. **perform_unit_conversion**: Falls True, wird das Framework versuchen, Einheiten von Parametern in die vom Catalog of Items vorgegebenen Einheiten umzurechnen. Dafür muss beim Punkt "data/database" die Datenbank angegeben werden, aus denen die geladenen Daten stammen. Das Framework nimmt dann an, dass jeder Parameter in der Einheit kommt, die in der Relevanztabelle für die entsprechende Datenbank angegeben ist.
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
5. **calculate_missing_params**: Falls True, wird das Framework versuchen, fehlende Parameter aus den vorhandenen Daten zu berechnen. Unter "data_processing/params_to_calculate" wird angegeben, welche Parameter noch berechnet werden sollen.
6. **perform_ards_onset_detection**: Falls True, wird das Framework zu jedem einzelnen Patienten in den Daten (identifiziert über die Spalte "patient_id" bestimmen, wann vermutlich ARDS erstmalig aufgetreten ist und diesen Zeitpunkt oder eine gewisse Zeitspanne darum herum zurückgeben. Nach welcher Regel genau der ARDS-Beginn bestimmt werden soll und welche Daten genau zurückgegeben werden sollen, kann im Punkt "data_processing/ards_onset_detection" festgelegt werden.
	Für die bessere Vergleichbarkeit wird der am ehesten passende Zeitpunkt auch bei den Nicht-ARDS-Patienten rausgesucht. 
7. **perform_feature_selection**: Um die Geschwindigkeit des Trainingsprozesses zu erhöhen, kann mithilfe der Feature Selection berechnet werden, welche Parameter einen besonderes hohen bzw. geringen Einfluss auf den Referenz-Parameter haben. Anschließend werden Parameter mit geringem Einfluss entfernt. Einstellungen zum Feature-Selection-Prozess werden unter "feature_selection" vorgenommen.
8. **perform_data_segregation**: Mit diesem Schritt werden die geladenen Daten in einen Datensatz fürs Modelltraining und einen Datensatz für die Evaluation/Prediktion aufgeteilt. Unter "data_segregation" können weitere Einstellungen vorgenommen werden. 
	Diese Schritt muss nur aktiviert werden, wenn in einem Durchlauf die Modelle trainiert und anschließend evaluiert werden sollen. Falls dieser Schritt deaktiviert wird, wird der komplette geladene Datensatz sowohl für die Klassifizierung und Evaluation als auch fürs Training verwendet; daher ergibt eine Deaktivierung der Aufteilung nur Sinn wenn entweder Klassifizierung/Evaluation oder Modell-Training durchgeführt werden soll. 
9. **perform_timeseries_training**: Wenn dies aktiviert wird, werden alle verfügbaren Zeitreihenmodelle mit den geladenen Daten trainiert. Wenn im gleichen Durchlauf eine Evaluation durchgeführt werden soll, sollte der Punkt "perform_data_segregation" ebenfalls aktiviert werden. 
10. **perform_timeseries_classification**: Falls aktiviert, werden werden alle verfügbaren Zeitreihenmodelle die geladenen Daten klassifizieren und für jede Zeile entsprechend 1 für "ARDS" oder 0 für "Nicht ARDS" ausgeben.
11. **calculate_evaluation_metrics**: Falls aktiviert, wird das Framework die geladenen Testdaten dazu verwenden, die Modelle diese klassifizieren zu lassen und zu berechnen, wie gut die Ergebnisse der Modelle sind. Dafür werden verschiedene Metriken berechnet. 
12. **perform_cross_validation**: Falls aktiviert, werden die Modelle mit den geladenen Daten kreuzvalidiert. Hierfür können unter "evaluation" noch einige Parameter eingestellt werden. 
13. **save_models**: Falls True, werden alle trainierten Modelle im Ordner "Save" gespeichert. 

## timeseries_models_to_execute

Hier wird festgelegt, welche Zeitreihenmodelle vom Framework berücksichtigt werden sollen. Jedes zu berücksichtigende Modell wird als eigener Unterpunkt aufgeführt. Der Unterpunkt muss hierbei der Name der entsprechenden Klasse sein. 

## loading_paths

Hier werden die Pfade angegeben, von wo die Modelle geladen werden sollen, falls im Bereich "process" der Punkt "load_models" auf True gesetzt wurde.
Jeder Unterpunkt gibt den genauen Dateipfad für das zu ladende Modell an. Der Unterpunkt muss exakt den Namen des Modells tragen, der im Konstruktor des entsprechenden Modells in die Variable "self.name" geschrieben wird. 
Die angegebenen Pfade dürfen die Dateiendung nicht enthalten, da diese von der load-Methode jeden Modells automatisch ergänzt wird. 
Falls als Pfad "default" angegeben wird oder ein Modell in dieser Liste gar nicht auftaucht, wird das Framework versuchen, das Modell den Standard-Speicherpfad (der Ordner "Save") zum Laden zu verwenden. 

## data

Hier wird definiert, wo die Patientendaten liegen, die geladen werden und zum Training oder zur Klassifizierung / Evaluation verwendet werden sollen. 
Mit dem Unterpunkt "file_path" wird der Dateipfad angegeben. Aktuell werden npy-Dateien und csv-Dateien unterstützt in dem Format, dass vom [Data-Extractor](https://git-ce.rwth-aachen.de/smith-project/ARDS-MLP/data-basis/data-extraction) erzeugt wird. Falls npy verwendet wird, muss ebenfalls die dazugehörige vars-Datei, die vom Data-Extractor beim Download erzeugt wird, im gleichen Ordner liegen.
Im Unterpunkt "database" wird angegeben, aus welcher Datenbank die Daten kommen. Dies ist für die Umrechnung der Einheiten relevant, s.o. Aktuell unterstützt werden "eICU", "MIMIC3", "MIMIC4" und "UKA".

## data_processing

In diesem Bereich werden einige Einstellungen vorgenommen, die genau festlegen, wie genau das Data-Preprocessing abläuft.

1. **imputation**: Hier wird festgelegt, bei welchen Parametern fehlende Daten imputiert werden sollen und welcher Imputationsalgorithmus für jeden Paramter verwendet werden soll. Zur Verfügung stehen die folgenden Algorithmen:
	- **forward**: Fehlende Werte werden mit dem letzten bekannten Wert aufgefüllt. Falls ganz zu Beginn fehlende Werte auftauchen, es also keinen vorherigen bekannten Wert gibt, wird der erste vorhandene Wert für diese Lücken verwendet.
	- **backfill**: Fehlende Werte werden mit dem nächsten bekannten Wert aufgefüllt. Falls ganz am Ende fehlende Werte auftauchen, es also keinen nächsten bekannten Wert gibt, wird der letzte vorhandene Wert für diese Lücken verwendet.
	- **mean**: Von allen vorhandenen Werten wird der Durchschnitt berechnet. Alle Lücken werden mit diesem Durchschnitt aufgefüllt. 
	- **linear_interpolation**: Fehlende Werte werden durch lineare Interpolation zwischen den zwei nächsten Nachbarn berechnet. Randwerte werden durch forward bzw. backfill aufgefüllt. 
	
	Einige Parameter stellen einen binären Zustand dar, z.B. weil sie angeben, ob eine bestimmte Diagnose gestellt wurde oder nicht. Aktuell sind dies die Parameter "ards", "heart-failure", "hypervolemia", "mech-vent", "pneumonia", "xray", "sepsis", "chest-injury". Bei diesen dürfen ausschließlich die Methoden "forward" oder "backfill" verwendet werden.
	
	Mit "impute_empty_columns" wird festgelegt, was mit Spalten passieren soll, in denen auch nach der Imputation noch leere Werte vorhanden sind. Dies kann passieren, wenn bestimmte Werte für einen Patienten komplett fehlen. Falls False, werden leere Spalten entfernt. Falls True, werden alle fehlenden Werte in solchen Spalten auf -100000 gesetzt. Falls False, werden alle Spalten, in denen nur NaN-Werte vorkommen, sowie alle Zeilen, in denen mind. ein NaN-Wert vorkommt, gelöscht.
	Unter "default_imputation_method" wird die Standard-Imputationsmethode angegeben, die für alle zu imputierenden Parameter verwendet wird, für die keine separate Methode angegeben wurde.
	Unter "params_to_impute" werden alle Parameter angegeben, die imputiert werden sollen. Wenn hier "all" angegeben wird, werden alle Parameter inmputiert. Für jeden Parameter kann ebenfalls der Imputationsalgorithmus angegeben werden, der für speziell diesen Parameter verwendet werden soll. Die Syntax dafür sieht so aus: "- ards, forward".
	Fehlende Werte in den Daten können zu Abstürzen des Programms führen. Daher ist es sinnvoll, immer alle Parameter zu imputieren. 
	
2. **params_to_calculate**: Hier wird angegeben, welche Parameter aus den vorhandenen Daten berechnet werden sollen. Aktuell sind die folgenden Parameter möglich:
	- delta-p (hierfür müssen die Parameter "p-ei" und "peep" in den Daten gegeben sein).
	- tidal-vol-per-kg (hierfür müssen die Parameter "height", "weight" und "tidal-volume" angegeben werden. Wünschenswert wäre zudem noch der Paramter "gender". Falls das Geschlecht nicht angegeben ist, wird die Formel für das männliche Geschlecht verwendet.)
	- liquid-balance (hierfür müssen die Parameter "liquid-input" und "liquid-output" gegeben sein)
	- lymphocytes_abs (hierfür müssen die Parameter "lymphocytes (percentage)" und "leucocytes" gegeben sein)
	- horovitz (hierfür werden die Parameter "fio2" und "pao2" benötigt)
	- i-e (hierfür werden die Parameter inspiry-time und expiry-time benötigt
	- lymphocytes (relative) (hierfür werden die Parameter lymphocytes_abs und leucocytes benötigt)
	Parameter, die berechnet werden sollen, aber bereits in den Daten vorhanden sind, werden übersprungen.
	Wenn einer der Parameter, die zur Berechnung eines neuen Parameters gebraucht wird, nicht in den Daten gegeben ist, wird das Framework eine Fehlermeldung ausgeben. 
	
	**ards_onset_detection**: Hier wird eingestellt, nach welcher Regel der Beginn eines ARDS-Verlaufs erkannt werden soll und welche Daten genau zurückgegeben werden sollen.
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

Hier können einige Einstellungen für die Kreuzvalidierung vorgenommen werden.
- **n_splits**: Hier wird die Anzahl an Teilmengen, in die die Testdaten zur Kreuzvalidierung aufgeteilt werden sollen, angegeben.
- **shuffle**: (True oder False) - hierüber wird angegeben, ob die Daten vor dem Aufteilen gemischt werden sollen. 