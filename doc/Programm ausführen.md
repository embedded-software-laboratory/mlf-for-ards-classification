# Ausführen des Frameworks

## Erstmalige Installation

Bei der erstmaligen Benutzung des Programms müssen folgende Schritte durchgeführt werden:
1. Anaconda herunterladen und installieren (https://www.anaconda.com/download/)
2. Anaconda-Konsole öffnen und in den Implementation-Ordner navigieren
3. Die Anaconda-Umgebung installieren mit folgender Anweisung: conda env create -f environment.yml

Alternativ können auch Python und alle benötigten Pakete per Hand installiert werden, aber das ist deutlich aufwändiger. 

## Das Framework ausführen

Sobald die o.g. Installationsschritte ausgeführt wurden, kann das Programm wie folgt gestartet werden:
1. Anaconda-Konsole öffnen und in den Implementation-Ordner navigieren
2. Die Anaconda-Umgebung aktivieren, mit folgender Anweisung: conda activate mlp_framework
3. Die Anweisung *python main.py -f config.yml* ausführen

## Fehlerbehenung

Nach erfolgter Installation kann möglicherweise bei der Programmausführung ein Fehler der folgenden Art auftreten:
 File "C:\Users\Name\.conda\envs\mlp_framework\lib\site-packages\torch\__init__.py", line 130, in <module> raise err OSError: [WinError 182] Das Betriebssystem kann %1 nicht ausführen. Error loading "C:\Users\Name\.conda\envs\mlp_framework\lib\site-packages\torch\lib\fbgemm.dll" or one of its dependencies.
 
Zur Behebung dieses Problems hat es sich bewährt, die folgenden Anweisung in der Anaconda-Konsole in der angegebenen Reihenfolge auszuführen:
1. conda install xgboost
2. conda install lightgbm
3. conda install cudatoolkit
4. conda uninstall cudatoolkit
5. pip install torch==2.2.1
6. pip install torchvision==0.17
7. pip install torchvision==0.15
8. conda uninstall pytorch
9. conda install pytorch==2.0.1
10. conda install torchmetrics
11. conda install timm
12. conda install statsmodels

Wo genau in dieser Abfolge die Magie liegt, die das Problem behebt, ist zurzeit leider nicht bekannt. 