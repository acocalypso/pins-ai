"""New train-only natural chat, independently authored from the MCP contracts.

Prompts deliberately omit tool names and JSON. Positions, names, hardware state
and job identifiers below are synthetic fixtures, not observatory measurements.
The ten-language rows share an intent group; these are training data only.
"""
from __future__ import annotations

from chat_dataset_common import assistant, conversation, tool_call, user
from v2_contracts import legacy_definitions as definitions


LOCALES = ('en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'pl', 'ja', 'zh')

# Each tuple is independently phrased plain chat in LOCALES order. Repeated
# argument fixtures make translations comparable; richer EN/DE rows below vary
# arguments, units, wording and conversation history instead of repeating JSON.
PROMPTS = {
    'get_status': (
        'Is my equipment connected? Show me what is running right now.',
        'Ist meine Ausrüstung verbunden? Zeig mir, was gerade läuft.',
        'Mon matériel est-il connecté ? Montre-moi ce qui tourne en ce moment.',
        '¿Está conectado mi equipo? Muéstrame qué está funcionando ahora.',
        'La mia attrezzatura è collegata? Mostrami cosa sta funzionando adesso.',
        'O meu equipamento está ligado? Mostra-me o que está em execução agora.',
        'Is mijn apparatuur verbonden? Laat zien wat er nu draait.',
        'Czy mój sprzęt jest połączony? Pokaż, co teraz działa.',
        '機材は接続されていますか？今何が動いているか見せてください。',
        '我的设备都连接了吗？告诉我现在有哪些任务正在运行。',
    ),
    'get_recent_images': (
        'Show the last eight pictures with their star counts and sharpness.',
        'Zeig die letzten acht Bilder mit Sternanzahl und Schärfe.',
        'Montre les huit dernières images avec le nombre d’étoiles et leur netteté.',
        'Enséñame las últimas ocho imágenes con el número de estrellas y la nitidez.',
        'Mostrami le ultime otto immagini con il numero di stelle e la nitidezza.',
        'Mostra as últimas oito imagens com a contagem de estrelas e a nitidez.',
        'Toon de laatste acht beelden met het aantal sterren en de scherpte.',
        'Pokaż ostatnie osiem zdjęć wraz z liczbą gwiazd i ostrością.',
        '直近の8枚の画像を、星の数と鮮明さと一緒に見せてください。',
        '显示最近八张照片，以及各自的星点数量和清晰度。',
    ),
    'get_sequence': (
        'Show the loaded imaging plan and its nested steps, down to five levels.',
        'Zeig die geladene Aufnahmesequenz mit ihren Schritten bis zur fünften Ebene.',
        'Montre la séquence chargée et ses étapes imbriquées jusqu’à cinq niveaux.',
        'Muéstrame la secuencia cargada y sus pasos anidados hasta cinco niveles.',
        'Mostrami la sequenza caricata e i suoi passaggi fino a cinque livelli.',
        'Mostra a sequência carregada e os seus passos até cinco níveis.',
        'Laat het geladen opnameplan zien met de stappen tot vijf niveaus diep.',
        'Pokaż wczytaną sekwencję i jej zagnieżdżone kroki do pięciu poziomów.',
        '読み込まれている撮影シーケンスの手順を、5階層まで見せてください。',
        '显示当前加载的拍摄序列，包括最多五层的子步骤。',
    ),
    'list_saved_sequences': (
        'Which imaging plans have I saved? Just list them; do not load anything.',
        'Welche Aufnahmesequenzen habe ich gespeichert? Nur auflisten, nichts laden.',
        'Quels plans de prise de vue ai-je enregistrés ? Liste-les sans rien charger.',
        '¿Qué planes de captura tengo guardados? Solo enuméralos, no cargues ninguno.',
        'Quali piani di ripresa ho salvato? Elencali soltanto, senza caricarli.',
        'Que planos de captura tenho guardados? Lista-os sem carregar nenhum.',
        'Welke opnameplannen heb ik opgeslagen? Alleen tonen, niets laden.',
        'Jakie plany zdjęć mam zapisane? Tylko je wymień, niczego nie wczytuj.',
        '保存済みの撮影プランは何がありますか？一覧だけで、読み込まないでください。',
        '我保存了哪些拍摄计划？只列出名称，不要加载。',
    ),
    'list_target_templates': (
        'What saved target recipes can I reuse when adding another object?',
        'Welche gespeicherten Zielvorlagen kann ich für ein weiteres Objekt verwenden?',
        'Quels modèles de cible enregistrés puis-je réutiliser pour un autre objet ?',
        '¿Qué plantillas de objetivo guardadas puedo reutilizar para otro objeto?',
        'Quali modelli di ripresa salvati posso riutilizzare per un altro oggetto?',
        'Que modelos de alvo guardados posso reutilizar para outro objeto?',
        'Welke opgeslagen doelsjablonen kan ik voor een ander object hergebruiken?',
        'Jakie zapisane szablony celu mogę wykorzystać dla kolejnego obiektu?',
        '別の天体を追加するときに使える保存済みの対象テンプレートを教えてください。',
        '添加另一个天体时，有哪些已保存的目标模板可以复用？',
    ),
    'get_job': (
        'Has job-chat-42 finished yet? Wait up to five seconds for its result.',
        'Ist job-chat-42 schon fertig? Warte höchstens fünf Sekunden auf das Ergebnis.',
        'La tâche job-chat-42 est-elle terminée ? Attends au plus cinq secondes son résultat.',
        '¿Ha terminado la tarea job-chat-42? Espera como máximo cinco segundos por el resultado.',
        'L’attività job-chat-42 è terminata? Attendi al massimo cinque secondi per il risultato.',
        'A tarefa job-chat-42 já terminou? Espera até cinco segundos pelo resultado.',
        'Is taak job-chat-42 al klaar? Wacht hoogstens vijf seconden op het resultaat.',
        'Czy zadanie job-chat-42 już się zakończyło? Poczekaj na wynik najwyżej pięć sekund.',
        '処理 job-chat-42 は終わりましたか？結果を最大5秒待って確認してください。',
        '任务 job-chat-42 完成了吗？最多等待五秒查看结果。',
    ),
    'list_jobs': (
        'I lost the task number. Show all current and recently finished background tasks.',
        'Ich habe die Auftragsnummer verloren. Zeig alle laufenden und kürzlich beendeten Hintergrundaufträge.',
        'J’ai perdu le numéro de la tâche. Montre toutes les tâches en cours et celles terminées récemment.',
        'Perdí el número de la tarea. Muestra las tareas activas y las terminadas recientemente.',
        'Ho perso il numero dell’attività. Mostra quelle in corso e quelle terminate di recente.',
        'Perdi o número da tarefa. Mostra as tarefas em curso e as concluídas recentemente.',
        'Ik ben het taaknummer kwijt. Toon alle lopende en onlangs afgeronde achtergrondtaken.',
        'Zgubiłem numer zadania. Pokaż trwające i niedawno zakończone zadania w tle.',
        '処理番号を忘れました。実行中と最近終了したバックグラウンド処理を一覧にしてください。',
        '我找不到任务编号了。列出正在运行和最近结束的后台任务。',
    ),
    'stop_all': (
        'Emergency, the telescope is moving the wrong way! Stop everything now.',
        'Notfall, das Teleskop fährt in die falsche Richtung! Sofort alles stoppen.',
        'Urgence, le télescope part dans la mauvaise direction ! Arrête tout immédiatement.',
        '¡Emergencia, el telescopio se mueve en la dirección equivocada! Detén todo ahora.',
        'Emergenza, il telescopio si muove nella direzione sbagliata! Ferma tutto subito.',
        'Emergência, o telescópio está a mover-se na direção errada! Para tudo agora.',
        'Noodgeval, de telescoop beweegt de verkeerde kant op! Stop onmiddellijk alles.',
        'Awaria, teleskop porusza się w złą stronę! Natychmiast zatrzymaj wszystko.',
        '緊急です！望遠鏡が違う方向へ動いています。今すぐ全部止めてください。',
        '紧急情况，望远镜正朝错误方向移动！立即停止所有操作。',
    ),
    'cancel_job': (
        'Cancel just job-chat-42 and wait five seconds; leave the other tasks alone.',
        'Brich nur job-chat-42 ab und warte fünf Sekunden; die anderen Aufträge bleiben unverändert.',
        'Annule uniquement job-chat-42 et attends cinq secondes ; ne touche pas aux autres tâches.',
        'Cancela solo job-chat-42 y espera cinco segundos; deja las demás tareas como están.',
        'Annulla soltanto job-chat-42 e attendi cinque secondi; lascia stare le altre attività.',
        'Cancela apenas job-chat-42 e espera cinco segundos; não alteres as outras tarefas.',
        'Annuleer alleen job-chat-42 en wacht vijf seconden; laat de andere taken met rust.',
        'Anuluj tylko job-chat-42 i poczekaj pięć sekund; nie ruszaj pozostałych zadań.',
        'job-chat-42 だけを中止して5秒待ってください。他の処理はそのままにしてください。',
        '只取消 job-chat-42，等待五秒，不要动其他任务。',
    ),
    'stop_sequence': (
        'Stop the current imaging sequence, but do not park or warm the camera.',
        'Stoppe die laufende Aufnahmesequenz, aber parke nicht und wärme die Kamera nicht auf.',
        'Arrête la séquence en cours, sans parquer la monture ni réchauffer la caméra.',
        'Detén la secuencia actual, sin aparcar la montura ni calentar la cámara.',
        'Ferma la sequenza attuale, senza parcheggiare la montatura né riscaldare la camera.',
        'Para a sequência atual, sem estacionar a montagem nem aquecer a câmara.',
        'Stop de huidige opnamesequentie, zonder te parkeren of de camera op te warmen.',
        'Zatrzymaj obecną sekwencję, ale nie parkuj montażu ani nie ogrzewaj kamery.',
        '現在の撮影シーケンスを止めてください。架台のパークやカメラの昇温はしないでください。',
        '停止当前拍摄序列，但不要停放赤道仪，也不要给相机升温。',
    ),
    'stop_guiding': (
        'Stop autoguiding only. Leave the mount tracking.',
        'Beende nur das Autoguiding. Die Montierung soll weiter nachführen.',
        'Arrête uniquement l’autoguidage. Laisse le suivi de la monture actif.',
        'Detén solo el autoguiado. Mantén el seguimiento de la montura.',
        'Interrompi solo l’autoguida. Lascia attivo l’inseguimento della montatura.',
        'Para apenas a autoguiagem. Mantém o seguimento da montagem ativo.',
        'Stop alleen het autoguiden. Laat de montering volgen.',
        'Zatrzymaj tylko autoguider. Pozostaw śledzenie montażu włączone.',
        'オートガイドだけを停止し、架台の追尾は続けてください。',
        '只停止自动导星，保持赤道仪继续跟踪。',
    ),
    'connect_device': (
        'Connect the camera selected in my current profile.',
        'Verbinde die Kamera, die in meinem aktuellen Profil ausgewählt ist.',
        'Connecte la caméra sélectionnée dans mon profil actuel.',
        'Conecta la cámara seleccionada en mi perfil actual.',
        'Collega la camera selezionata nel mio profilo attuale.',
        'Liga a câmara selecionada no meu perfil atual.',
        'Verbind de camera die in mijn huidige profiel is geselecteerd.',
        'Połącz kamerę wybraną w moim obecnym profilu.',
        '現在のプロファイルで選ばれているカメラを接続してください。',
        '连接当前配置中选中的相机。',
    ),
    'disconnect_device': (
        'The mount is parked now; disconnect it from PINS.',
        'Die Montierung ist jetzt geparkt; trenne ihre Verbindung zu PINS.',
        'La monture est maintenant parquée ; déconnecte-la de PINS.',
        'La montura ya está aparcada; desconéctala de PINS.',
        'La montatura è parcheggiata; scollegala da PINS.',
        'A montagem já está estacionada; desliga-a do PINS.',
        'De montering staat nu geparkeerd; verbreek de verbinding met PINS.',
        'Montaż jest już zaparkowany; rozłącz go z PINS.',
        '架台はパーク済みです。PINS との接続を切ってください。',
        '赤道仪已经停放好了，请断开它与 PINS 的连接。',
    ),
    'slew_mount': (
        'Point the telescope to my verified J2000 position: right ascension 7.25 hours, declination minus 18.5 degrees.',
        'Fahre zu meiner verifizierten J2000-Position: Rektaszension 7,25 Stunden, Deklination minus 18,5 Grad.',
        'Pointe le télescope vers ma position J2000 vérifiée : ascension droite 7,25 heures, déclinaison moins 18,5 degrés.',
        'Apunta el telescopio a mi posición J2000 verificada: ascensión recta 7,25 horas, declinación menos 18,5 grados.',
        'Punta il telescopio alle mie coordinate J2000 verificate: ascensione retta 7,25 ore, declinazione meno 18,5 gradi.',
        'Aponta o telescópio para a minha posição J2000 verificada: ascensão reta 7,25 horas, declinação menos 18,5 graus.',
        'Richt de telescoop op mijn gecontroleerde J2000-positie: rechte klimming 7,25 uur, declinatie min 18,5 graden.',
        'Skieruj teleskop na moje sprawdzone współrzędne J2000: rektascensja 7,25 godziny, deklinacja minus 18,5 stopnia.',
        '確認済みの J2000 座標、赤経7.25時間、赤緯マイナス18.5度へ望遠鏡を向けてください。',
        '请把望远镜转到我确认过的 J2000 坐标：赤经7.25小时，赤纬负18.5度。',
    ),
    'park_mount': (
        'Put the mount in its parked position now.',
        'Fahre die Montierung jetzt in ihre Parkposition.',
        'Place maintenant la monture dans sa position de parc.',
        'Lleva ahora la montura a su posición de aparcado.',
        'Porta ora la montatura nella posizione di parcheggio.',
        'Leva agora a montagem para a posição de estacionamento.',
        'Zet de montering nu in de parkeerstand.',
        'Ustaw teraz montaż w pozycji parkowania.',
        '架台をパーク位置へ移動してください。',
        '现在将赤道仪移到停放位置。',
    ),
    'unpark_mount': (
        'Release the mount from park so it can move again.',
        'Hebe die Parkstellung auf, damit die Montierung wieder fahren kann.',
        'Déparque la monture pour qu’elle puisse à nouveau se déplacer.',
        'Saca la montura del estado aparcado para que pueda moverse otra vez.',
        'Sblocca la montatura dal parcheggio perché possa muoversi di nuovo.',
        'Retira a montagem do estado estacionado para poder voltar a mover-se.',
        'Haal de montering uit de parkeerstand zodat die weer kan bewegen.',
        'Odparkuj montaż, aby mógł znów się poruszać.',
        '架台のパークを解除して、再び動けるようにしてください。',
        '解除赤道仪的停放状态，让它可以再次移动。',
    ),
    'home_mount': (
        'Ask the mount to find its physical home position, not its parking position.',
        'Lass die Montierung ihre physische Referenzposition suchen, nicht die Parkposition.',
        'Fais rechercher à la monture sa position d’origine physique, pas sa position de parc.',
        'Haz que la montura busque su posición física de origen, no la de aparcado.',
        'Fai cercare alla montatura la posizione fisica di riferimento, non quella di parcheggio.',
        'Faz a montagem procurar a posição física de referência, não a de estacionamento.',
        'Laat de montering haar fysieke referentiepositie zoeken, niet de parkeerstand.',
        'Każ montażowi znaleźć fizyczną pozycję bazową, a nie pozycję parkowania.',
        'パーク位置ではなく、架台の物理的な原点を検出してください。',
        '让赤道仪寻找物理原点，不是停放位置。',
    ),
    'set_tracking': (
        'Turn on sidereal tracking for the mount; do not start guiding.',
        'Schalte die siderische Nachführung der Montierung ein, ohne das Guiding zu starten.',
        'Active le suivi sidéral de la monture, sans lancer l’autoguidage.',
        'Activa el seguimiento sideral de la montura, sin iniciar el autoguiado.',
        'Attiva l’inseguimento siderale della montatura, senza avviare l’autoguida.',
        'Ativa o seguimento sideral da montagem, sem iniciar a autoguiagem.',
        'Zet het volgen op sterrensnelheid aan, zonder autoguiden te starten.',
        'Włącz śledzenie z prędkością gwiazdową, ale nie uruchamiaj autoguidera.',
        '架台の恒星時追尾を有効にしてください。オートガイドは開始しないでください。',
        '开启赤道仪的恒星跟踪，不要启动自动导星。',
    ),
    'capture_image': (
        'Take one test picture for a minute and a half, without saving it.',
        'Mach ein einzelnes Testbild mit anderthalb Minuten Belichtung, ohne es zu speichern.',
        'Prends une seule image de test avec une minute et demie de pose, sans l’enregistrer.',
        'Haz una sola imagen de prueba de un minuto y medio, sin guardarla.',
        'Scatta una sola immagine di prova da un minuto e mezzo, senza salvarla.',
        'Faz uma única imagem de teste com um minuto e meio de exposição, sem guardar.',
        'Maak één testopname van anderhalve minuut, zonder die op te slaan.',
        'Zrób jedno zdjęcie próbne z ekspozycją półtorej minuty, bez zapisywania go.',
        '1分半のテスト画像を1枚撮影してください。保存は不要です。',
        '拍一张测试照片，曝光一分半，不要保存。',
    ),
    'cool_camera': (
        'Cool the camera to minus twelve degrees Celsius over eight minutes.',
        'Kühle die Kamera innerhalb von acht Minuten auf minus zwölf Grad Celsius.',
        'Refroidis la caméra à moins douze degrés Celsius en huit minutes.',
        'Enfría la cámara a menos doce grados Celsius durante ocho minutos.',
        'Raffredda la camera a meno dodici gradi Celsius nell’arco di otto minuti.',
        'Arrefece a câmara até menos doze graus Celsius ao longo de oito minutos.',
        'Koel de camera in acht minuten af naar min twaalf graden Celsius.',
        'Schłodź kamerę do minus dwunastu stopni Celsjusza w ciągu ośmiu minut.',
        'カメラを8分かけて摂氏マイナス12度まで冷却してください。',
        '用八分钟把相机冷却到零下十二摄氏度。',
    ),
    'warm_camera': (
        'Warm the camera gradually to ambient temperature over fifteen minutes. Keep it connected.',
        'Wärme die Kamera über fünfzehn Minuten auf Umgebungstemperatur auf. Lass sie verbunden.',
        'Réchauffe progressivement la caméra à la température ambiante en quinze minutes. Garde-la connectée.',
        'Calienta gradualmente la cámara hasta la temperatura ambiente en quince minutos. Déjala conectada.',
        'Porta gradualmente la camera alla temperatura ambiente in quindici minuti. Lasciala collegata.',
        'Aquece gradualmente a câmara até à temperatura ambiente durante quinze minutos. Mantém-na ligada.',
        'Warm de camera in vijftien minuten op naar de omgevingstemperatuur. Laat haar verbonden.',
        'Ogrzej kamerę stopniowo do temperatury otoczenia przez piętnaście minut. Pozostaw ją połączoną.',
        'カメラを15分かけて周囲の温度まで昇温してください。接続は維持してください。',
        '用十五分钟让相机逐渐升到环境温度，保持连接。',
    ),
    'move_focuser': (
        'Move the focuser to the absolute position 18420 steps.',
        'Fahre den Fokussierer auf die absolute Position 18420 Schritte.',
        'Déplace le porte-oculaire à la position absolue de 18420 pas.',
        'Mueve el enfocador a la posición absoluta de 18420 pasos.',
        'Porta il focheggiatore alla posizione assoluta di 18420 passi.',
        'Move o focador para a posição absoluta de 18420 passos.',
        'Verplaats de focuser naar de absolute positie van 18420 stappen.',
        'Przesuń wyciąg do bezwzględnej pozycji 18420 kroków.',
        'フォーカサーを絶対位置18420ステップへ移動してください。',
        '把调焦器移动到绝对位置18420步。',
    ),
    'run_autofocus': (
        'Run the automatic focus routine now.',
        'Starte jetzt den automatischen Fokussierlauf.',
        'Lance maintenant la procédure de mise au point automatique.',
        'Ejecuta ahora la rutina de enfoque automático.',
        'Avvia ora la procedura di messa a fuoco automatica.',
        'Executa agora a rotina de focagem automática.',
        'Start nu de automatische scherpstelprocedure.',
        'Uruchom teraz procedurę automatycznego ustawiania ostrości.',
        '今、オートフォーカスを実行してください。',
        '现在运行自动对焦。',
    ),
    'change_filter': (
        'Switch the filter wheel to the configured Ha filter.',
        'Wechsle im Filterrad auf den konfigurierten Ha-Filter.',
        'Passe la roue à filtres sur le filtre Ha configuré.',
        'Cambia la rueda al filtro Ha configurado.',
        'Sposta la ruota portafiltri sul filtro Ha configurato.',
        'Muda a roda de filtros para o filtro Ha configurado.',
        'Zet het filterwiel op het ingestelde Ha-filter.',
        'Przełącz koło filtrowe na skonfigurowany filtr Ha.',
        'フィルターホイールを設定済みの Ha フィルターへ切り替えてください。',
        '把滤镜轮切换到已配置的 Ha 滤镜。',
    ),
    'start_guiding': (
        'Start autoguiding using the existing calibration; do not recalibrate.',
        'Starte das Autoguiding mit der vorhandenen Kalibrierung; nicht neu kalibrieren.',
        'Démarre l’autoguidage avec l’étalonnage existant, sans le refaire.',
        'Inicia el autoguiado con la calibración actual, sin volver a calibrar.',
        'Avvia l’autoguida con la calibrazione esistente, senza rifarla.',
        'Inicia a autoguiagem com a calibração atual, sem voltar a calibrar.',
        'Start autoguiden met de bestaande kalibratie; niet opnieuw kalibreren.',
        'Rozpocznij autoguiding z istniejącą kalibracją; nie kalibruj ponownie.',
        '既存のキャリブレーションを使ってオートガイドを開始してください。再キャリブレーションは不要です。',
        '使用现有校准开始自动导星，不要重新校准。',
    ),
    'set_switch': (
        'Set the dew heater at zero-based switch index 2 to a value of 40.',
        'Stelle die Taukappenheizung mit nullbasiertem Schalterindex 2 auf den Wert 40.',
        'Règle la résistance antibuée d’indice 2, en comptant depuis zéro, sur la valeur 40.',
        'Pon el calentador antirrocío de índice 2, contando desde cero, en el valor 40.',
        'Imposta la fascia anticondensa all’indice 2, contando da zero, al valore 40.',
        'Define o aquecedor antiorvalho no índice 2, contando a partir de zero, para o valor 40.',
        'Zet de dauwverwarming met schakelindex 2, geteld vanaf nul, op waarde 40.',
        'Ustaw grzałkę przeciwroszeniową o indeksie 2, licząc od zera, na wartość 40.',
        'ゼロから数えたスイッチ番号2のヒーターを値40に設定してください。',
        '把从零开始编号的2号除露加热器设为40。',
    ),
    'load_sequence': (
        'Load the saved plan Plan A, replacing the current plan. I have saved my edits. Do not start it.',
        'Lade die gespeicherte Sequenz Plan A und ersetze die aktuelle. Meine Änderungen sind gespeichert. Nicht starten.',
        'Charge le plan enregistré Plan A à la place du plan actuel. Mes modifications sont sauvegardées. Ne le démarre pas.',
        'Carga el plan guardado Plan A en lugar del actual. Ya guardé mis cambios. No lo inicies.',
        'Carica il piano salvato Plan A al posto di quello attuale. Ho salvato le modifiche. Non avviarlo.',
        'Carrega o plano guardado Plan A em vez do atual. Já guardei as alterações. Não o inicies.',
        'Laad het opgeslagen Plan A in plaats van het huidige plan. Mijn wijzigingen zijn bewaard. Start het niet.',
        'Wczytaj zapisany Plan A zamiast obecnego planu. Moje zmiany są zapisane. Nie uruchamiaj go.',
        '変更は保存済みです。現在のプランを保存済みの Plan A に置き換えてください。開始はしないでください。',
        '我的修改已保存。加载已保存的 Plan A 来替换当前计划，但不要开始拍摄。',
    ),
    'add_target': (
        'Append M31 to the loaded plan: 300 pictures, five minutes each, no filter change. Use my verified J2000 position: right ascension 0.7123 hours, declination 41.2688 degrees. Do not start imaging.',
        'Ergänze M31 in der geladenen Sequenz: 300 Aufnahmen zu je fünf Minuten, ohne Filterwechsel. Meine verifizierte J2000-Position: Rektaszension 0,7123 Stunden, Deklination 41,2688 Grad. Nicht starten.',
        'Ajoute M31 au plan chargé : 300 images de cinq minutes, sans changer de filtre. Ma position J2000 vérifiée : ascension droite 0,7123 heure, déclinaison 41,2688 degrés. Ne lance pas les prises de vue.',
        'Añade M31 al plan cargado: 300 imágenes de cinco minutos, sin cambiar de filtro. Mi posición J2000 verificada: ascensión recta 0,7123 horas, declinación 41,2688 grados. No inicies las capturas.',
        'Aggiungi M31 al piano caricato: 300 immagini da cinque minuti, senza cambiare filtro. Le mie coordinate J2000 verificate: ascensione retta 0,7123 ore, declinazione 41,2688 gradi. Non iniziare le riprese.',
        'Acrescenta M31 ao plano carregado: 300 imagens de cinco minutos, sem mudar o filtro. A minha posição J2000 verificada: ascensão reta 0,7123 horas, declinação 41,2688 graus. Não inicies as capturas.',
        'Voeg M31 aan het geladen plan toe: 300 opnamen van vijf minuten, zonder filterwisseling. Mijn gecontroleerde J2000-positie: rechte klimming 0,7123 uur, declinatie 41,2688 graden. Start geen opnamen.',
        'Dodaj M31 do wczytanego planu: 300 zdjęć po pięć minut, bez zmiany filtra. Moje sprawdzone współrzędne J2000: rektascensja 0,7123 godziny, deklinacja 41,2688 stopnia. Nie zaczynaj zdjęć.',
        '読み込み済みのプランに M31 を追加してください。5分露光を300枚、フィルター変更なしです。確認済みの J2000 座標は赤経0.7123時間、赤緯41.2688度です。撮影は開始しないでください。',
        '在已加载的计划中追加 M31：拍300张，每张曝光五分钟，不更换滤镜。使用我确认的 J2000 坐标：赤经0.7123小时，赤纬41.2688度。不要开始拍摄。',
    ),
    'clear_targets': (
        'Remove all targets from the loaded plan. Keep its start and end sections.',
        'Entferne alle Ziele aus der geladenen Sequenz. Behalte den Start- und Endbereich.',
        'Supprime toutes les cibles du plan chargé. Garde ses sections de début et de fin.',
        'Elimina todos los objetivos del plan cargado. Conserva las secciones de inicio y final.',
        'Rimuovi tutti gli obiettivi dal piano caricato. Mantieni le sezioni iniziale e finale.',
        'Remove todos os alvos do plano carregado. Mantém as secções inicial e final.',
        'Verwijder alle doelen uit het geladen plan. Behoud het begin- en eindgedeelte.',
        'Usuń wszystkie cele z wczytanego planu. Zachowaj sekcje początkową i końcową.',
        '読み込まれているプランから全対象を削除してください。開始部分と終了部分は残してください。',
        '删除已加载计划中的所有目标，保留开头和结尾部分。',
    ),
    'save_sequence': (
        'Save the loaded plan as Plan B. Use a new file, without replacing anything.',
        'Speichere die geladene Sequenz als Plan B. Neue Datei anlegen, nichts überschreiben.',
        'Enregistre le plan chargé sous le nom Plan B dans un nouveau fichier, sans rien écraser.',
        'Guarda el plan cargado como Plan B en un archivo nuevo, sin reemplazar nada.',
        'Salva il piano caricato come Plan B in un nuovo file, senza sovrascrivere nulla.',
        'Guarda o plano carregado como Plan B num ficheiro novo, sem substituir nada.',
        'Sla het geladen plan op als Plan B in een nieuw bestand, zonder iets te overschrijven.',
        'Zapisz wczytany plan jako Plan B w nowym pliku, niczego nie nadpisując.',
        '読み込み済みのプランを Plan B という新しいファイルに保存してください。上書きはしないでください。',
        '把当前加载的计划另存为 Plan B，新建文件，不要覆盖已有文件。',
    ),
    'start_sequence': (
        'Start the loaded imaging plan now, keeping all validation checks enabled.',
        'Starte jetzt die geladene Aufnahmesequenz mit allen Validierungsprüfungen.',
        'Démarre maintenant le plan chargé en conservant toutes les vérifications.',
        'Inicia ahora el plan de captura cargado manteniendo todas las validaciones.',
        'Avvia ora il piano caricato mantenendo tutti i controlli di validazione.',
        'Inicia agora o plano carregado mantendo todas as verificações de validação.',
        'Start nu het geladen opnameplan met alle controles ingeschakeld.',
        'Uruchom teraz wczytany plan zdjęć z włączonymi wszystkimi kontrolami poprawności.',
        '検証をすべて有効にしたまま、読み込み済みの撮影プランを開始してください。',
        '现在启动已加载的拍摄计划，保留所有验证检查。',
    ),
    'reset_sequence': (
        'Reset the loaded plan’s progress to zero so it can run from the beginning. Do not start it yet.',
        'Setze den Fortschritt der geladenen Sequenz zurück, damit sie von vorn laufen kann. Noch nicht starten.',
        'Remets à zéro la progression du plan chargé pour repartir du début. Ne le démarre pas encore.',
        'Reinicia el progreso del plan cargado para ejecutarlo desde el principio. No lo inicies todavía.',
        'Azzera l’avanzamento del piano caricato per ripartire dall’inizio. Non avviarlo ancora.',
        'Repõe o progresso do plano carregado a zero para voltar ao início. Ainda não o inicies.',
        'Zet de voortgang van het geladen plan op nul om opnieuw te beginnen. Start het nog niet.',
        'Wyzeruj postęp wczytanego planu, aby zaczął od początku. Jeszcze go nie uruchamiaj.',
        '読み込み済みプランの進捗をリセットして、最初から実行できるようにしてください。まだ開始しないでください。',
        '将已加载计划的进度清零，以便从头执行，但暂时不要启动。',
    ),
}


ARGUMENTS = {
    'get_status': {}, 'get_recent_images': {'count': 8},
    'get_sequence': {'max_depth': 5}, 'list_saved_sequences': {},
    'list_target_templates': {},
    'get_job': {'job_id': 'job-chat-42', 'wait_seconds': 5}, 'list_jobs': {},
    'stop_all': {}, 'cancel_job': {'job_id': 'job-chat-42', 'wait_seconds': 5},
    'stop_sequence': {}, 'stop_guiding': {},
    'connect_device': {'device': 'camera'}, 'disconnect_device': {'device': 'mount'},
    'slew_mount': {'ra': 7.25, 'dec': -18.5},
    'park_mount': {}, 'unpark_mount': {}, 'home_mount': {},
    'set_tracking': {'enabled': True},
    'capture_image': {'exposure_seconds': 90, 'save': False},
    'cool_camera': {'target_celsius': -12, 'duration_minutes': 8},
    'warm_camera': {'duration_minutes': 15},
    'move_focuser': {'position': 18420}, 'run_autofocus': {},
    'change_filter': {'name': 'Ha'}, 'start_guiding': {'force_calibration': False},
    'set_switch': {'index': 2, 'value': 40}, 'load_sequence': {'name': 'Plan A'},
    'add_target': {'name': 'M31', 'ra': 0.7123, 'dec': 41.2688,
                   'exposures': [{'exposure_seconds': 300, 'count': 300}]},
    'clear_targets': {}, 'save_sequence': {'name': 'Plan B'},
    'start_sequence': {}, 'reset_sequence': {},
}


# Competing actions teach distinctions: inspection versus changes, focusing
# versus moving, parking versus homing, sequence versus test exposure.
COMPETITORS = {
    'get_status': ['get_sequence', 'list_jobs'],
    'get_recent_images': ['capture_image', 'get_status'],
    'get_sequence': ['list_saved_sequences', 'get_status'],
    'list_saved_sequences': ['list_target_templates', 'get_sequence'],
    'list_target_templates': ['list_saved_sequences', 'get_sequence'],
    'get_job': ['list_jobs', 'get_status'],
    'list_jobs': ['get_job', 'get_status'],
    'stop_all': ['stop_sequence', 'get_status'],
    'cancel_job': ['get_job', 'list_jobs'],
    'stop_sequence': ['stop_all', 'get_sequence'],
    'stop_guiding': ['set_tracking', 'get_status'],
    'connect_device': ['disconnect_device', 'get_status'],
    'disconnect_device': ['connect_device', 'get_status'],
    'slew_mount': ['park_mount', 'get_status'],
    'park_mount': ['home_mount', 'get_status'],
    'unpark_mount': ['park_mount', 'get_status'],
    'home_mount': ['park_mount', 'get_status'],
    'set_tracking': ['start_guiding', 'get_status'],
    'capture_image': ['get_recent_images', 'get_status'],
    'cool_camera': ['warm_camera', 'get_status'],
    'warm_camera': ['cool_camera', 'get_status'],
    'move_focuser': ['run_autofocus', 'get_status'],
    'run_autofocus': ['move_focuser', 'get_status'],
    'change_filter': ['get_status', 'move_focuser'],
    'start_guiding': ['set_tracking', 'get_status'],
    'set_switch': ['get_status', 'connect_device'],
    'load_sequence': ['list_saved_sequences', 'get_sequence'],
    'add_target': ['get_sequence'],
    'clear_targets': ['reset_sequence', 'get_sequence'],
    'save_sequence': ['load_sequence', 'list_saved_sequences'],
    'start_sequence': ['get_sequence', 'stop_sequence'],
    'reset_sequence': ['clear_targets', 'get_sequence'],
}


IDLE = ('Synthetic fixture: this session already checked get_status. Sequence is idle; '
        'no hardware jobs run; equipment prerequisites have been checked. ')

CONTEXT = {
    'stop_all': 'Synthetic fixture: unexpected mount movement was reported. No completion has been confirmed.',
    'cancel_job': 'Synthetic fixture: job-chat-42 is running. Cancellation has not been requested.',
    'stop_sequence': 'Synthetic fixture: the loaded sequence is running. No stop has been confirmed.',
    'stop_guiding': IDLE + 'Guider is connected and guiding; mount is tracking.',
    'connect_device': IDLE + 'Profile camera slot is configured but disconnected.',
    'disconnect_device': IDLE + 'Mount is connected and parked.',
    'slew_mount': IDLE + 'Mount is connected and unparked; guider is stopped.',
    'park_mount': IDLE + 'Mount is connected and unparked; guider is stopped.',
    'unpark_mount': IDLE + 'Mount is connected and parked.',
    'home_mount': IDLE + 'Mount is connected, unparked and supports homing; guider is stopped.',
    'set_tracking': IDLE + 'Mount is connected and unparked; tracking is off; guider is stopped.',
    'capture_image': IDLE + 'Camera is connected; mount position is stable.',
    'cool_camera': IDLE + 'Camera is connected and supports temperature control.',
    'warm_camera': IDLE + 'Camera is connected with cooler active.',
    'move_focuser': IDLE + 'Focuser is connected; requested position is within its verified travel range.',
    'run_autofocus': IDLE + 'Camera and focuser are connected; autofocus is configured.',
    'change_filter': IDLE + 'Filter wheel is connected and lists Ha, OIII and L filters.',
    'start_guiding': IDLE + 'Guider is connected and stopped; mount is unparked and tracking.',
    'set_switch': IDLE + 'Connected switch reports index 2 writable, min 0, max 100, step 5.',
    'load_sequence': IDLE + 'Saved sequence list contains Plan A; current edits are saved.',
    'add_target': IDLE + 'An advanced sequence is loaded; append changes only its target area.',
    'clear_targets': IDLE + 'An advanced sequence is loaded with saved targets.',
    'save_sequence': IDLE + 'An advanced sequence is loaded; Plan B is not an existing saved name.',
    'start_sequence': IDLE + 'Loaded sequence was inspected; devices and configured filters are ready.',
    'reset_sequence': IDLE + 'An advanced sequence is loaded with finished and unfinished steps.',
}


def _tool_order(name, offset):
    names = [name, *COMPETITORS[name]]
    start = offset % len(names)
    return names[start:] + names[:start]


def build_rows():
    """Return new training rows without reading any existing conversation split."""
    catalog = definitions()
    if set(PROMPTS) != set(catalog) or set(ARGUMENTS) != set(catalog):
        raise ValueError('Natural-chat translations must cover every current MCP tool')
    rows = []
    for index, name in enumerate(catalog):
        if len(PROMPTS[name]) != len(LOCALES):
            raise ValueError(f'Incomplete natural-chat translations: {name}')
        for language_index, locale in enumerate(LOCALES):
            rows.append(conversation(
                f'chat-intent-{name}-plain', locale, 'natural_chat_intent',
                _tool_order(name, index + language_index),
                [user(PROMPTS[name][language_index]), tool_call(name, ARGUMENTS[name])],
                context=CONTEXT.get(name, ''),
            ))
    rows.extend(_rich_rows())
    return rows


def _rich_rows():
    rows = []

    def emit(case, name, args, en, de, *, context=None, history=None, category='chat_arguments'):
        for i, (locale, prompt) in enumerate((('en', en), ('de', de))):
            messages = list((history or {}).get(locale, []))
            messages.extend([user(prompt), tool_call(name, args)])
            rows.append(conversation(
                'chat-intent-' + case, locale, category, _tool_order(name, len(rows) + i),
                messages, context=CONTEXT.get(name, '') if context is None else context,
            ))

    variants = [
        ('status-colloquial', 'get_status', {},
         'Hey, how are things looking? Is the scope tracking and is the camera cooled?',
         'Na, wie siehts aus? Führt die Montierung nach und ist die Kamera gekühlt?'),
        ('images-quality', 'get_recent_images', {'count': 3},
         'How sharp were my last three exposures? Get their measured image stats.',
         'Wie scharf waren meine letzten drei Aufnahmen? Hol die gemessenen Bildwerte.'),
        ('sequence-progress', 'get_sequence', {},
         'Which target and exposure block are we on in the current plan?',
         'Bei welchem Ziel und Aufnahmeblock sind wir gerade in der aktuellen Sequenz?'),
        ('saved-plan-picker', 'list_saved_sequences', {},
         'I want to choose a saved session plan. What choices do I have?',
         'Ich möchte eine gespeicherte Aufnahmesequenz auswählen. Welche gibt es?'),
        ('template-picker', 'list_target_templates', {},
         'Before I add a target, show the reusable target setups I saved.',
         'Zeig vor dem Hinzufügen eines Ziels meine gespeicherten Zielvorlagen.'),
        ('job-immediate', 'get_job', {'job_id': 'job-focus-17', 'wait_seconds': 0},
         'Check job-focus-17 once right now, without waiting.',
         'Prüfe job-focus-17 einmal sofort, ohne zu warten.'),
        ('find-busy-resource', 'list_jobs', {},
         'Which background tasks are still holding equipment? I do not have their numbers.',
         'Welche Hintergrundaufträge belegen noch Geräte? Ich habe ihre Nummern nicht.'),
        ('emergency-typo', 'stop_all', {},
         'oh no stop evrything immediatly!',
         'Mist, sofrot alles stopppen!'),
        ('cancel-one-now', 'cancel_job', {'job_id': 'job-slew-19', 'wait_seconds': 0},
         'Abort only job-slew-19. Return immediately, do not wait for it.',
         'Brich nur job-slew-19 ab. Sofort zurückmelden, nicht darauf warten.'),
        ('stop-plan-not-guiding', 'stop_sequence', {},
         'Pause this night’s imaging by stopping the running plan. Leave guiding alone.',
         'Unterbrich die Aufnahmen dieser Nacht, indem du die laufende Sequenz stoppst. Guiding unverändert lassen.'),
        ('stop-guide-not-tracking', 'stop_guiding', {},
         'Switch off the guide corrections, but keep sidereal tracking on.',
         'Schalte die Guiding-Korrekturen aus, aber lass die siderische Nachführung an.'),
        ('camera-connect-typo', 'connect_device', {'device': 'camera'},
         'Can you conect my imaging cam please?',
         'Kannst du bitte meine Aufnahmekamera verbidnen?'),
        ('mount-disconnect-alias', 'disconnect_device', {'device': 'mount'},
         'The telescope mount is safely parked. Drop its connection now.',
         'Die Teleskopmontierung ist sicher geparkt. Jetzt die Verbindung trennen.'),
        ('coordinate-degrees', 'slew_mount', {'ra': 7, 'dec': -12.25},
         'Go to my verified J2000 position: right ascension 105 degrees and declination minus 12.25 degrees.',
         'Fahre zu meinen verifizierten J2000-Koordinaten: Rektaszension 105 Grad und Deklination minus 12,25 Grad.'),
        ('park-shutdown', 'park_mount', {},
         'The session is done. Park the mount, keeping it connected.',
         'Die Sitzung ist vorbei. Parke die Montierung und lass sie verbunden.'),
        ('unpark-only', 'unpark_mount', {},
         'Unpark the mount. Do not slew to any target yet.',
         'Entparke die Montierung. Noch kein Ziel anfahren.'),
        ('home-reference', 'home_mount', {},
         'Run the mount’s home search to find its reference sensors.',
         'Starte die Referenzfahrt der Montierung, um ihre Referenzsensoren zu finden.'),
        ('tracking-off', 'set_tracking', {'enabled': False},
         'Turn mount tracking off. Do not park or disconnect it.',
         'Schalte die Nachführung aus. Nicht parken oder trennen.'),
        ('capture-milliseconds', 'capture_image', {'exposure_seconds': 0.5, 'save': False},
         'Take a single 500 millisecond test frame and discard the file.',
         'Mach ein einzelnes Testbild mit 500 Millisekunden Belichtung und speichere es nicht.'),
        ('cool-fahrenheit', 'cool_camera', {'target_celsius': -10, 'duration_minutes': 6},
         'Cool the sensor to 14 degrees Fahrenheit over six minutes.',
         'Kühle den Sensor in sechs Minuten auf 14 Grad Fahrenheit.'),
        ('warm-short-ramp', 'warm_camera', {'duration_minutes': 3},
         'Ramp the camera back to ambient in three minutes before we disconnect it. Only warm it for now.',
         'Fahre die Kamera vor dem Trennen in drei Minuten auf Umgebungstemperatur. Jetzt nur aufwärmen.'),
        ('focus-short-wording', 'move_focuser', {'position': 20350},
         'Set focus to the absolute step position 20350.',
         'Stell den Fokus auf die absolute Schrittposition 20350.'),
        ('autofocus-colloquial', 'run_autofocus', {},
         'Please autofocus again; the stars look bloated.',
         'Bitte nochmal automatisch fokussieren, die Sterne wirken aufgebläht.'),
        ('filter-second-slot', 'change_filter', {'position': 1},
         'Move to the second slot in the wheel, counting the first slot as zero.',
         'Fahre zur zweiten Position im Filterrad, wobei die erste Position den Index null hat.'),
        ('guiding-fresh-calibration', 'start_guiding', {'force_calibration': True},
         'Start guiding and do a fresh calibration this time.',
         'Starte das Guiding und kalibriere dieses Mal neu.'),
        ('switch-heater-off', 'set_switch', {'index': 2, 'value': 0},
         'Turn the dew heater at index 2 down to zero.',
         'Dreh die Taukappenheizung mit Index 2 auf null.'),
        ('load-do-not-start', 'load_sequence', {'name': 'Plan A'},
         'Open my saved Plan A for editing, replacing this saved current plan. Do not run it.',
         'Öffne meinen gespeicherten Plan A zum Bearbeiten und ersetze die aktuell gespeicherte Sequenz. Nicht ausführen.'),
        ('target-short-exposures', 'add_target',
         {'name': 'M31', 'ra': 0.7123, 'dec': 41.2688,
          'exposures': [{'exposure_seconds': 150, 'count': 24}]},
         'Add M31 at my verified J2000 coordinates, right ascension 0.7123 hours and declination 41.2688 degrees, to this plan: 24 shots of two and a half minutes, no filter change. Just prepare it.',
         'Ergänze M31 an meinen verifizierten J2000-Koordinaten, Rektaszension 0,7123 Stunden und Deklination 41,2688 Grad, in dieser Sequence: 24 Bilder zu zweieinhalb Minuten, ohne Filterwechsel. Nur vorbereiten.'),
        ('clear-not-reset', 'clear_targets', {},
         'Delete the targets themselves, not just their progress. Keep the loaded sequence root.',
         'Lösche die Ziele selbst, nicht nur ihren Fortschritt. Die geladene Sequenzwurzel behalten.'),
        ('save-new-name', 'save_sequence', {'name': 'Plan B'},
         'Keep this setup for another night under the new saved name Plan B.',
         'Bewahre diese Zusammenstellung für eine weitere Nacht unter dem neuen Namen Plan B auf.'),
        ('resume-sequence', 'start_sequence', {},
         'Continue the loaded sequence from its unfinished steps with the normal checks.',
         'Setze die geladene Sequenz bei ihren unfertigen Schritten mit den normalen Prüfungen fort.'),
        ('reset-not-delete', 'reset_sequence', {},
         'Make all exposure counts start from zero again, without deleting any targets or starting the plan.',
         'Lass alle Aufnahmezähler wieder bei null beginnen, ohne Ziele zu löschen oder die Sequenz zu starten.'),
    ]
    for case, name, args, en, de in variants:
        emit(case, name, args, en, de)

    # The caller supplies device meaning in natural language. The model must
    # map equipment aliases to the configured slot, not invent driver names.
    device_aliases = (
        ('mount', 'telescope mount', 'Teleskopmontierung'),
        ('focuser', 'motorized focuser', 'Motorfokussierer'),
        ('filter_wheel', 'filter wheel', 'Filterrad'),
        ('guider', 'autoguider', 'Autoguider'),
        ('rotator', 'camera rotator', 'Kamerarotator'),
        ('flat_device', 'flat panel', 'Flatpanel'),
        ('dome', 'observatory dome', 'Sternwartenkuppel'),
        ('switch', 'switch box', 'Schaltbox'),
        ('weather', 'weather station', 'Wetterstation'),
        ('safety_monitor', 'safety monitor', 'Sicherheitsmonitor'),
    )
    for device, en_name, de_name in device_aliases:
        emit('connect-slot-' + device, 'connect_device', {'device': device},
             f'Connect the {en_name} selected in this profile.',
             f'Verbinde das im Profil ausgewählte Gerät für {de_name}.',
             context=IDLE + f'Profile slot {device} is configured and disconnected.')

    emit('capture-driver-options', 'capture_image',
         {'exposure_seconds': 120, 'filter': 'OIII', 'gain': 120, 'offset': 20,
          'binning': 2, 'save': True},
         'Take and save one two-minute frame through OIII, gain 120, offset 20, binning two by two.',
         'Mach und speichere ein einzelnes Bild mit zwei Minuten durch OIII, Gain 120, Offset 20 und Binning zwei mal zwei.',
         context=IDLE + 'Camera and filter wheel are connected; OIII is a configured filter; gain, offset and binning are supported.')
    emit('coordinate-sexagesimal', 'slew_mount', {'ra': 2.25, 'dec': -0.5},
         'Point to my verified J2000 position: right ascension 2 hours 15 minutes, declination minus zero degrees 30 arcminutes.',
         'Fahre zu meiner verifizierten J2000-Position: Rektaszension 2 Stunden 15 Minuten, Deklination minus null Grad 30 Bogenminuten.')
    emit('focuser-relative-known-position', 'move_focuser', {'position': 18520},
         'Move the focuser 100 steps higher than its measured current position.',
         'Fahre den Fokussierer 100 Schritte höher als seine gemessene aktuelle Position.',
         context=IDLE + 'Connected focuser position is 18420 steps; valid travel is 0 to 50000 steps.')
    emit('heater-percent-known-range', 'set_switch', {'index': 2, 'value': 50},
         'Set the measured dew-heater channel to fifty percent.',
         'Stelle den gemessenen Taukappenheizungs-Kanal auf fünfzig Prozent.',
         context=IDLE + 'The sole dew-heater channel has index 2, writable true, min 0, max 100, step 5; its values are percent.')
    emit('target-template', 'add_target',
         {'name': 'M31', 'ra': 0.7123, 'dec': 41.2688, 'template': 'Template A'},
         'Append M31 using my saved Template A. Use my verified J2000 position: right ascension 0.7123 hours, declination 41.2688 degrees. Leave the template’s own instructions unchanged.',
         'Ergänze M31 mit meiner gespeicherten Vorlage Template A. Verwende meine verifizierte J2000-Position: Rektaszension 0,7123 Stunden, Deklination 41,2688 Grad. Die eigenen Anweisungen der Vorlage unverändert lassen.',
         context=CONTEXT['add_target'] + ' Saved target templates include Template A.')
    emit('target-options', 'add_target',
         {'name': 'M31', 'ra': 0.7123, 'dec': 41.2688,
          'exposures': [{'exposure_seconds': 180, 'count': 12, 'filter': 'Ha'},
                        {'exposure_seconds': 120, 'count': 8, 'filter': 'OIII'}],
          'slew_and_center': False, 'start_guiding': False,
          'autofocus_at_start': True, 'dither_every': 4, 'rotation_degrees': 45},
         'Append M31 at my verified J2000 position, right ascension 0.7123 hours, declination 41.2688 degrees. Plan twelve three-minute Ha frames then eight two-minute OIII frames, position angle 45 degrees, autofocus at the beginning, dither every four frames. No slewing or guiding start in this plan. Do not run it.',
         'Ergänze M31 an meiner verifizierten J2000-Position, Rektaszension 0,7123 Stunden, Deklination 41,2688 Grad. Plane zwölf Drei-Minuten-Bilder mit Ha, danach acht Zwei-Minuten-Bilder mit OIII, Positionswinkel 45 Grad, Autofokus am Anfang und Dithering alle vier Bilder. Kein Anfahren und kein Guiding-Start in diesem Plan. Nicht ausführen.',
         context=CONTEXT['add_target'] + ' Configured filters include Ha and OIII; a rotator is configured.')

    # Corrections happen before proposal. History contains ordinary assistant
    # clarification rather than fabricated executed hardware actions.
    emit('correct-exposure-before-proposal', 'capture_image', {'exposure_seconds': 30, 'save': False},
         'Actually make that thirty seconds, just one test frame, no saving.',
         'Doch lieber dreißig Sekunden, nur ein Testbild, nicht speichern.',
         category='chat_correction', history={
             'en': [user('I might want a five-minute exposure.'), assistant('Do you want a single test frame, and should it be saved?')],
             'de': [user('Vielleicht möchte ich eine Fünf-Minuten-Aufnahme.'), assistant('Möchtest du ein einzelnes Testbild, und soll es gespeichert werden?')],
         })
    emit('correct-coordinate-sign', 'slew_mount', {'ra': 7.25, 'dec': -18.5},
         'Correction: declination is minus 18.5 degrees. The verified J2000 right ascension stays 7.25 hours. Go there.',
         'Korrektur: Deklination minus 18,5 Grad. Die verifizierte J2000-Rektaszension bleibt 7,25 Stunden. Fahre dorthin.',
         category='chat_correction', history={
             'en': [user('My J2000 right ascension is 7.25 hours; I am unsure of the declination sign.'), assistant('Please verify the declination and its sign before I prepare a movement.')],
             'de': [user('Meine J2000-Rektaszension ist 7,25 Stunden; beim Vorzeichen der Deklination bin ich unsicher.'), assistant('Bitte bestätige die Deklination samt Vorzeichen, bevor ich eine Bewegung vorbereite.')],
         })
    emit('correct-filter-before-proposal', 'change_filter', {'name': 'OIII'},
         'Use OIII instead, not Ha. Change to OIII now.',
         'Nimm stattdessen OIII, nicht Ha. Jetzt zu OIII wechseln.',
         category='chat_correction', history={
             'en': [user('I am choosing between Ha and OIII.'), assistant('Which configured filter do you want to select?')],
             'de': [user('Ich entscheide noch zwischen Ha und OIII.'), assistant('Welchen konfigurierten Filter möchtest du auswählen?')],
         })
    emit('recover-task-id', 'get_job', {'job_id': 'job-exposure-8'},
         'How did the exposure I just asked about turn out?',
         'Wie ist die Aufnahme ausgegangen, nach der ich gerade gefragt habe?',
         context='', category='chat_context_reference', history={
             'en': [user('The app displayed job-exposure-8 as the number of my test exposure.'), assistant('I can check that background task’s result.')],
             'de': [user('Die App hat job-exposure-8 als Nummer meiner Testaufnahme angezeigt.'), assistant('Ich kann das Ergebnis dieses Hintergrundauftrags prüfen.')],
         })
    rows.extend(_negative_rows())
    return rows


def _negative_rows():
    rows = []

    def emit(case, names, en, de, en_reply, de_reply, *, context='', history=None,
             category='chat_clarification'):
        for locale, prompt, reply in (('en', en, en_reply), ('de', de, de_reply)):
            messages = list((history or {}).get(locale, []))
            messages.extend([user(prompt), assistant(reply)])
            rows.append(conversation('chat-intent-' + case, locale, category, names,
                                     messages, context=context))

    emit('ambiguous-start', ['start_sequence', 'start_guiding', 'get_sequence'],
         'Start it.', 'Starte das.',
         'Do you mean the loaded imaging sequence or autoguiding?',
         'Meinst du die geladene Aufnahmesequenz oder das Autoguiding?')
    emit('ambiguous-stop', ['stop_all', 'stop_sequence', 'stop_guiding'],
         'How would I stop it?', 'Wie würde ich das stoppen?',
         'Do you mean stopping the imaging sequence, guiding, or all operations? I have not requested a stop.',
         'Meinst du die Aufnahmesequenz, das Guiding oder alle Vorgänge? Ich habe keinen Stopp angefordert.',
         category='chat_no_action_question')
    emit('missing-device', ['connect_device', 'disconnect_device', 'get_status'],
         'Connect my device.', 'Verbinde mein Gerät.',
         'Which device slot do you want to connect, for example the camera or mount?',
         'Welchen Geräteslot möchtest du verbinden, zum Beispiel Kamera oder Montierung?')
    emit('missing-job-id', ['cancel_job', 'get_job', 'list_jobs'],
         'Cancel that job. I have two running tasks and do not remember which one I meant.',
         'Brich den Auftrag ab. Ich habe zwei laufende Aufträge und weiß nicht mehr, welchen ich meinte.',
         'Which task should be cancelled? Please identify it; I will not guess between the two.',
         'Welcher Auftrag soll abgebrochen werden? Bitte identifiziere ihn; ich werde nicht zwischen beiden raten.')
    emit('named-target-no-coordinate-resolver', ['slew_mount', 'get_status'],
         'Slew to M31. I have not supplied coordinates.',
         'Fahre zu M31. Ich habe keine Koordinaten angegeben.',
         'Please provide verified J2000 right ascension in hours and declination in degrees. These tools cannot resolve a target name to coordinates.',
         'Bitte gib verifizierte J2000-Koordinaten an: Rektaszension in Stunden und Deklination in Grad. Diese Werkzeuge können aus einem Zielnamen keine Koordinaten bestimmen.')
    emit('target-plan-missing-position', ['add_target', 'get_sequence'],
         'Create a sequence for M31, 300 five-minute exposures without a filter change.',
         'Erstelle eine Sequence für M31, 300 Fünf-Minuten-Aufnahmen ohne Filterwechsel.',
         'I can append that target plan to a loaded sequence. What are your verified J2000 right ascension in hours and declination in degrees? Appending does not start imaging.',
         'Ich kann diesen Zielplan in einer geladenen Sequenz ergänzen. Welche verifizierten J2000-Koordinaten soll ich verwenden: Rektaszension in Stunden und Deklination in Grad? Das Ergänzen startet keine Aufnahme.',
         context='Synthetic fixture: sequence is idle and an advanced sequence is loaded.')
    emit('coordinate-unit-ambiguous', ['slew_mount', 'get_status'],
         'Go to right ascension 12, declination 20, J2000. I am not sure whether 12 is hours or degrees.',
         'Fahre zu Rektaszension 12, Deklination 20, J2000. Ich weiß nicht, ob 12 Stunden oder Grad sind.',
         'Please verify the right-ascension unit first. Twelve hours and twelve degrees describe different positions.',
         'Bitte prüfe zuerst die Einheit der Rektaszension. Zwölf Stunden und zwölf Grad sind verschiedene Positionen.')
    emit('coordinates-outside-range', ['slew_mount', 'get_status'],
         'Point to J2000 right ascension 25 hours, declination 95 degrees.',
         'Fahre zu J2000-Rektaszension 25 Stunden und Deklination 95 Grad.',
         'Those coordinates are out of range. Please correct them: right ascension must be from 0 up to but not including 24 hours, and declination from minus 90 to plus 90 degrees.',
         'Diese Koordinaten liegen außerhalb des gültigen Bereichs. Bitte korrigiere sie: Rektaszension von 0 bis unter 24 Stunden, Deklination von minus 90 bis plus 90 Grad.')
    emit('missing-exposure-time', ['capture_image', 'get_recent_images'],
         'Take a test frame.', 'Mach ein Testbild.',
         'How many seconds should the test exposure last?',
         'Wie viele Sekunden soll das Testbild belichtet werden?')
    emit('exposure-too-long', ['capture_image', 'get_status'],
         'Take one two-hour exposure.', 'Mach eine einzelne Zwei-Stunden-Aufnahme.',
         'This tool allows at most one hour per exposure. What supported exposure length should I use?',
         'Dieses Werkzeug erlaubt höchstens eine Stunde pro Aufnahme. Welche unterstützte Belichtungsdauer soll ich verwenden?')
    emit('exposure-zero', ['capture_image', 'get_status'],
         'Take a zero-second picture.', 'Mach eine Aufnahme mit null Sekunden.',
         'An exposure must be positive and at least 0.0001 seconds. What exposure time do you want?',
         'Eine Aufnahme muss positiv und mindestens 0,0001 Sekunden lang sein. Welche Belichtungszeit möchtest du?')
    emit('temperature-too-cold', ['cool_camera', 'warm_camera', 'get_status'],
         'Cool the sensor to minus 80 Celsius.', 'Kühle den Sensor auf minus 80 Grad Celsius.',
         'That is outside the tool’s supported temperature range of minus 50 to plus 30 Celsius. Which supported target should I prepare?',
         'Das liegt außerhalb des unterstützten Bereichs von minus 50 bis plus 30 Grad Celsius. Welchen unterstützten Zielwert soll ich vorbereiten?')
    emit('temperature-missing-value', ['cool_camera', 'warm_camera', 'get_status'],
         'Make the camera colder.', 'Mach die Kamera kälter.',
         'What target sensor temperature in Celsius should I use?',
         'Welche Zieltemperatur des Sensors in Grad Celsius soll ich verwenden?')
    emit('focus-missing-position', ['move_focuser', 'run_autofocus', 'get_status'],
         'Move the focuser somewhere better.', 'Fahre den Fokussierer an eine bessere Stelle.',
         'Do you want the autofocus routine, or a particular absolute focuser position in steps?',
         'Möchtest du einen Autofokuslauf oder eine bestimmte absolute Fokussiererposition in Schritten?')
    emit('filter-unknown-name', ['change_filter', 'get_status'],
         'Switch to the green filter.', 'Wechsle zum grünen Filter.',
         'The configured filters are Ha, OIII and L, with no filter named green. Which configured name do you mean?',
         'Die konfigurierten Filter heißen Ha, OIII und L; ein Filter namens grün fehlt. Welchen konfigurierten Namen meinst du?',
         context='Synthetic checked status: configured filter names are Ha, OIII and L.')
    emit('filter-name-and-index-conflict', ['change_filter', 'get_status'],
         'Choose Ha, but also use the OIII slot instead.',
         'Wähle Ha, aber verwende stattdessen auch die OIII-Position.',
         'Which filter do you want, Ha or OIII? I can propose only one selected filter.',
         'Welchen Filter möchtest du, Ha oder OIII? Ich kann nur einen ausgewählten Filter vorschlagen.')
    emit('switch-exceeds-range', ['set_switch', 'get_status'],
         'Put the dew heater at index 2 on 130.', 'Stelle die Taukappenheizung mit Index 2 auf 130.',
         'That channel only permits values from 0 to 100 in steps of 5. What supported value do you want?',
         'Dieser Kanal erlaubt nur Werte von 0 bis 100 in Fünferschritten. Welchen unterstützten Wert möchtest du?',
         context='Synthetic checked switch: index 2 writable true, min 0, max 100, step 5.')
    emit('switch-readonly-channel', ['set_switch', 'get_status'],
         'Set the voltage sensor channel at index 3 to 12.',
         'Setze den Spannungssensorkanal mit Index 3 auf 12.',
         'That channel is read-only, so I cannot set its value.',
         'Dieser Kanal ist schreibgeschützt; ich kann seinen Wert nicht ändern.',
         context='Synthetic checked switch: channel index 3 is a read-only voltage sensor.',
         category='chat_unsupported')
    emit('saved-name-missing', ['load_sequence', 'list_saved_sequences'],
         'Load one of my saved plans.', 'Lade eine meiner gespeicherten Sequenzen.',
         'Which saved plan should I load? Please give its name.',
         'Welche gespeicherte Sequenz soll ich laden? Bitte nenne ihren Namen.')
    emit('new-root-not-supported', ['load_sequence', 'clear_targets', 'get_sequence'],
         'Create a completely new blank sequencer root, leaving the old root in memory.',
         'Erstelle eine ganz neue leere Sequenzwurzel und behalte die alte im Speicher.',
         'These tools cannot create another blank sequence root. I can inspect the current sequence or help load a saved one.',
         'Diese Werkzeuge können keine weitere leere Sequenzwurzel erstellen. Ich kann die aktuelle Sequenz prüfen oder beim Laden einer gespeicherten helfen.',
         category='chat_unsupported')
    emit('rename-single-target-not-supported', ['get_sequence', 'add_target', 'clear_targets'],
         'Rename only the second target to M31 and leave every other target untouched.',
         'Benenne nur das zweite Ziel in M31 um und lasse alle anderen Ziele unverändert.',
         'These tools cannot rename an individual existing target. Please edit that target in the sequencer; clearing all targets would remove the others too.',
         'Diese Werkzeuge können ein einzelnes vorhandenes Ziel nicht umbenennen. Bitte bearbeite es im Sequencer; alle Ziele zu löschen würde auch die anderen entfernen.',
         category='chat_unsupported')
    emit('save-path-not-allowed', ['save_sequence', 'list_saved_sequences'],
         'Save this plan as ../../my-plan.', 'Speichere diese Sequenz als ../../mein-plan.',
         'Please choose a plain file name without path separators; the sequence is saved inside the profile folder.',
         'Bitte wähle einen einfachen Dateinamen ohne Pfadtrennzeichen; gespeichert wird im Profilordner.')
    emit('save-reserved-name', ['save_sequence', 'list_saved_sequences'],
         'Save it as Night.template.', 'Speichere sie als Nacht.template.',
         'Names ending in .template are reserved. Please choose a different sequence name.',
         'Namen mit der Endung .template sind reserviert. Bitte wähle einen anderen Sequenznamen.')
    emit('fractional-frame-count', ['add_target', 'get_sequence'],
         'Append M31 at verified J2000 right ascension 0.7123 hours and declination 41.2688 degrees with 2.5 five-minute frames.',
         'Ergänze M31 an verifizierter J2000-Rektaszension 0,7123 Stunden und Deklination 41,2688 Grad mit 2,5 Fünf-Minuten-Bildern.',
         'The frame count must be a whole number. How many five-minute frames should I plan?',
         'Die Bildanzahl muss eine ganze Zahl sein. Wie viele Fünf-Minuten-Aufnahmen soll ich planen?')
    emit('explain-not-execute-focus', ['run_autofocus', 'get_recent_images'],
         'Explain what autofocus does. Do not run it.',
         'Erklär mir, was Autofokus macht. Nicht ausführen.',
         'Autofocus measures star sharpness at several focuser positions and estimates the best focus position. No focus run has been requested.',
         'Autofokus misst die Sternschärfe an mehreren Fokussiererpositionen und ermittelt die beste Fokusposition. Es wurde kein Fokussierlauf angefordert.',
         category='chat_no_action_question')
    emit('hypothetical-park', ['park_mount', 'get_status'],
         'Would parking stop the mount tracking? This is only a question.',
         'Würde Parken die Nachführung stoppen? Das ist nur eine Frage.',
         'Yes, parking normally stops tracking. I have not proposed parking the mount.',
         'Ja, Parken stoppt normalerweise die Nachführung. Ich habe kein Parken der Montierung vorgeschlagen.',
         category='chat_no_action_question')
    emit('deny-camera-connect', ['connect_device', 'get_status'],
         'Do not connect the camera.', 'Verbinde die Kamera nicht.',
         'Understood. I will not propose connecting the camera.',
         'Verstanden. Ich werde keine Verbindung der Kamera vorschlagen.',
         category='chat_denial')
    emit('withdraw-capture', ['capture_image', 'get_status'],
         'Never mind. Do not take any picture.', 'Doch nicht. Keine Aufnahme machen.',
         'The capture request is withdrawn. I will not propose an exposure.',
         'Die Aufnahmeanforderung ist zurückgezogen. Ich werde keine Belichtung vorschlagen.',
         history={
             'en': [user('I might take a ten-second test frame.'), assistant('Should that be saved, or only analyzed?')],
             'de': [user('Vielleicht möchte ich ein Zehn-Sekunden-Testbild.'), assistant('Soll es gespeichert oder nur ausgewertet werden?')],
         }, category='chat_denial')
    emit('withdraw-sequence-start', ['start_sequence', 'get_sequence'],
         'Cancel my request to start it; keep the sequence idle.',
         'Zieh meine Startanforderung zurück; die Sequenz soll im Leerlauf bleiben.',
         'I will not propose starting the sequence.',
         'Ich werde keinen Start der Sequenz vorschlagen.',
         context='Synthetic fixture: sequence is idle; no start action has been executed.',
         category='chat_denial')
    emit('no-equipment-access', [],
         'Is my mount connected right now?', 'Ist meine Montierung gerade verbunden?',
         'I have no equipment tools in this chat, so I cannot check its connection state.',
         'In diesem Chat habe ich keine Gerätewerkzeuge und kann den Verbindungszustand nicht prüfen.',
         category='chat_no_tools')

    # Basic informational and denied-action contrasts are also multilingual,
    # preventing ordinary discussion from being learned as an equipment command.
    question = (
        'What does HFR mean in an image? Do not change anything.',
        'Was bedeutet HFR in einem Bild? Nichts verändern.',
        'Que signifie HFR sur une image ? Ne change rien.',
        '¿Qué significa HFR en una imagen? No cambies nada.',
        'Che cosa significa HFR in un’immagine? Non cambiare nulla.',
        'O que significa HFR numa imagem? Não alteres nada.',
        'Wat betekent HFR in een beeld? Verander niets.',
        'Co oznacza HFR na zdjęciu? Niczego nie zmieniaj.',
        '画像の HFR とは何ですか？何も変更しないでください。',
        '照片中的 HFR 是什么意思？不要修改任何设置。',
    )
    explanation = (
        'HFR is the radius containing half a star’s measured light. Lower values generally indicate sharper stars when conditions are comparable.',
        'HFR ist der Radius, der die Hälfte des gemessenen Sternlichts enthält. Unter vergleichbaren Bedingungen deuten kleinere Werte meist auf schärfere Sterne hin.',
        'HFR est le rayon contenant la moitié de la lumière mesurée d’une étoile. Dans des conditions comparables, une valeur plus faible indique généralement des étoiles plus nettes.',
        'HFR es el radio que contiene la mitad de la luz medida de una estrella. En condiciones comparables, valores menores suelen indicar estrellas más nítidas.',
        'HFR è il raggio che contiene metà della luce misurata di una stella. A condizioni comparabili, valori più bassi indicano generalmente stelle più nitide.',
        'HFR é o raio que contém metade da luz medida de uma estrela. Em condições comparáveis, valores menores geralmente indicam estrelas mais nítidas.',
        'HFR is de straal die de helft van het gemeten sterlicht bevat. Bij vergelijkbare omstandigheden wijzen lagere waarden doorgaans op scherpere sterren.',
        'HFR to promień obejmujący połowę zmierzonego światła gwiazdy. W porównywalnych warunkach niższe wartości zwykle oznaczają ostrzejsze gwiazdy.',
        'HFR は、測定した星の光の半分を含む半径です。同じような条件では、値が小さいほど星が鮮明な傾向があります。',
        'HFR 是包含一颗星一半测量光量的半径。在条件相近时，数值越小通常表示星点越清晰。',
    )
    denied = (
        'Do not disconnect the mount.', 'Trenne die Montierung nicht.',
        'Ne déconnecte pas la monture.', 'No desconectes la montura.',
        'Non scollegare la montatura.', 'Não desligues a montagem.',
        'Verbreek de verbinding met de montering niet.', 'Nie rozłączaj montażu.',
        '架台の接続を切らないでください。', '不要断开赤道仪的连接。',
    )
    acknowledged = (
        'Understood. I will keep the mount connection unchanged.',
        'Verstanden. Die Verbindung der Montierung bleibt unverändert.',
        'Compris. Je laisse la connexion de la monture inchangée.',
        'Entendido. Mantendré la conexión de la montura sin cambios.',
        'Capito. Lascio invariata la connessione della montatura.',
        'Entendido. Vou manter a ligação da montagem sem alterações.',
        'Begrepen. Ik laat de verbinding met de montering ongewijzigd.',
        'Rozumiem. Pozostawię połączenie montażu bez zmian.',
        '承知しました。架台の接続はそのままにします。',
        '明白，我会保持赤道仪的连接不变。',
    )
    for i, locale in enumerate(LOCALES):
        rows.append(conversation('chat-intent-explain-hfr', locale, 'chat_no_action_question',
                                 ['get_recent_images', 'run_autofocus'],
                                 [user(question[i]), assistant(explanation[i])]))
        rows.append(conversation('chat-intent-deny-mount-disconnect', locale, 'chat_denial',
                                 ['disconnect_device', 'get_status'],
                                 [user(denied[i]), assistant(acknowledged[i])]))
    return rows
