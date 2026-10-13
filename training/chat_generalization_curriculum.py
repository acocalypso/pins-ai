"""Contract-authored train-only parameter variation, not evaluation answer repair.

Values are synthetic and independently chosen. Native language review is pending.
No existing split, prediction or held-out answer is loaded by this module.
"""
from chat_dataset_common import LANGUAGES, assistant, conversation, tool_call, tool_result, user
from chat_intent_curriculum import COMPETITORS, CONTEXT, PROMPTS
from chat_curriculum import DENIAL, DENIED_ACK
from chat_workflow_curriculum import _job
from v2_contracts import definitions

# Columns follow LANGUAGES. Values remain grounded in user text, never a system
# authorization hint. Fractional minutes and degrees require explicit conversion.
TEXT = {
 'capture_seconds': (
    'Take a {seconds} second test exposure, without saving it.',
    'Mach eine Testaufnahme mit {seconds} Sekunden, ohne sie zu speichern.',
    'Prends une pose de test de {seconds} secondes sans l’enregistrer.',
    'Haz una exposición de prueba de {seconds} segundos sin guardarla.',
    'Scatta una posa di prova di {seconds} secondi senza salvarla.',
    'Faz uma exposição de teste de {seconds} segundos sem a guardar.',
    'Maak een testopname van {seconds} seconden zonder die op te slaan.',
    'Zrób próbną ekspozycję przez {seconds} sekund bez zapisywania.',
    '{seconds}秒のテスト露光を撮って、保存しないでください。',
    '拍一张{seconds}秒的测试曝光，不要保存。'),
 'capture_minutes': (
    'Expose for {minutes} minutes and save this frame.',
    'Belichte {minutes} Minuten und speichere dieses Bild.',
    'Expose pendant {minutes} minutes et enregistre cette image.',
    'Expón durante {minutes} minutos y guarda esta imagen.',
    'Esponi per {minutes} minuti e salva questa immagine.',
    'Expõe durante {minutes} minutos e guarda esta imagem.',
    'Belicht {minutes} minuten en sla deze opname op.',
    'Naświetlaj przez {minutes} minut i zapisz tę klatkę.',
    '{minutes}分間露光して、この画像を保存してください。',
    '曝光{minutes}分钟并保存这张图像。'),
 'focus': (
    'Move the focuser to absolute position {position} steps.',
    'Fahre den Fokussierer auf die absolute Position {position} Schritte.',
    'Déplace le moteur de mise au point à la position absolue {position} pas.',
    'Mueve el enfocador a la posición absoluta de {position} pasos.',
    'Sposta il focheggiatore alla posizione assoluta di {position} passi.',
    'Move o focador para a posição absoluta de {position} passos.',
    'Zet de focuser op de absolute positie {position} stappen.',
    'Przesuń wyciąg do bezwzględnej pozycji {position} kroków.',
    'フォーカサーを絶対位置{position}ステップに移動してください。',
    '把调焦器移动到绝对位置{position}步。'),
 'cool': (
    'Cool the camera to {temperature} °C over {duration} minutes.',
    'Kühle die Kamera in {duration} Minuten auf {temperature} °C.',
    'Refroidis la caméra à {temperature} °C en {duration} minutes.',
    'Enfría la cámara a {temperature} °C durante {duration} minutos.',
    'Raffredda la camera a {temperature} °C in {duration} minuti.',
    'Arrefece a câmara até {temperature} °C em {duration} minutos.',
    'Koel de camera in {duration} minuten tot {temperature} °C.',
    'Schłodź kamerę do {temperature} °C w ciągu {duration} minut.',
    '{duration}分かけてカメラを{temperature} °Cまで冷却してください。',
    '用{duration}分钟把相机冷却到{temperature} °C。'),
 'switch': (
    'Set switch index {index}, numbered from zero, to exactly {value}.',
    'Setze Schalterindex {index}, ab null gezählt, auf genau {value}.',
    'Règle le canal d’indice {index}, numéroté depuis zéro, à exactement {value}.',
    'Pon el canal de índice {index}, numerado desde cero, exactamente en {value}.',
    'Imposta il canale con indice {index}, contando da zero, esattamente a {value}.',
    'Define o canal de índice {index}, contado a partir de zero, para exatamente {value}.',
    'Stel schakelaarindex {index}, geteld vanaf nul, in op precies {value}.',
    'Ustaw kanał o indeksie {index}, liczonym od zera, dokładnie na {value}.',
    'ゼロから数えたスイッチ番号{index}を正確に{value}に設定してください。',
    '把从零开始编号的开关索引{index}准确设置为{value}。'),
 'filter': (
    'Select filter slot {index}, using zero-based slot numbers.',
    'Wähle Filterplatz {index}; die Plätze sind ab null nummeriert.',
    'Sélectionne la case de filtre {index}, avec une numérotation depuis zéro.',
    'Selecciona la posición de filtro {index}, contando desde cero.',
    'Seleziona lo slot del filtro {index}, contando da zero.',
    'Seleciona a posição de filtro {index}, contando a partir de zero.',
    'Kies filterpositie {index}, met nummering vanaf nul.',
    'Wybierz pozycję filtra {index}, licząc od zera.',
    'ゼロから数えたフィルター位置{index}を選んでください。',
    '选择从零开始编号的滤镜位置{index}。'),
 'slew_hours': (
    'Point to J2000 RA {ra} hours, Dec {dec} degrees.',
    'Fahre zu J2000 RA {ra} Stunden und Dec {dec} Grad.',
    'Pointe vers J2000 : AD {ra} heures, déclinaison {dec} degrés.',
    'Apunta a J2000: AR {ra} horas, declinación {dec} grados.',
    'Punta a J2000: AR {ra} ore, declinazione {dec} gradi.',
    'Aponta para J2000: AR {ra} horas, declinação {dec} graus.',
    'Richt op J2000 RA {ra} uur, declinatie {dec} graden.',
    'Skieruj na J2000: RA {ra} godzin, deklinacja {dec} stopni.',
    'J2000の赤経{ra}時間、赤緯{dec}度へ向けてください。',
    '指向J2000赤经{ra}小时、赤纬{dec}度。'),
 'slew_degrees': (
    'My J2000 RA is {degrees} degrees, not hours; Dec is {dec} degrees. Point there.',
    'Meine J2000 RA ist {degrees} Grad, nicht Stunden; Dec ist {dec} Grad. Fahre dorthin.',
    'Mon AD J2000 est {degrees} degrés, pas des heures ; déclinaison {dec} degrés. Pointe là.',
    'Mi AR J2000 es {degrees} grados, no horas; declinación {dec} grados. Apunta allí.',
    'La mia AR J2000 è {degrees} gradi, non ore; declinazione {dec} gradi. Punta lì.',
    'A minha AR J2000 é {degrees} graus, não horas; declinação {dec} graus. Aponta para lá.',
    'Mijn J2000 RA is {degrees} graden, geen uren; declinatie {dec} graden. Richt daarop.',
    'Moje RA J2000 to {degrees} stopni, nie godzin; deklinacja {dec} stopni. Skieruj tam.',
    'J2000の赤経は時間ではなく{degrees}度、赤緯は{dec}度です。そこへ向けてください。',
    '我的J2000赤经是{degrees}度，不是小时；赤纬{dec}度。请指向那里。'),
 'recent': (
    'Show the most recent {count} images.', 'Zeige die letzten {count} Bilder.',
    'Montre les {count} images les plus récentes.', 'Muestra las últimas {count} imágenes.',
    'Mostra le ultime {count} immagini.', 'Mostra as últimas {count} imagens.',
    'Toon de laatste {count} beelden.', 'Pokaż ostatnie {count} obrazów.',
    '最新の{count}枚の画像を見せてください。', '显示最近的{count}张图像。'),
 'job': (
    'Check task {job}; wait at most {wait} seconds.',
    'Prüfe Auftrag {job}; warte höchstens {wait} Sekunden.',
    'Vérifie la tâche {job} ; attends au maximum {wait} secondes.',
    'Comprueba la tarea {job}; espera como máximo {wait} segundos.',
    'Controlla l’attività {job}; attendi al massimo {wait} secondi.',
    'Verifica a tarefa {job}; espera no máximo {wait} segundos.',
    'Controleer taak {job}; wacht maximaal {wait} seconden.',
    'Sprawdź zadanie {job}; czekaj najwyżej {wait} sekund.',
    '処理「{job}」を確認して、最大{wait}秒待ってください。',
    '检查任务“{job}”，最多等待{wait}秒。'),
 'target': (
    'Append target "{name}" at J2000 RA {ra} hours, Dec {dec} degrees: {count} frames of {seconds} seconds, keep the current filter.',
    'Füge Ziel "{name}" hinzu, J2000 RA {ra} Stunden, Dec {dec} Grad: {count} Bilder zu {seconds} Sekunden; behalte den aktuellen Filter.',
    'Ajoute la cible "{name}" à J2000 AD {ra} heures, déclinaison {dec} degrés : {count} poses de {seconds} secondes, garde le filtre actuel.',
    'Añade el objetivo "{name}" en J2000 AR {ra} horas, declinación {dec} grados: {count} tomas de {seconds} segundos, conserva el filtro actual.',
    'Aggiungi il target "{name}" a J2000 AR {ra} ore, declinazione {dec} gradi: {count} pose di {seconds} secondi, mantieni il filtro attuale.',
    'Adiciona o alvo "{name}" em J2000 AR {ra} horas, declinação {dec} graus: {count} imagens de {seconds} segundos, mantém o filtro atual.',
    'Voeg doel "{name}" toe op J2000 RA {ra} uur, declinatie {dec} graden: {count} beelden van {seconds} seconden, behoud het huidige filter.',
    'Dodaj cel "{name}" na J2000 RA {ra} godzin, deklinacji {dec} stopni: {count} klatek po {seconds} sekund, zachowaj obecny filtr.',
    '対象"{name}"をJ2000赤経{ra}時間、赤緯{dec}度で追加してください。現在のフィルターのまま{seconds}秒を{count}枚撮る計画です。',
    '添加目标"{name}"，J2000赤经{ra}小时、赤纬{dec}度：计划拍{count}张{seconds}秒曝光，保持当前滤镜。'),
 'load': (
    'Load the saved sequence named "{name}".', 'Lade die gespeicherte Sequenz "{name}".',
    'Charge la séquence enregistrée "{name}".', 'Carga la secuencia guardada "{name}".',
    'Carica la sequenza salvata "{name}".', 'Carrega a sequência guardada "{name}".',
    'Laad de opgeslagen reeks "{name}".', 'Wczytaj zapisaną sekwencję "{name}".',
    '保存済みのシーケンス"{name}"を読み込んでください。', '加载已保存的序列"{name}"。'),
 'save': (
    'Save this sequence under the new name "{name}".', 'Speichere diese Sequenz unter dem neuen Namen "{name}".',
    'Enregistre cette séquence sous le nouveau nom "{name}".', 'Guarda esta secuencia con el nuevo nombre "{name}".',
    'Salva questa sequenza con il nuovo nome "{name}".', 'Guarda esta sequência com o novo nome "{name}".',
    'Sla deze reeks op onder de nieuwe naam "{name}".', 'Zapisz tę sekwencję pod nową nazwą "{name}".',
    'このシーケンスを新しい名前"{name}"で保存してください。', '用新名称"{name}"保存这个序列。'),
}


def build_rows():
    rows = []
    for variant in range(3):
        seconds = (52.5, 105, 210)[variant]
        minutes = (1.25, 2.75, 4.5)[variant]
        position = (12673, 20849, 31507)[variant]
        temperature, duration = (-6.5, -17.25, 2.5)[variant], (3, 7, 12)[variant]
        index, value = (6, 9, 12)[variant], (37.25, 62.5, 0)[variant]
        slot = (2, 4, 7)[variant]
        ra, dec = (11.375, 16.125, 21.625)[variant], (-27.25, 8.75, -61.5)[variant]
        degrees = (199.125, 72.375, 308.625)[variant]
        count = (23, 47, 81)[variant]
        job, wait = f'job-juniper-{173 + variant * 19}', (0, 2.5, 4)[variant]
        name = ('Juniper Mosaic', 'Copper Lantern', 'Quiet Meadow')[variant]
        values = dict(seconds=seconds, minutes=minutes, position=position, temperature=temperature,
                      duration=duration, index=index, value=value, ra=ra, dec=dec,
                      degrees=degrees, count=count, job=job, wait=wait, name=name)
        fixtures = [
            ('capture_seconds', 'capture_image', {'exposure_seconds': seconds, 'save': False}, ''),
            ('capture_minutes', 'capture_image', {'exposure_seconds': minutes * 60, 'save': True}, ''),
            ('focus', 'move_focuser', {'position': position}, ''),
            ('cool', 'cool_camera', {'target_celsius': temperature, 'duration_minutes': duration}, ''),
            ('switch', 'set_switch', {'index': index, 'value': value},
             f'Synthetic status: channel {index} is writable, min 0, max 100, step 0.25.'),
            ('filter', 'change_filter', {'position': slot}, 'Synthetic status: connected filter wheel has eight slots indexed 0..7.'),
            ('slew_hours', 'slew_mount', {'ra': ra, 'dec': dec}, ''),
            ('slew_degrees', 'slew_mount', {'ra': degrees / 15, 'dec': dec}, ''),
            ('recent', 'get_recent_images', {'count': (3, 7, 11)[variant]}, ''),
            ('job', 'get_job', {'job_id': job, 'wait_seconds': wait}, ''),
            ('target', 'add_target', {'name': name, 'ra': ra, 'dec': dec,
                                    'exposures': [{'exposure_seconds': seconds, 'count': count}]}, ''),
            ('load', 'load_sequence', {'name': name}, f'Synthetic list_saved_sequences returned exact name {name}; current edits saved.'),
            ('save', 'save_sequence', {'name': name}, f'Synthetic list_saved_sequences does not contain {name}; an idle sequence is loaded.'),
        ]
        for kind, tool, args, context in fixtures:
            for language_index, locale in enumerate(LANGUAGES):
                fields = {k: str(v).replace('.', ',') if locale in ('de', 'fr', 'es', 'it', 'pt', 'nl', 'pl')
                          and isinstance(v, float) else v for k, v in values.items()}
                if kind == 'filter': fields['index'] = slot
                if kind == 'recent': fields['count'] = args['count']
                selected = [tool, COMPETITORS[tool][variant % len(COMPETITORS[tool])]]
                if variant % 2: selected.reverse()
                rows.append(conversation(f'chat-generalization-{kind}-{variant}', locale,
                    'multilingual_parameters', selected,
                    [user(TEXT[kind][language_index].format(**fields)), tool_call(tool, args)],
                    context or CONTEXT.get(tool, '')))
    # Clear imperative requests are proposals; withdrawing the same request is
    # a no-tool response. This contrast is based only on existing TRAIN phrases.
    for tool, definition in definitions().items():
        if definition['annotations']['readOnlyHint']:
            continue
        for language_index, locale in enumerate(LANGUAGES):
            rows.append(conversation('chat-generalization-withdraw-' + tool, locale,
                'multilingual_withdrawal', [tool, COMPETITORS[tool][0]],
                [user(PROMPTS[tool][language_index] + '\n' + DENIAL[locale]), assistant(DENIED_ACK[locale])],
                CONTEXT.get(tool, '')))
    # An observed pending job permits polling, not another equipment action.
    pending = ('It is still running; completion is not confirmed.',
               'Der Auftrag läuft noch; der Abschluss ist nicht bestätigt.',
               'La tâche est toujours en cours ; la fin n’est pas confirmée.',
               'La tarea sigue ejecutándose; no se ha confirmado que haya terminado.',
               'L’attività è ancora in corso; il completamento non è confermato.',
               'A tarefa ainda está em execução; a conclusão não está confirmada.',
               'De taak loopt nog; voltooiing is niet bevestigd.',
               'Zadanie nadal trwa; zakończenie nie jest potwierdzone.',
               '処理はまだ実行中です。完了は確認されていません。',
               '任务仍在运行，尚未确认完成。')
    for language_index, locale in enumerate(LANGUAGES):
        job = 'job-juniper-history-281'
        text = TEXT['job'][language_index].format(job=job, wait=0)
        rows.append(conversation('chat-generalization-pending-job', locale, 'multilingual_job_history', ['get_job'],
            [user(text), tool_call('get_job', {'job_id': job, 'wait_seconds': 0}),
             tool_result('get_job', _job(job, 'running')), assistant(pending[language_index]),
             user(text), tool_call('get_job', {'job_id': job, 'wait_seconds': 0})]))
    return rows
